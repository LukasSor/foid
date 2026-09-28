from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def default_devices_dir() -> Path:
    return Path(__file__).resolve().parent / "devices"


@dataclass(frozen=True)
class DeviceAction:
    id: str
    aliases: tuple[str, ...]
    set: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    clamp: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    ui: str
    aliases: tuple[str, ...]
    defaults: dict[str, Any]
    actions: tuple[DeviceAction, ...]

    def action(self, action_id: str) -> DeviceAction | None:
        for item in self.actions:
            if item.id == action_id:
                return item
        return None

    def all_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for name in (self.name, *self.aliases, self.id):
            key = name.strip()
            if key and key not in seen:
                seen.append(key)
        return tuple(seen)


@dataclass
class Phrase:
    text: str
    device_id: str
    action_id: str


class Catalog:
    def __init__(self, devices: list[Device]):
        self.devices = {device.id: device for device in devices}
        self.order = [device.id for device in devices]
        self.phrases = tuple(_build_phrases(devices))

    def get(self, device_id: str) -> Device:
        try:
            return self.devices[device_id]
        except KeyError as exc:
            raise KeyError(f"unknown device: {device_id}") from exc

    def initial_state(self) -> dict[str, dict[str, Any]]:
        return {device_id: dict(device.defaults) for device_id, device in self.devices.items()}

    def apply(self, state: dict[str, dict[str, Any]], device_id: str, action_id: str) -> dict[str, Any]:
        device = self.get(device_id)
        action = device.action(action_id)
        if action is None:
            raise KeyError(f"unknown action {action_id!r} for device {device_id}")
        current = dict(state.get(device_id, dict(device.defaults)))
        current.update(action.set)
        for key, amount in action.delta.items():
            current[key] = float(current.get(key, 0)) + float(amount)
        for key, bounds in action.clamp.items():
            low, high = bounds
            current[key] = min(high, max(low, float(current.get(key, 0))))
        state[device_id] = current
        return current

    def whisper_prompt(self) -> str:
        names = []
        for device_id in self.order:
            device = self.devices[device_id]
            names.append(device.name)
            names.extend(device.aliases[:4])
            for action in device.actions:
                names.extend(action.aliases[:4])
        unique: list[str] = []
        for name in names:
            if name not in unique:
                unique.append(name)
        return " ".join(unique[:80])

    def llm_tools(self) -> list[dict[str, Any]]:
        device_enum = list(self.order)
        action_enum = sorted({action.id for device in self.devices.values() for action in device.actions})
        lines = []
        for device_id in self.order:
            device = self.devices[device_id]
            acts = ", ".join(f"{action.id} ({action.aliases[0]})" for action in device.actions)
            lines.append(f"- {device.id} ({device.name}): {acts}")
        description = (
            "Control a device in the room. Use only existing devices and actions.\n"
            + "\n".join(lines)
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "control_device",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device": {"type": "string", "enum": device_enum},
                            "action": {"type": "string", "enum": action_enum},
                        },
                        "required": ["device", "action"],
                    },
                },
            }
        ]

    def llm_system_prompt(self) -> str:
        return (
            "You are the controller of a room. "
            "Never reply with chit-chat. "
            "Always call the control_device tool when the user wants to operate a device. "
            "If the command is unclear, call nothing."
        )


def load_catalog(devices_dir: Path | None = None) -> Catalog:
    if devices_dir is None:
        directory = Path(os.environ["FOID_DEVICES_DIR"]) if os.environ.get("FOID_DEVICES_DIR") else default_devices_dir()
    else:
        directory = devices_dir
    paths = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not paths:
        raise FileNotFoundError(f"no device YAML files in {directory}")
    devices = [_load_device(path) for path in paths]
    return Catalog(devices)


def _as_str(value: Any) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


def _stringify_map(raw: dict[Any, Any] | None) -> dict[str, Any]:
    return {_as_str(key): value for key, value in (raw or {}).items()}


def _load_device(path: Path) -> Device:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    actions = []
    for item in raw.get("actions") or []:
        clamp = {}
        for key, bounds in (item.get("clamp") or {}).items():
            clamp[_as_str(key)] = (float(bounds[0]), float(bounds[1]))
        actions.append(
            DeviceAction(
                id=_as_str(item["id"]),
                aliases=tuple(_as_str(alias) for alias in item.get("aliases") or [item["id"]]),
                set=_stringify_map(item.get("set")),
                delta={_as_str(k): float(v) for k, v in (item.get("delta") or {}).items()},
                clamp=clamp,
            )
        )
    return Device(
        id=_as_str(raw["id"]),
        name=str(raw.get("name") or raw["id"]),
        ui=str(raw.get("ui") or raw["id"]),
        aliases=tuple(_as_str(alias) for alias in raw.get("aliases") or []),
        defaults=_stringify_map(raw.get("defaults")),
        actions=tuple(actions),
    )


_PREFIXES = ("", "mach ", "mache ", "mach das ", "das ", "bitte ", "sei so gut ", "kannst du ")
_SUFFIXES = ("", " bitte")


def _build_phrases(devices: list[Device]) -> list[Phrase]:
    phrases: list[Phrase] = []
    seen: set[tuple[str, str, str]] = set()
    for device in devices:
        for action in device.actions:
            for device_name in device.all_names():
                for action_name in action.aliases:
                    bases = (
                        f"{device_name} {action_name}",
                        f"{action_name} {device_name}",
                        f"{device_name} nach {action_name}",
                    )
                    for base in bases:
                        for prefix in _PREFIXES:
                            for suffix in _SUFFIXES:
                                text = f"{prefix}{base}{suffix}".strip()
                                key = (text, device.id, action.id)
                                if key in seen:
                                    continue
                                seen.add(key)
                                phrases.append(Phrase(text=text, device_id=device.id, action_id=action.id))
    return phrases
