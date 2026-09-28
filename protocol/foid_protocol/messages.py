"""Wire message models for the room-control JSON/TCP protocol."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Command(BaseModel):
    type: Literal["command"] = "command"
    id: str
    device: str
    action: str
    extra: dict[str, Any] = Field(default_factory=dict)


class Ack(BaseModel):
    type: Literal["ack"] = "ack"
    id: str
    ok: bool
    error: str | None = None
    state: dict[str, Any] | None = None


class StateEvent(BaseModel):
    type: Literal["state"] = "state"
    device: str
    state: dict[str, Any]
    all: dict[str, dict[str, Any]] | None = None


class Snapshot(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    devices: dict[str, dict[str, Any]]


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    role: str = "voice"


def parse_message(data: dict[str, Any]) -> Command | Ack | StateEvent | Snapshot | Hello:
    kind = data.get("type")
    mapping = {
        "command": Command,
        "ack": Ack,
        "state": StateEvent,
        "snapshot": Snapshot,
        "hello": Hello,
    }
    cls = mapping.get(kind)
    if cls is None:
        raise ValueError(f"unknown message type: {kind!r}")
    return cls.model_validate(data)
