from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from foid_protocol.catalog import Catalog

log = logging.getLogger("foid.llm")


class ToolLLM:
    def __init__(self, models_dir: Path, filename: str, n_gpu_layers: int | None = None):
        self.path = models_dir / filename
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self.available = False
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.error = f"LLM file missing: {self.path}"
            log.warning("LLM file missing: %s", self.path)
            return
        try:
            from llama_cpp import Llama

            layers = self.n_gpu_layers
            if layers is None:
                layers = -1
            self._llm = Llama(
                model_path=str(self.path),
                n_ctx=2048,
                n_gpu_layers=layers,
                chat_format="chatml-function-calling",
                verbose=False,
            )
            self.available = True
            log.info("loaded LLM %s", self.path.name)
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            log.warning("LLM unavailable: %s", exc)

    def interpret(self, catalog: Catalog, transcript: str) -> tuple[str, str] | None:
        if not self.available or self._llm is None:
            return None
        messages = [
            {"role": "system", "content": catalog.llm_system_prompt()},
            {"role": "user", "content": transcript},
        ]
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                tools=catalog.llm_tools(),
                tool_choice="auto",
                temperature=0.1,
                max_tokens=160,
            )
        except Exception as exc:
            log.warning("LLM call failed: %s", exc)
            return None
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            fn = call.get("function") or {}
            if fn.get("name") != "control_device":
                return None
            try:
                args = fn.get("arguments") or "{}"
                payload = json.loads(args) if isinstance(args, str) else args
                device = str(payload["device"])
                action = str(payload["action"])
            except Exception:
                return None
            if device in catalog.devices and catalog.devices[device].action(action):
                return device, action
            return None
        content = (message.get("content") or "").strip()
        return _parse_json_command(catalog, content)


def _parse_json_command(catalog: Catalog, content: str) -> tuple[str, str] | None:
    if not content:
        return None
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < 0:
            return None
        payload: dict[str, Any] = json.loads(content[start : end + 1])
        device = str(payload.get("device") or "")
        action = str(payload.get("action") or "")
        if device in catalog.devices and catalog.devices[device].action(action):
            return device, action
    except Exception:
        return None
    return None
