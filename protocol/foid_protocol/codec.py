"""Encode/decode one JSON object per line (UTF-8)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from foid_protocol.messages import parse_message


def encode(message: BaseModel | dict[str, Any]) -> bytes:
    if isinstance(message, BaseModel):
        payload = message.model_dump(mode="json")
    else:
        payload = message
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode_line(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        text = line.decode("utf-8")
    else:
        text = line
    text = text.strip()
    if not text:
        raise ValueError("empty line")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON line must be an object")
    return data


def decode_message(line: bytes | str):
    return parse_message(decode_line(line))
