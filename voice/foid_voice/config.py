"""Persisted voice-box settings (JSON under FOID_CONFIG or ~/.config/foid)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from foid_protocol.i18n import DEFAULT_LOCALE, normalize_locale
from foid_voice.stt_profiles import DEFAULT_STT_LANGUAGE, DEFAULT_STT_MODEL, migrate_legacy_profile, resolve_stt

WakeWord = str
Locale = Literal["en", "de"]
_WAKE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def default_config_path() -> Path:
    env = os.environ.get("FOID_CONFIG")
    if env:
        return Path(env)
    return Path.home() / ".config" / "foid" / "settings.json"


def default_models_dir() -> Path:
    env = os.environ.get("FOID_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "foid" / "models"


class VoiceSettings(BaseModel):
    locale: Locale = DEFAULT_LOCALE
    wake_word: WakeWord = "computer"
    stt_language: str = DEFAULT_STT_LANGUAGE
    stt_model: str = DEFAULT_STT_MODEL
    silence_ms: int = Field(default=450, ge=250, le=900)
    beam_size: int = Field(default=1, ge=1, le=5)
    listen_timeout_s: float = Field(default=6.0, ge=3.0, le=12.0)
    earcon_volume: float = Field(default=0.7, ge=0.0, le=1.0)
    room_host: str = "127.0.0.1"
    room_port: int = 9500
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    models_dir: Path = Field(default_factory=default_models_dir)
    device: str = "auto"
    llm_filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    match_threshold: float = 82.0
    llm_threshold: float = 70.0

    @field_validator("locale", mode="before")
    @classmethod
    def _coerce_locale(cls, value: Any) -> str:
        return normalize_locale(value)

    @model_validator(mode="before")
    @classmethod
    def _migrate_stt_profile(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        has_language = bool(str(payload.get("stt_language") or "").strip())
        has_model = bool(str(payload.get("stt_model") or "").strip())
        if has_language and has_model:
            return payload
        migrated = migrate_legacy_profile(payload.get("stt_profile"))
        if migrated is None:
            return payload
        language, model = migrated
        payload.setdefault("stt_language", language)
        payload.setdefault("stt_model", model)
        return payload

    @model_validator(mode="after")
    def _resolve_stt(self) -> VoiceSettings:
        language, model = resolve_stt(self.stt_language, self.stt_model)
        self.stt_language = language
        self.stt_model = model
        return self

    @field_validator("wake_word", mode="before")
    @classmethod
    def _coerce_wake_word(cls, value: Any) -> str:
        text = str(value or "computer").strip()
        if not text:
            return "computer"
        path = Path(text)
        if path.suffix.lower() in {".onnx", ".tflite"}:
            text = path.stem
        text = text.strip().lower().replace(" ", "_")
        if not _WAKE_ID.fullmatch(text):
            raise ValueError("wake_word must be a simple id or .onnx stem")
        return text

    def save(self, path: Path | None = None) -> Path:
        target = path or default_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> VoiceSettings:
        target = Path(path) if path is not None else default_config_path()
        if not target.exists():
            settings = cls()
            settings.save(target)
            return settings
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def merged(self, **changes: Any) -> VoiceSettings:
        payload = self.model_dump()
        payload.update(changes)
        return VoiceSettings.model_validate(payload)
