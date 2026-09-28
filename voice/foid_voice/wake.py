from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from foid_voice.audio import WAKE_FRAME, to_int16
from foid_voice.config import default_models_dir
from foid_voice.wake_models import model_by_id, resolve_wake_model, wake_dir

log = logging.getLogger("foid.wake")

WAKE_THRESHOLD = 0.5
# reset() plants random features; a couple of silence frames knock them down.
# Do not prime a full embedding window here — that is ~1.6s of blocking work
# and makes the first real wake look like a ghost. Speaker audio is dropped
# at the mic during playback instead.
PRIME_FRAMES = 3


class WakeDetector:
    def __init__(self, wake_word: str = "computer", models_dir: Path | None = None):
        self.wake_word = wake_word
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self._model = None
        self.available = False
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from openwakeword.model import Model

            model_ref = resolve_wake_model(self.wake_word, self.models_dir)
            model_path = Path(model_ref)
            if model_path.suffix.lower() == ".onnx" and not model_path.is_file():
                hint = wake_dir(self.models_dir)
                extra = ""
                spec = model_by_id(self.wake_word)
                if spec is not None:
                    extra = " save settings to download it, or"
                raise FileNotFoundError(
                    f"wake model {self.wake_word!r} not found at {model_path};"
                    f"{extra} drop a custom .onnx into {hint}"
                )

            self._model = Model(
                wakeword_models=[model_ref],
                inference_framework="onnx",
                vad_threshold=0.5,
                enable_speex_noise_suppression=False,
            )
            self.available = True
            self.error = None
            log.info("wake word loaded: %s (%s)", self.wake_word, model_ref)
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            log.warning("wake word unavailable: %s", exc)

    def reload(self, wake_word: str, models_dir: Path | None = None) -> None:
        self.wake_word = wake_word
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        self._load()

    def reset(self) -> None:
        """Drop openWakeWord mel/embedding/prediction buffers so speaker echo cannot retrigger."""
        if self._model is None:
            return
        try:
            self._model.reset()
        except Exception as exc:
            log.debug("wake model reset failed: %s", exc)
            return
        vad = getattr(self._model, "vad", None)
        if vad is None:
            return
        try:
            if hasattr(vad, "reset_states"):
                vad.reset_states()
            elif hasattr(vad, "reset"):
                vad.reset()
        except Exception:
            pass

    def prime(self) -> None:
        """Reset, then flush the embedding window with silence so random features cannot score as a wake."""
        self.reset()
        if not self.available or self._model is None:
            return
        silence = np.zeros(WAKE_FRAME, dtype=np.float32)
        for _ in range(PRIME_FRAMES):
            try:
                self.predict(silence)
            except Exception:
                break

    def predict(self, frame: np.ndarray) -> float:
        if not self.available or self._model is None:
            return 0.0
        pcm = to_int16(frame)
        scores = self._model.predict(pcm)
        if not scores:
            return 0.0
        return float(max(scores.values()))

    def triggered(self, frame: np.ndarray) -> bool:
        return self.predict(frame) >= WAKE_THRESHOLD
