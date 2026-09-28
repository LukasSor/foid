from __future__ import annotations

import numpy as np

from foid_voice.audio import rms

SPEECH_RMS = 0.012
SILENCE_RMS = 0.008


class VoiceActivity:
    """Silero VAD when installed, energy gate otherwise."""

    def __init__(self):
        self._iterator = None
        self.available = False
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from silero_vad import VADIterator, load_silero_vad

            model = load_silero_vad()
            self._iterator = VADIterator(model, sampling_rate=16000, threshold=0.5)
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def reset(self) -> None:
        if self._iterator is not None:
            try:
                self._iterator.reset_states()
            except Exception:
                pass

    def is_speech(self, frame: np.ndarray) -> bool:
        if self.available and self._iterator is not None:
            try:
                result = self._iterator(frame.astype(np.float32), return_seconds=False)
                if result:
                    return True
                return rms(frame) > SILENCE_RMS
            except Exception:
                pass
        return rms(frame) > SPEECH_RMS
