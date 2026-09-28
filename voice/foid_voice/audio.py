from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator

import numpy as np

SAMPLE_RATE = 16_000
WAKE_FRAME = 1280  # 80 ms
VAD_FRAME = 512  # 32 ms


class MicStream:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block: int = VAD_FRAME):
        self.sample_rate = sample_rate
        self.block = block
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Drop queued frames so speaker bleed does not reach wake/VAD later."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(None)
                return


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


def to_int16(frame: np.ndarray) -> np.ndarray:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)
