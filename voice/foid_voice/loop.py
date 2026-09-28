from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from foid_protocol.catalog import Catalog
from foid_voice.audio import SAMPLE_RATE, VAD_FRAME, WAKE_FRAME, MicStream
from foid_voice.config import VoiceSettings
from foid_voice.llm import ToolLLM
from foid_voice.nlu import HybridNLU, Intent
from foid_voice.room_client import RoomLink
from foid_voice.sounds import earcon_muted, earcon_playing, ensure_sounds, play
from foid_voice.stt import SpeechToText, SpeechToTextBusy
from foid_voice.vad import VoiceActivity
from foid_voice.wake import WAKE_THRESHOLD, WakeDetector

log = logging.getLogger("foid.loop")
BUSY_PHASES = {"listening", "understanding", "sending"}


class VoiceBox:
    def __init__(self, settings: VoiceSettings, catalog: Catalog):
        self.settings = settings
        self.catalog = catalog
        self.status: dict[str, Any] = {
            "phase": "idle",
            "transcript": "",
            "last_intent": None,
            "wake": "loading",
            "stt": "loading",
            "llm": "loading",
            "room": "offline",
            "error": None,
        }
        ensure_sounds()
        self.wake = WakeDetector(settings.wake_word, settings.models_dir)
        self.vad = VoiceActivity()
        self.stt = SpeechToText(
            settings.stt_model,
            language=settings.stt_language,
            device=settings.device,
            beam_size=settings.beam_size,
            prompt=catalog.whisper_prompt(),
        )
        self.llm = ToolLLM(settings.models_dir, settings.llm_filename)
        self.nlu = HybridNLU(catalog, self.llm, settings.match_threshold, settings.llm_threshold)
        self.room = RoomLink(settings.room_host, settings.room_port)
        self._wake_event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending_wake = False
        self._wake_dirty = False
        self._update_component_status()

    def _update_component_status(self) -> None:
        self.status["wake"] = "ok" if self.wake.available else "missing"
        self.status["stt"] = "ok" if self.stt.available else "missing"
        self.status["llm"] = "ok" if self.llm.available else "missing"
        room_up = self.room.probe()
        self.status["room"] = "connected" if room_up else "offline"
        self.status["detail"] = {
            "wake": self.wake.error,
            "stt": self.stt.error,
            "llm": self.llm.error,
            "room": None if room_up else f"{self.room.host}:{self.room.port} unreachable",
        }

    def stop(self) -> None:
        self._stop.set()
        self._wake_event.set()
        self.room.close()

    def _busy(self) -> bool:
        return self.status.get("phase") in BUSY_PHASES

    def _can_accept_wake(self) -> bool:
        # Only block on a live listen/STT/dispatch, or the short speaker-tail mute.
        if self._stop.is_set() or self._busy() or earcon_muted():
            return False
        return True

    def _queue_or_trigger_wake(self) -> None:
        if self._can_accept_wake():
            log.info("wake accepted (api)")
            self._wake_event.set()
            return
        if self._busy():
            log.info("wake ignored (already %s)", self.status.get("phase"))
            return
        self._pending_wake = True
        log.info("wake deferred until speaker tail ends")

    def trigger_wake(self) -> None:
        self._queue_or_trigger_wake()

    def _play_cue(self, name: str) -> None:
        play(name, self.settings.earcon_volume)
        self._wake_dirty = True

    def _play_fail(self, reason: str) -> None:
        log.info("fail cue (%s)", reason)
        self._play_cue("fail")

    def _arm_wake(self) -> None:
        self.wake.prime()
        self._wake_dirty = False

    def _reset_wake_once(self) -> None:
        if not self._wake_dirty:
            return
        self._arm_wake()
        log.info("wake model reset after earcon")

    def _idle(self) -> None:
        self.status["phase"] = "idle"

    def _settle_after_turn(self, mic: MicStream) -> None:
        """Drop the speaker tail, then re-arm wake. No extra ignore window."""
        self._drop_while_muted(mic)
        mic.drain()
        self._reset_wake_once()
        self._idle()
        log.info("settled idle; listening for wake")

    def apply_settings(self, settings: VoiceSettings) -> None:
        with self._lock:
            old = self.settings
            self.settings = settings
            self.room.configure(settings.room_host, settings.room_port)
            if old.wake_word != settings.wake_word or old.models_dir != settings.models_dir:
                self.wake.reload(settings.wake_word, settings.models_dir)
            if (
                old.stt_model != settings.stt_model
                or old.stt_language != settings.stt_language
                or old.device != settings.device
                or old.beam_size != settings.beam_size
            ):
                self.stt.configure(
                    settings.stt_model,
                    settings.stt_language,
                    settings.device,
                    settings.beam_size,
                    self.catalog.whisper_prompt(),
                )
            self.nlu.match_threshold = settings.match_threshold
            self.nlu.llm_threshold = settings.llm_threshold
            self._update_component_status()

    def handle_text(self, text: str) -> dict[str, Any]:
        try:
            intent = self.nlu.interpret(text)
            return self._dispatch(intent, text)
        except Exception as exc:
            log.exception("text command failed")
            self._play_fail("text command")
            self._idle()
            self.status["error"] = str(exc)
            return {"ok": False, "transcript": text, "error": str(exc)}

    def record_seconds(self, seconds: float = 3.0) -> np.ndarray:
        mic = MicStream()
        mic.start()
        chunks: list[np.ndarray] = []
        deadline = time.time() + seconds
        try:
            while time.time() < deadline:
                frame = mic.read(timeout=0.3)
                if frame is not None:
                    chunks.append(frame)
        finally:
            mic.stop()
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        return self.stt.transcribe(audio, wait=True)

    def run_forever(self) -> None:
        mic = MicStream()
        try:
            mic.start()
        except Exception as exc:
            self.status["error"] = f"microphone: {exc}"
            log.exception("microphone failed")
            while not self._stop.is_set():
                if self._wake_event.wait(timeout=0.25):
                    self._wake_event.clear()
                time.sleep(0.2)
            return

        preroll: deque[np.ndarray] = deque(maxlen=int(0.4 * SAMPLE_RATE / VAD_FRAME))
        wake_buffer = np.zeros(0, dtype=np.float32)
        self.status["phase"] = "idle"
        log.info("voice loop ready")
        try:
            while not self._stop.is_set():
                try:
                    wake_buffer = self._idle_tick(mic, preroll, wake_buffer)
                except Exception as exc:
                    log.exception("voice loop error")
                    self._play_fail("voice loop")
                    self.status["error"] = str(exc)
                    self._idle()
                    wake_buffer = np.zeros(0, dtype=np.float32)
                    preroll.clear()
                    time.sleep(0.2)
        finally:
            mic.stop()

    def _take_pending_wake(self) -> bool:
        if self._pending_wake and self._can_accept_wake():
            self._pending_wake = False
            log.info("wake accepted (deferred)")
            return True
        return False

    def _drop_while_playing(self, mic: MicStream) -> None:
        """Drop mic frames until the earcon WAV is done — do not wait the echo tail."""
        while not self._stop.is_set() and earcon_playing():
            mic.read(timeout=0.05)
        mic.drain()

    def _drop_while_muted(self, mic: MicStream) -> None:
        """Throw away frames while the speaker is playing and during the echo tail."""
        while not self._stop.is_set() and earcon_muted():
            mic.read(timeout=0.05)
        mic.drain()

    def _start_utterance(self, mic: MicStream, preroll: deque[np.ndarray]) -> None:
        self._wake_event.clear()
        self.status["phase"] = "listening"
        # After the wake chime, listen immediately so the spoken command is not eaten.
        self._drop_while_playing(mic)
        preroll.clear()
        captured = self._listen(mic, [])
        self._process_utterance(captured)
        self._settle_after_turn(mic)
        preroll.clear()

    def _idle_tick(
        self, mic: MicStream, preroll: deque[np.ndarray], wake_buffer: np.ndarray
    ) -> np.ndarray:
        if not self._can_accept_wake():
            mic.read(timeout=0.05)
            preroll.clear()
            return np.zeros(0, dtype=np.float32)
        if self._take_pending_wake() or self._wake_event.is_set():
            self._start_utterance(mic, preroll)
            return np.zeros(0, dtype=np.float32)
        frame = mic.read(timeout=0.2)
        if frame is None:
            return wake_buffer
        preroll.append(frame)
        wake_buffer = np.concatenate([wake_buffer, frame])
        while wake_buffer.size >= WAKE_FRAME:
            chunk = wake_buffer[:WAKE_FRAME]
            wake_buffer = wake_buffer[WAKE_FRAME:]
            try:
                score = self.wake.predict(chunk)
            except Exception as exc:
                log.warning("wake detect failed: %s", exc)
                continue
            if score < WAKE_THRESHOLD:
                continue
            log.info("wake accepted score=%.2f", score)
            self._play_cue("wake")
            self._start_utterance(mic, preroll)
            return np.zeros(0, dtype=np.float32)
        return wake_buffer

    def _listen(self, mic: MicStream, preroll: list[np.ndarray]) -> np.ndarray:
        self.status["phase"] = "listening"
        self._wake_event.clear()
        self.vad.reset()
        chunks = list(preroll)
        speech_seen = False
        silent_ms = 0
        started = time.monotonic()
        timeout = self.settings.listen_timeout_s
        need_silence = self.settings.silence_ms
        min_speech = 280
        speech_ms = 0
        log.info("listen start timeout=%.1fs", timeout)
        while not self._stop.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                log.info("wake ignored (already listening)")
            if time.monotonic() - started > timeout:
                log.info("listen timeout after %.0f ms speech", speech_ms)
                break
            frame = mic.read(timeout=0.2)
            if frame is None:
                continue
            chunks.append(frame)
            frame_ms = 1000.0 * frame.size / SAMPLE_RATE
            speaking = self.vad.is_speech(frame)
            if speaking:
                speech_seen = True
                speech_ms += frame_ms
                silent_ms = 0
            elif speech_seen:
                silent_ms += frame_ms
                if speech_ms >= min_speech and silent_ms >= need_silence:
                    break
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        log.info(
            "listen end speech_ms=%.0f audio_s=%.2f",
            speech_ms,
            audio.size / float(SAMPLE_RATE),
        )
        return audio

    def _process_utterance(self, audio: np.ndarray) -> None:
        if audio.size < SAMPLE_RATE * 0.25:
            self.status["transcript"] = ""
            log.info("transcript '' (too short)")
            self._play_fail("empty audio")
            return
        self.status["phase"] = "understanding"
        try:
            text = self.stt.transcribe(audio, wait=False)
        except SpeechToTextBusy:
            log.info("dropped overlapping utterance; STT busy")
            self._play_fail("stt busy")
            return
        except Exception as exc:
            log.warning("transcribe failed: %s", exc)
            self._play_fail("transcribe")
            self.status["error"] = str(exc)
            return
        log.info("transcript %r", text)
        self.status["transcript"] = text
        if not text.strip():
            self._play_fail("empty transcript")
            return
        try:
            intent = self.nlu.interpret(text)
            self._dispatch(intent, text)
        except Exception as exc:
            log.exception("utterance handling failed")
            self._play_fail("utterance")
            self.status["error"] = str(exc)

    def _dispatch(self, intent: Intent | None, transcript: str) -> dict[str, Any]:
        self.status["transcript"] = transcript
        if intent is None:
            log.info("intent none transcript=%r", transcript)
            self._play_fail("not understood")
            self.status["last_intent"] = None
            self._idle()
            return {"ok": False, "transcript": transcript, "error_key": "status.not_understood"}
        log.info(
            "intent device=%s action=%s source=%s score=%.1f",
            intent.device,
            intent.action,
            intent.source,
            intent.score,
        )
        self.status["phase"] = "sending"
        self.status["last_intent"] = {
            "device": intent.device,
            "action": intent.action,
            "source": intent.source,
            "score": intent.score,
        }
        try:
            ack = self.room.command(intent.device, intent.action)
        except Exception as exc:
            log.warning("room command failed: %s", exc)
            self._play_fail("room command")
            self.status["error"] = str(exc)
            self.status["room"] = "offline"
            self._idle()
            return {"ok": False, "transcript": transcript, "error": str(exc), "intent": self.status["last_intent"]}
        log.info(
            "room ack ok=%s id=%s error=%s",
            ack.ok,
            ack.id,
            ack.error,
        )
        if ack.ok:
            self._play_cue("ok")
            self.status["room"] = "connected"
            self.status["error"] = None
        else:
            self._play_fail("room nack")
            self.status["error"] = ack.error
        self._idle()
        return {
            "ok": ack.ok,
            "transcript": transcript,
            "intent": self.status["last_intent"],
            "state": ack.state,
            "error": ack.error,
        }
