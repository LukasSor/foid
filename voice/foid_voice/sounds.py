from __future__ import annotations

import contextlib
import math
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
SOUNDS = ("wake", "listening", "heard", "timeout", "ok", "fail")
# Speaker-tail only: DAC drain / room bounce after the WAV actually finished.
# Long enough to stop the ok/fail chime from retriggering wake, short enough
# that the next spoken "Computer" is not treated as a ghost.
ECHO_MUTE_S = 0.35
_play_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
_player_started = False
_player_lock = threading.Lock()
_mute_lock = threading.Lock()
_inflight = 0
_queued_samples = 0
_mute_until = 0.0
_last_play_end = 0.0
_play_idle = threading.Event()
_play_idle.set()


def sounds_dir() -> Path:
    return Path(__file__).resolve().parent / "static" / "sounds"


def ensure_sounds() -> Path:
    directory = sounds_dir()
    directory.mkdir(parents=True, exist_ok=True)
    generators = {
        "wake": _wake,
        "listening": _listening,
        "heard": _heard,
        "timeout": _timeout,
        "ok": _ok,
        "fail": _fail,
    }
    for name, gen in generators.items():
        path = directory / f"{name}.wav"
        if not path.exists():
            _write_wav(path, gen())
    return directory


def play(name: str, volume: float = 0.7) -> None:
    path = sounds_dir() / f"{name}.wav"
    if not path.exists():
        ensure_sounds()
    samples = _read_wav(path) * float(volume)
    _ensure_player()
    _submit(samples)


def earcon_playing() -> bool:
    """True while an earcon is still queued or in the player thread."""
    with _mute_lock:
        return _inflight > 0


def earcon_muted() -> bool:
    """True while an earcon is queued/playing or during the post-playback echo window."""
    with _mute_lock:
        return _inflight > 0 or time.monotonic() < _mute_until


def last_earcon_end() -> float:
    """Monotonic time when the player thread last finished a buffer (0 if none yet)."""
    with _mute_lock:
        return _last_play_end


def earcon_mute_until() -> float:
    """Monotonic deadline: speaker-quiet plus echo tail.

    While a buffer is still in the player thread, remaining WAV length is only a
    lower bound — mute stays asserted via ``_inflight`` until play actually ends.
    """
    with _mute_lock:
        if _inflight <= 0:
            return _mute_until
        playback = _queued_samples / float(SAMPLE_RATE)
        return max(_mute_until, time.monotonic() + playback + ECHO_MUTE_S)


def _submit(samples: np.ndarray) -> None:
    global _inflight, _queued_samples
    with _mute_lock:
        try:
            _play_queue.put_nowait(samples)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                dropped = _play_queue.get_nowait()
                if dropped is not None:
                    _inflight = max(0, _inflight - 1)
                    _queued_samples = max(0, _queued_samples - int(dropped.size))
            try:
                _play_queue.put_nowait(samples)
            except queue.Full:
                return
        _inflight += 1
        _queued_samples += int(samples.size)
        _play_idle.clear()


def _ensure_player() -> None:
    global _player_started
    with _player_lock:
        if _player_started:
            return
        _player_started = True
        threading.Thread(target=_player_loop, name="foid-earcon", daemon=True).start()


def _player_loop() -> None:
    global _inflight, _queued_samples, _mute_until, _last_play_end
    while True:
        samples = _play_queue.get()
        if samples is None:
            return
        try:
            import sounddevice as sd

            sd.play(samples.astype(np.float32), SAMPLE_RATE, blocking=True)
            sd.wait()
        except Exception:
            pass
        now = time.monotonic()
        with _mute_lock:
            _inflight = max(0, _inflight - 1)
            _queued_samples = max(0, _queued_samples - int(samples.size))
            _last_play_end = now
            # Echo window starts when the player thread finished, not when play() queued.
            _mute_until = now + ECHO_MUTE_S
            if _inflight <= 0:
                _play_idle.set()


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return data


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm_i.tobytes())


def _time(duration: float) -> tuple[np.ndarray, int]:
    n = max(1, int(round(SAMPLE_RATE * duration)))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return t, n


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(max(n, 1), dtype=np.float32)
    if n <= 2:
        env[0] = 0.0
        env[-1] = 0.0
        return env
    a = max(1, int(round(attack * SAMPLE_RATE)))
    r = max(1, int(round(release * SAMPLE_RATE)))
    if a + r >= n:
        a = max(1, n // 5)
        r = max(1, n - a - 1)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _mix(*events: tuple[float, np.ndarray]) -> np.ndarray:
    prepared: list[tuple[int, np.ndarray]] = []
    end = 0
    for offset, samples in events:
        start = int(round(offset * SAMPLE_RATE))
        prepared.append((start, samples))
        end = max(end, start + samples.size)
    out = np.zeros(end, dtype=np.float32)
    for start, samples in prepared:
        out[start : start + samples.size] += samples
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out *= np.float32(0.95 / peak)
    return out


def _bell(freq: float, duration: float, *, attack: float = 0.008, release: float | None = None, brightness: float = 0.22) -> np.ndarray:
    """Soft sine with a short bell-like harmonic that decays faster than the fundamental."""
    t, n = _time(duration)
    rel = duration * 0.62 if release is None else release
    fund = np.sin(2 * math.pi * freq * t)
    h2 = brightness * np.sin(2 * math.pi * freq * 2 * t)
    h3 = brightness * 0.35 * np.sin(2 * math.pi * freq * 3 * t)
    harmonic_env = np.exp(-t * 14.0).astype(np.float32)
    wave_ = (fund + (h2 + h3) * harmonic_env).astype(np.float32)
    return wave_ * _envelope(n, attack, rel)


def _sweep(f0: float, f1: float, duration: float, *, attack: float = 0.004, release: float = 0.03) -> np.ndarray:
    t, n = _time(duration)
    span = t[-1] if t[-1] > 0 else 1.0
    freq = f0 + (f1 - f0) * (t / span)
    phase = np.cumsum(2 * math.pi * freq / SAMPLE_RATE)
    return (np.sin(phase) * _envelope(n, attack, release)).astype(np.float32)


def _click(duration: float = 0.08) -> np.ndarray:
    """Soft woodblock: noisy attack plus a damped mid resonance."""
    rng = np.random.default_rng(7)
    t, n = _time(duration)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(noise, prepend=noise[:1])
    noise_peak = float(np.max(np.abs(noise))) or 1.0
    noise /= noise_peak
    wood = np.sin(2 * math.pi * 980.0 * t) * np.exp(-t * 58.0)
    body = np.sin(2 * math.pi * 490.0 * t) * np.exp(-t * 40.0) * 0.35
    env = np.exp(-t * 38.0).astype(np.float32) * _envelope(n, 0.001, duration * 0.4)
    mixed = (0.42 * noise + 0.9 * wood + body) * env
    peak = float(np.max(np.abs(mixed))) or 1.0
    return (mixed / peak).astype(np.float32)


def _lowpass(samples: np.ndarray, taps: int) -> np.ndarray:
    kernel = np.linspace(1.0, 0.05, max(4, taps), dtype=np.float32)
    kernel /= float(np.sum(kernel))
    return np.convolve(samples, kernel, mode="same").astype(np.float32)


def _glass_tone(freq: float, duration: float, *, attack: float = 0.014, air: float = 0.06) -> np.ndarray:
    """Inharmonic glassy bar: bell partials plus a quiet high shimmer."""
    t, n = _time(duration)
    specs = (
        (1.00, 1.00, 5.2),
        (2.01, 0.30, 8.8),
        (2.76, 0.22, 13.5),
        (5.40, 0.11, 19.0),
        (8.93, 0.04, 26.0),
    )
    wave_ = np.zeros(n, dtype=np.float32)
    nyq = SAMPLE_RATE * 0.45
    for ratio, amp, decay in specs:
        partial = freq * ratio
        if partial >= nyq:
            continue
        wave_ += (amp * np.sin(2 * math.pi * partial * t) * np.exp(-t * decay)).astype(np.float32)
    shimmer_f = min(freq * 6.7, nyq * 0.92)
    wave_ += np.float32(air) * np.sin(2 * math.pi * shimmer_f * t) * np.exp(-t * 18.0)
    return (wave_ * _envelope(n, attack, duration * 0.72)).astype(np.float32)


def _mallet(duration: float = 0.032) -> np.ndarray:
    """Soft wooden strike — noise only, no pitched click."""
    rng = np.random.default_rng(5)
    t, n = _time(duration)
    noise = _lowpass(rng.standard_normal(n).astype(np.float32), 22)
    env = np.exp(-t * 88.0).astype(np.float32) * _envelope(n, 0.003, duration * 0.55)
    return (noise * env).astype(np.float32)


def _fm_sparkle(freq: float, duration: float, *, index: float = 4.6) -> np.ndarray:
    """Short digital ping: FM sidebands, staccato, not a ringing bell."""
    t, n = _time(duration)
    I = np.float32(index) * np.exp(-t * 40.0)
    modulator = np.sin(2 * math.pi * freq * 2.01 * t)
    carrier = np.sin(2 * math.pi * freq * t + I * modulator)
    edge = 0.16 * np.sin(2 * math.pi * freq * 3 * t) * np.exp(-t * 52.0)
    env = _envelope(n, 0.005, duration * 0.36) * np.exp(-t * 15.0)
    return ((carrier + edge) * env).astype(np.float32)


def _thud(duration: float = 0.11) -> np.ndarray:
    """Dull low-frequency hit: filtered noise plus a sub body."""
    rng = np.random.default_rng(19)
    t, n = _time(duration)
    noise = _lowpass(rng.standard_normal(n).astype(np.float32), 72)
    body = np.sin(2 * math.pi * 72.0 * t) * np.exp(-t * 26.0)
    sub = np.sin(2 * math.pi * 46.0 * t) * np.exp(-t * 15.0) * 0.72
    env = np.exp(-t * 20.0).astype(np.float32) * _envelope(n, 0.010, duration * 0.55)
    mixed = (0.50 * noise + 0.95 * body + sub) * env
    peak = float(np.max(np.abs(mixed))) or 1.0
    return (mixed / peak).astype(np.float32)


def _muted_brass(f0: float, f1: float, duration: float) -> np.ndarray:
    """Descending muted buzz: band-limited odd harmonics, no high sparkle."""
    t, n = _time(duration)
    span = t[-1] if t[-1] > 0 else 1.0
    freq = f0 + (f1 - f0) * (t / span)
    phase = np.cumsum(2 * math.pi * freq / SAMPLE_RATE)
    brass = np.sin(phase) + 0.34 * np.sin(3.0 * phase) + 0.10 * np.sin(5.0 * phase)
    roughness = 0.88 + 0.12 * np.sin(2 * math.pi * 15.0 * t)
    return (brass * roughness * _envelope(n, 0.018, duration * 0.52)).astype(np.float32)


def _wake() -> np.ndarray:
    # High glassy two-tone marimba (A5 then E6), airy overlap, ~400ms.
    strike = _mallet(0.030) * 0.14
    a5 = _glass_tone(880.00, 0.28, air=0.11) * 0.48
    e6 = _glass_tone(1318.51, 0.28, air=0.14) * 0.42
    return _mix((0.0, strike), (0.0, a5), (0.12, strike * 0.85), (0.12, e6))


def _listening() -> np.ndarray:
    # One high rising tick — open-ear chirp, not a second chime.
    return _sweep(1500.0, 2400.0, 0.12, attack=0.003, release=0.045) * 0.32


def _heard() -> np.ndarray:
    return _click(0.08) * 0.52


def _timeout() -> np.ndarray:
    # Falling two-note, lower than wake: E4 then A3, ~250ms.
    e4 = _bell(329.63, 0.13, release=0.08, brightness=0.12) * 0.46
    a3 = _bell(220.00, 0.15, release=0.10, brightness=0.10) * 0.40
    return _mix((0.0, e4), (0.11, a3))


def _ok() -> np.ndarray:
    # Bright 4-note major sparkle (FM collect), mid register, ~350ms.
    notes = (523.25, 659.25, 783.99, 1046.50)
    durs = (0.080, 0.080, 0.080, 0.155)
    amps = (0.50, 0.54, 0.58, 0.64)
    offs = (0.00, 0.065, 0.130, 0.195)
    idxs = (3.8, 3.5, 3.2, 2.2)
    return _mix(
        *((off, _fm_sparkle(freq, dur, index=idx) * amp) for freq, dur, amp, off, idx in zip(notes, durs, amps, offs, idxs))
    )


def _fail() -> np.ndarray:
    # Dull thud plus descending muted brass, low and slow, ~280ms.
    thud = _thud(0.11) * 0.58
    brass = _muted_brass(220.00, 110.00, 0.24) * 0.40
    return _mix((0.0, thud), (0.04, brass))
