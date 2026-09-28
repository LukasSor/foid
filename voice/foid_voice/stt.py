from __future__ import annotations

import ctypes
import glob
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from foid_voice.stt_profiles import get_model, resolve_stt, whisper_language

log = logging.getLogger("foid.stt")


class SpeechToTextBusy(RuntimeError):
    """Raised when a second transcribe is refused because CUDA/CT2 is already running."""

_CUBLAS12 = "libcublas.so.12"
_CUDA12_PRELOAD = (
    "libnvJitLink.so.12",
    "libcudart.so.12",
    "libnvrtc.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn.so.9",
)
_CUDA_ERROR_MARKERS = (
    "libcublas",
    "libcudnn",
    "libcudart",
    "cublas",
    "cudnn",
    "cuda",
    "cannot be loaded",
    "not found",
    "no kernel image",
    "not supported",
)
_CUDA12_READY = False


def _nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import nvidia

        for entry in getattr(nvidia, "__path__", []):
            root = Path(entry)
            dirs.extend(sorted(root.glob("*/lib")))
    except Exception:
        pass
    seen: list[Path] = []
    for path in dirs:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.append(resolved)
    # Prefer CUDA 12 runtime/cublas/cudnn over leftover cu13 stubs.
    preferred = ("cublas", "cuda_runtime", "cudnn", "nvjitlink", "cuda_nvrtc")
    dirs_pref = [p for p in seen if p.parent.name in preferred]
    dirs_rest = [p for p in seen if p.parent.name not in preferred]
    return dirs_pref + dirs_rest


def _library_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if part:
            dirs.append(Path(part))
    dirs.extend(_nvidia_lib_dirs())
    dirs.extend(
        Path(p)
        for p in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/opt/cuda/lib64",
            "/opt/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        )
    )
    dirs.extend(Path(p) for p in glob.glob("/usr/local/cuda-12*/lib64"))
    dirs.extend(Path(p) for p in glob.glob("/usr/local/cuda-12*/lib"))
    dirs.extend(Path(p) for p in glob.glob("/usr/local/cuda-11*/lib64"))
    seen: list[Path] = []
    for path in dirs:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.append(resolved)
    return seen


def _find_soname(soname: str) -> Path | None:
    for directory in _library_search_dirs():
        candidate = directory / soname
        if candidate.is_file():
            return candidate
    try:
        handle = ctypes.CDLL(soname)
        if handle._name:
            return Path(soname)
    except OSError:
        pass
    return None


def _prepend_ld_library_path(directories: list[Path]) -> bool:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    added = False
    for directory in reversed(directories):
        prefix = str(directory)
        if prefix not in parts:
            parts.insert(0, prefix)
            added = True
    if added:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    return added


def _preload_cuda12_libs() -> Path | None:
    cublas12 = _find_soname(_CUBLAS12)
    if cublas12 is None:
        return None
    directories = []
    if cublas12.is_absolute():
        directories.append(cublas12.parent)
    directories.extend(_nvidia_lib_dirs())
    _prepend_ld_library_path(directories)
    for soname in _CUDA12_PRELOAD:
        path = _find_soname(soname)
        if path is None:
            continue
        target = str(path) if path.is_absolute() else soname
        try:
            ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            if soname == _CUBLAS12:
                log.warning("libcublas.so.12 exists but cannot be loaded (%s).", exc)
                return None
            log.debug("optional CUDA 12 lib %s skipped (%s)", soname, exc)
    return cublas12


def prepare_cuda12_runtime(*, reexec: bool = False) -> None:
    """Make CT2's CUDA 12 wheels find cuBLAS on a CUDA 13 driver.

    glibc snapshots LD_LIBRARY_PATH at process start, so the voice entrypoint
    may re-exec once after pointing it at the nvidia-* pip lib dirs.
    """
    global _CUDA12_READY
    nvidia_dirs = _nvidia_lib_dirs()
    if nvidia_dirs:
        changed = _prepend_ld_library_path(nvidia_dirs)
        if reexec and changed and os.environ.get("FOID_CUDA12_LIBS") != "1":
            os.environ["FOID_CUDA12_LIBS"] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])
    _preload_cuda12_libs()
    _CUDA12_READY = True


def _cublas_major_versions() -> list[str]:
    found: list[str] = []
    for major in ("13", "12", "11"):
        if _find_soname(f"libcublas.so.{major}") is not None:
            found.append(major)
    return found


def _cuda_runtime_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_ERROR_MARKERS)


def _cuda_usable() -> bool:
    prepare_cuda12_runtime()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    cublas12 = _find_soname(_CUBLAS12)
    if cublas12 is None:
        versions = _cublas_major_versions()
        if versions:
            log.warning(
                "CUDA GPU found but libcublas.so.12 is not loadable (found cuBLAS %s). Falling back to CPU int8.",
                "/".join(versions),
            )
        else:
            log.warning("CUDA GPU found but libcublas.so.12 is not loadable. Falling back to CPU int8.")
        return False
    return True


def pick_device(requested: str = "auto") -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "int8"
    if requested in {"cuda", "auto"} and _cuda_usable():
        # RTX 50 / sm_120 cannot run INT8 GEMMs; float16 is the working GPU path.
        return "cuda", "float16"
    if requested == "cuda":
        log.warning("Requested CUDA STT is unavailable. Falling back to CPU int8.")
    return "cpu", "int8"


class SpeechToText:
    def __init__(
        self,
        model_id: str,
        language: str = "de",
        device: str = "auto",
        beam_size: int = 1,
        prompt: str = "",
    ):
        language, model_id = resolve_stt(language, model_id)
        self.model_id = model_id
        self.language = language
        self.requested_device = device
        self.beam_size = beam_size
        self.prompt = prompt
        self.model = None
        self.device = "cpu"
        self.compute = "int8"
        self.available = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._load()

    def spec(self):
        return get_model(self.model_id)

    def _load(self, force_cpu: bool = False) -> None:
        try:
            from faster_whisper import WhisperModel

            spec = self.spec()
            if force_cpu:
                self.device, self.compute = "cpu", "int8"
            else:
                self.device, self.compute = pick_device(self.requested_device)
            compute = self.compute
            if self.device == "cpu":
                compute = spec.compute
                self.compute = compute
            self.model = self._open_model(spec.id, self.device, compute)
            self.available = True
            self.error = None
            log.info("loaded STT %s lang=%s on %s/%s", spec.id, self.language, self.device, self.compute)
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            log.warning("STT unavailable: %s", exc)

    def _open_model(self, model_name: str, device: str, compute: str):
        from faster_whisper import WhisperModel

        try:
            return WhisperModel(model_name, device=device, compute_type=compute, num_workers=1)
        except Exception as exc:
            if device != "cuda" or not _cuda_runtime_error(exc):
                raise
            if compute != "float16":
                log.warning("CUDA STT failed with %s (%s). Retrying float16.", compute, exc)
                try:
                    self.compute = "float16"
                    return WhisperModel(model_name, device="cuda", compute_type="float16", num_workers=1)
                except Exception as retry_exc:
                    if not _cuda_runtime_error(retry_exc):
                        raise
                    exc = retry_exc
            log.warning("CUDA STT failed to load (%s). Falling back to CPU int8.", exc)
            self.device, self.compute = "cpu", "int8"
            return WhisperModel(model_name, device="cpu", compute_type="int8", num_workers=1)

    def configure(
        self,
        model_id: str,
        language: str,
        device: str,
        beam_size: int,
        prompt: str,
    ) -> None:
        with self._lock:
            language, model_id = resolve_stt(language, model_id)
            self.language = language
            self.beam_size = beam_size
            self.prompt = prompt
            need_reload = model_id != self.model_id or device != self.requested_device or not self.available
            self.model_id = model_id
            self.requested_device = device
            if need_reload:
                self._load()

    def busy(self) -> bool:
        return self._lock.locked()

    def transcribe(self, audio: np.ndarray, *, wait: bool = True) -> str:
        if not self.available or self.model is None:
            raise RuntimeError(self.error or "speech-to-text is not loaded")
        acquired = self._lock.acquire(blocking=wait)
        if not acquired:
            raise SpeechToTextBusy("speech-to-text is busy")
        try:
            return self._transcribe_locked(audio)
        finally:
            self._lock.release()

    def _transcribe_locked(self, audio: np.ndarray) -> str:
        try:
            return self._decode(audio)
        except SpeechToTextBusy:
            raise
        except Exception as exc:
            if self.device != "cuda" or not _cuda_runtime_error(exc):
                raise
            log.warning("CUDA STT failed during transcription (%s). Falling back to CPU int8.", exc)
            self._load(force_cpu=True)
            if not self.available or self.model is None:
                raise RuntimeError(self.error or "speech-to-text is not loaded") from exc
            return self._decode(audio)

    def _decode(self, audio: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError(self.error or "speech-to-text is not loaded")
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language=whisper_language(self.language),
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.prompt or None,
            without_timestamps=True,
        )
        parts: list[str] = []
        try:
            for segment in segments:
                parts.append(segment.text.strip())
        finally:
            closer = getattr(segments, "close", None)
            if callable(closer):
                closer()
        return " ".join(part for part in parts if part).strip()
