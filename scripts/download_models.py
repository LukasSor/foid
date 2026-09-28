#!/usr/bin/env python3
"""Prefetch wake-word, Whisper, and LLM files for offline use."""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

from foid_voice.wake_models import PREFETCH_COMMUNITY_IDS, community_model_meta, wake_dir

# Drop extra custom classifiers into models/wake/*.onnx — they appear in settings
# automatically. Training a new wake phrase is out of scope here; use the
# openWakeWord Colab if you need a name that is not pretrained.


def models_dir(explicit: str | None) -> Path:
    path = Path(explicit or os.environ.get("FOID_MODELS") or Path.home() / ".local/share/foid/models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Foid-model-prefetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def download_community_wake(target: Path) -> None:
    folder = wake_dir(target)
    folder.mkdir(parents=True, exist_ok=True)
    meta_by_id = community_model_meta()
    for model_id in PREFETCH_COMMUNITY_IDS:
        meta = meta_by_id.get(model_id)
        if meta is None:
            print(f"wake community {model_id} skipped: not in catalog")
            continue
        dest = folder / meta["filename"]
        if dest.is_file() and dest.stat().st_size > 10_000:
            print(f"wake community: {model_id} already present")
            continue
        print(f"wake community: {model_id} ({meta['license']})")
        try:
            _fetch(meta["url"], dest)
            print(f"saved {dest}")
        except (urllib.error.URLError, OSError) as exc:
            print(f"wake community {model_id} skipped: {exc}")


def download_wake(target: Path) -> None:
    download_community_wake(target)
    try:
        import openwakeword

        openwakeword.utils.download_models(model_names=["hey_jarvis", "alexa", "hey_mycroft"])
        print("wake official: hey_jarvis, alexa, hey_mycroft")
    except Exception as exc:
        print(f"wake official download skipped: {exc}")


def download_whisper() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("install huggingface_hub to prefetch Whisper models")
        return
    repos = [
        "cstr/whisper-large-v3-turbo-german-int8_float32",
        "Systran/faster-whisper-small",
        "Systran/faster-whisper-medium",
    ]
    for repo in repos:
        print(f"whisper: {repo}")
        snapshot_download(repo_id=repo)


def download_llm(target: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("install huggingface_hub to prefetch the LLM")
        return
    filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    print(f"llm: {filename}")
    path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename=filename,
        local_dir=str(target),
    )
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    target = models_dir(args.models_dir)
    download_wake(target)
    if not args.skip_whisper:
        download_whisper()
    if not args.skip_llm:
        download_llm(target)
    print(f"models dir: {target}")


if __name__ == "__main__":
    main()
