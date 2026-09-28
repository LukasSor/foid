# Jetson Orin Nano Super (8 GB)

Foid is meant to run as two Linux services, not as its own OS. Flash JetPack 6.1+ on the Orin Nano Super, use an NVMe SSD, and keep the desktop environment off if you need RAM.

## Hardware

- Jetson Orin Nano Super 8 GB
- NVMe SSD (models do not belong on SD cards)
- USB conference microphone
- Speaker on 3.5 mm or USB
- Active cooling, power mode MAXN SUPER (~25 W)

CUDA compute capability is **8.7**. If a wheel silently falls back to CPU, that flag is usually wrong.

## Install

```bash
sudo mkdir -p /opt/foid /opt/foid/models /opt/foid/data
sudo chown -R $USER:$USER /opt/foid
cd /opt/foid
# clone or copy this repo here, then:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ./protocol -e ./room-sim -e "./voice[ml,llm]"
```

`llama-cpp-python` on Jetson must be built with CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" pip install llama-cpp-python --no-binary llama-cpp-python
```

`faster-whisper` needs a CTranslate2 build that sees the iGPU. After install:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

That should print `1`.

## Models

```bash
export FOID_MODELS=/opt/foid/models
python scripts/download_models.py
```

Keep only one Whisper profile loaded. Default is German turbo (~2 GB) plus Qwen 1.5B Q4 (~1.2 GB). Do not load the large German fine-tune and a 3B LLM at the same time on 8 GB.

## Audio

Add the service user to `audio` and `video`. Confirm the mic:

```bash
arecord -l
python -c "import sounddevice as sd; print(sd.query_devices())"
```

If USB audio is noisy, raise the wake / silence thresholds in settings rather than retraining a wake word.

## Services

Unit files live in `scripts/`:

```bash
sudo cp scripts/foid-voice.service scripts/foid-sim.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now foid-sim foid-voice
```

- Settings: `http://<jetson>:8080`
- Room view: `http://<jetson>:8090`
- Room TCP: `<jetson>:9500`

On a real room-control box, keep the same JSON-line protocol. The simulator is only the stand-in.

## Memory

Budget that fits Super 8 GB:

| Piece | RAM |
| --- | --- |
| JetPack, no desktop | ~1.5 GB |
| openWakeWord + Silero | ~0.1 GB |
| German turbo Whisper int8 | 1.5–2 GB |
| Qwen2.5-1.5B Q4 | ~1.2 GB |
| headroom | the rest |

If the board starts swapping, pick a smaller multilingual Whisper size under Auto (or the weaker German option if available) and leave the LLM as fallback only. The YAML matcher handles most commands without it.

## Latency tricks

- `beam_size=1` unless dialect recognition is failing
- silence slider around 450 ms; older speakers need more
- Whisper `language=de` locked for German profiles (not Auto)
- models stay loaded; never reload in the listen loop
- NVMe for model files, not a network share
