# Foid

Offline voice room-control for older and disabled people (Tirol). A Jetson-class box listens for a wake word, understands spoken commands, and drives room devices over newline-delimited JSON on TCP. Feedback is short earcons — not full TTS.

German-first in use; English in source; UI strings live in i18n (EN/DE). Target hardware later: **Jetson Orin Nano Super**.

## Processes

| Process | Role | Ports |
| --- | --- | --- |
| `foid-voice` | Wake → STT → hybrid NLU → room commands; settings UI | HTTP `:8080` |
| `foid-sim` | Browser room view + stand-in room box | HTTP `:8090`, TCP `:9500` |

The voice box does not care whether the peer is the simulator or a real control box — both speak the same protocol ([docs/protocol.md](docs/protocol.md)).

```
mic → openWakeWord → earcon → Silero VAD → faster-whisper
                                              ↓
                                    YAML aliases + fuzzy NLU
                                    (optional local LLM fallback)
                                              ↓
                              JSON/TCP → room box / foid-sim
```

Add a device by dropping YAML under [`protocol/foid_protocol/devices/`](protocol/foid_protocol/devices/). No code change.

## Packages

| Path | Package | Purpose |
| --- | --- | --- |
| `protocol/` | `foid-protocol` | Device catalog, i18n, JSON-line codec, TCP helpers |
| `voice/` | `foid-voice` | Wake, STT, NLU, earcons, settings server |
| `room-sim/` | `foid-sim` | Live room page + TCP room simulator |

## Quick start (desktop)

Python 3.11+. Desktop install runs without GPU models; type commands on the settings page to exercise the room.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./protocol -e ./room-sim -e ./voice
scripts/run_servers.sh
```

- Room view: http://127.0.0.1:8090  
- Settings: http://127.0.0.1:8080  

Type e.g. `Fenster auf` on the settings page to move the window without a microphone. Stop with `scripts/stop_servers.sh`.

Short-lived shells (CI, some IDE agent sessions) kill foreground children when the task ends — prefer the run/stop scripts so the processes detach cleanly.

### Full ear (wake + Whisper + optional LLM)

```bash
pip install -e "./voice[ml,llm]"
python scripts/download_models.py
```

Models land under `~/.local/share/foid/models` by default (`FOID_MODELS` to override). They are **not** in this repo.

Jetson install, CUDA, memory budget, systemd: [docs/jetson.md](docs/jetson.md).

## Architecture

1. **Wake** — [openWakeWord](https://github.com/dscripka/openWakeWord). Default id is `computer` (pretrained community model). Official options include Hey Jarvis, Alexa, Hey Mycroft; more appear in settings (including Home Assistant catalog entries, downloaded on save).
2. **STT** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Settings language dropdown: **Auto**, **German** (special fine-tunes), **English** (`.en` packs). Default: German turbo fine-tune.
3. **NLU** — hybrid: catalog phrases from device YAML aliases + RapidFuzz; optional local GGUF LLM (`llama-cpp-python`) when the fuzzy score is mid-range.
4. **Feedback** — packaged WAV earcons (wake / listen / ok / fail / …), not spoken replies.
5. **Room link** — hello + command / ack / state / snapshot over TCP (see protocol doc).

## Protocol (overview)

Newline JSON on TCP port **9500**:

```json
{"type":"hello","role":"voice"}
{"type":"command","id":"…","device":"window","action":"open"}
{"type":"ack","id":"…","ok":true,"state":{"open":true}}
{"type":"state","device":"window","state":{"open":true}}
```

Full message set, catalog rules, and device YAML shape: [docs/protocol.md](docs/protocol.md).

Example phrases after the wake cue: `Fenster auf`, `Fernseher an`, `Lehne nach oben`, `Licht aus`.

## Configuration

| Item | Default | Notes |
| --- | --- | --- |
| Settings file | `~/.config/foid/settings.json` | Or `FOID_CONFIG` / `scripts/run_servers.sh` → `data/settings.json` |
| Models dir | `~/.local/share/foid/models` | `FOID_MODELS` |
| Devices dir | packaged YAML | Override with `FOID_DEVICES_DIR` |
| UI locale | `en` | Switch to `de` in settings |
| Room peer | `127.0.0.1:9500` | Host/port in settings |

Useful settings fields: wake word id, STT language/model, silence ms, beam size, earcon volume, match/LLM thresholds, LLM GGUF filename.

## License

MIT — see [LICENSE](LICENSE).

Model weights you download separately keep their own upstream licenses (Whisper forks, openWakeWord, Qwen GGUF, etc.).
