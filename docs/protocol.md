# Room protocol

Foid talks to a room-control box with **newline-delimited JSON over TCP** (default port **9500**). One JSON object per line, UTF-8, no framing beyond `\n`.

The voice box (`foid-voice`) is the client. `foid-sim` (or a real room box) is the server. Either peer can be swapped without changing the other.

## Message types

| `type` | Direction | Purpose |
| --- | --- | --- |
| `hello` | client → server | Announce role (default `"voice"`). Server may reply with `snapshot`. |
| `command` | client → server | Ask to run one device action. |
| `ack` | server → client | Result for a `command` (`ok`, optional `error`, optional `state`). |
| `state` | server → clients | Device changed; optional `all` full snapshot. |
| `snapshot` | server → client | Full device map (e.g. after `hello`). |

### Examples

```json
{"type":"hello","role":"voice"}
{"type":"snapshot","devices":{"window":{"open":false},"lamp":{"on":false}}}
{"type":"command","id":"a1b2","device":"window","action":"open","extra":{}}
{"type":"ack","id":"a1b2","ok":true,"state":{"open":true}}
{"type":"state","device":"window","state":{"open":true},"all":{"window":{"open":true}}}
```

- `command.id` is a client-chosen string (UUID in the reference client). `ack.id` must match.
- `extra` is reserved for future parameters; empty object is fine.
- Unknown `type` values should be ignored or rejected by the server; the reference codec raises on parse.

Pydantic models and helpers live in `protocol/foid_protocol/` (`messages.py`, `codec.py`, `tcp.py`).

## Device catalog (YAML)

Devices are declarative files in `protocol/foid_protocol/devices/*.yaml`. The catalog drives:

- allowed `device` / `action` ids on the wire
- NLU phrase generation (aliases × light German prefixes/suffixes)
- simulator UI widgets
- Whisper initial prompt and optional LLM tool schema

Minimal shape:

```yaml
id: window
name: window
ui: window          # widget kind in the simulator
aliases:
  - fenster
  - window
defaults:
  open: false
actions:
  - id: open
    aliases: [auf, öffnen, open]
    set:
      open: true
  - id: close
    aliases: [zu, schließen, close]
    set:
      open: false
```

Action effects:

- `set` — assign keys
- `delta` — add to numeric keys
- `clamp` — `[low, high]` after delta/set

Override the directory with `FOID_DEVICES_DIR`. Ship only what the room can actually control.

## Implementing a real room box

1. Listen on TCP; read lines; parse JSON objects.
2. On `hello`, optionally send `snapshot`.
3. On `command`, apply the action if known; reply with matching `ack`.
4. Broadcast `state` (and keep TCP clients in sync) when something changes.

You do not need Python — any language that can do TCP + JSON lines works. Keep latency low; the voice loop waits on `ack` with a short timeout.
