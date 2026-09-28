#!/usr/bin/env bash
# Start foid-sim and foid-voice so they survive the calling shell.
# Cursor agent child processes are killed when the agent task ends — do not run
# foid-voice / foid-sim as foreground children of an agent shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${FOID_VENV:-$ROOT/.venv}"
LOG_DIR="${FOID_LOG_DIR:-$ROOT/data/logs}"
export FOID_CONFIG="${FOID_CONFIG:-$ROOT/data/settings.json}"

SIM_HOST="${FOID_SIM_HOST:-0.0.0.0}"
SIM_PORT="${FOID_SIM_PORT:-8090}"
TCP_HOST="${FOID_TCP_HOST:-0.0.0.0}"
TCP_PORT="${FOID_TCP_PORT:-9500}"
VOICE_HOST="${FOID_VOICE_HOST:-0.0.0.0}"
VOICE_PORT="${FOID_VOICE_PORT:-8080}"

SIM_BIN="$VENV/bin/foid-sim"
VOICE_BIN="$VENV/bin/foid-voice"
SIM_PIDFILE="$LOG_DIR/foid-sim.pid"
VOICE_PIDFILE="$LOG_DIR/foid-voice.pid"
SIM_LOG="$LOG_DIR/foid-sim.log"
VOICE_LOG="$LOG_DIR/foid-voice.log"
STOP_SCRIPT="$ROOT/scripts/stop_servers.sh"
USE_SYSTEMD=0

mkdir -p "$LOG_DIR"

if [[ ! -x "$SIM_BIN" || ! -x "$VOICE_BIN" ]]; then
  echo "missing venv binaries in $VENV (expected foid-sim and foid-voice)" >&2
  exit 1
fi

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  USE_SYSTEMD=1
fi

http_ok() {
  curl -sf -o /dev/null --connect-timeout 1 --max-time 2 "$1"
}

ppid_of() {
  ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '
}

listen_pid() {
  local port="$1"
  ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

unit_pid() {
  systemctl --user show -p MainPID --value "$1" 2>/dev/null || true
}

healthy_systemd() {
  local unit="$1" url="$2"
  systemctl --user is-active --quiet "$unit" || return 1
  http_ok "$url"
}

wait_http() {
  local url="$1" name="$2" timeout="${3:-90}"
  local i
  for i in $(seq 1 "$timeout"); do
    if http_ok "$url"; then
      echo "$name healthy at $url"
      return 0
    fi
    sleep 1
  done
  echo "$name did not become healthy at $url" >&2
  return 1
}

kill_pid() {
  local pid="$1"
  [[ -n "$pid" && "$pid" != "0" ]] || return 0
  kill "$pid" 2>/dev/null || true
  local i
  for i in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -9 "$pid" 2>/dev/null || true
}

claim_port() {
  local port="$1" expected="$2" name="$3"
  local listen
  [[ -n "$expected" && "$expected" != "0" ]] || return 0
  listen="$(listen_pid "$port")"
  if [[ "$listen" == "$expected" ]]; then
    return 0
  fi
  if [[ -n "$listen" ]]; then
    echo "replacing interloper pid=$listen on :$port with $name pid=$expected"
    kill_pid "$listen"
  fi
}

write_pid() {
  local pidfile="$1" pid="$2"
  if [[ -n "$pid" && "$pid" != "0" ]]; then
    echo "$pid" >"$pidfile"
  fi
}

start_systemd_unit() {
  local unit="$1" logfile="$2" bin="$3"
  shift 3
  systemctl --user stop "$unit" 2>/dev/null || true
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user \
    --unit="$unit" \
    --collect \
    --working-directory="$ROOT" \
    --setenv="FOID_CONFIG=$FOID_CONFIG" \
    --setenv="PATH=$VENV/bin:${PATH:-/usr/bin}" \
    --property=Restart=on-failure \
    --property=RestartSec=3 \
    --property="StandardOutput=append:${logfile}" \
    --property="StandardError=append:${logfile}" \
    -- "$bin" "$@"
}

start_doublefork() {
  local pidfile="$1" logfile="$2"
  shift 2
  "$VENV/bin/python" - "$pidfile" "$logfile" "$ROOT" "$@" <<'PY'
import os, sys, fcntl

pidfile, logfile, root, *cmd = sys.argv[1:]
os.chdir(root)
os.environ.setdefault("PWD", root)

r, w = os.pipe()
pid = os.fork()
if pid > 0:
    os.close(w)
    status = os.read(r, 16)
    os.close(r)
    os.waitpid(pid, 0)
    sys.exit(0 if status == b"ok" else 1)

os.close(r)
os.setsid()
pid2 = os.fork()
if pid2 > 0:
    os._exit(0)

fcntl.fcntl(w, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
os.umask(0o22)
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)
os.close(devnull)
logfd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(logfd, 1)
os.dup2(logfd, 2)
if logfd > 2:
    os.close(logfd)

with open(pidfile, "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))
os.write(w, b"ok")
os.close(w)
os.execvpe(cmd[0], cmd, os.environ)
PY
}

ensure_service() {
  local name="$1" url="$2" pidfile="$3" logfile="$4" unit="$5" bin="$6"
  shift 6
  if [[ "$USE_SYSTEMD" -eq 1 ]] && healthy_systemd "$unit" "$url"; then
    local pid
    pid="$(unit_pid "$unit")"
    write_pid "$pidfile" "$pid"
    echo "$name already systemd-user and healthy (pid ${pid:-unknown})"
    return 0
  fi
  if [[ -x "$STOP_SCRIPT" ]]; then
    "$STOP_SCRIPT" "$name" || true
  fi
  sleep 0.4
  echo "starting $name -> $logfile"
  if [[ "$USE_SYSTEMD" -eq 1 ]]; then
    start_systemd_unit "$unit" "$logfile" "$bin" "$@"
    write_pid "$pidfile" "$(unit_pid "$unit")"
  else
    start_doublefork "$pidfile" "$logfile" "$bin" "$@"
  fi
}

ensure_service foid-sim "http://127.0.0.1:${SIM_PORT}/" "$SIM_PIDFILE" "$SIM_LOG" foid-sim "$SIM_BIN" \
  --host "$SIM_HOST" --port "$SIM_PORT" --tcp-host "$TCP_HOST" --tcp-port "$TCP_PORT"

ensure_service foid-voice "http://127.0.0.1:${VOICE_PORT}/api/status" "$VOICE_PIDFILE" "$VOICE_LOG" foid-voice "$VOICE_BIN" \
  --host "$VOICE_HOST" --port "$VOICE_PORT" --config "$FOID_CONFIG"

wait_http "http://127.0.0.1:${SIM_PORT}/" "foid-sim" 20
if [[ "$USE_SYSTEMD" -eq 1 ]]; then
  claim_port "$SIM_PORT" "$(unit_pid foid-sim)" foid-sim
fi
wait_http "http://127.0.0.1:${SIM_PORT}/" "foid-sim" 20

wait_http "http://127.0.0.1:${VOICE_PORT}/api/status" "foid-voice" 90
if [[ "$USE_SYSTEMD" -eq 1 ]]; then
  claim_port "$VOICE_PORT" "$(unit_pid foid-voice)" foid-voice
fi
wait_http "http://127.0.0.1:${VOICE_PORT}/api/status" "foid-voice" 90

if [[ "$USE_SYSTEMD" -eq 1 ]]; then
  write_pid "$SIM_PIDFILE" "$(unit_pid foid-sim)"
  write_pid "$VOICE_PIDFILE" "$(unit_pid foid-voice)"
fi
if [[ ! -s "$VOICE_PIDFILE" ]] || ! kill -0 "$(cat "$VOICE_PIDFILE")" 2>/dev/null; then
  write_pid "$VOICE_PIDFILE" "$(listen_pid "$VOICE_PORT")"
fi
if [[ ! -s "$SIM_PIDFILE" ]] || ! kill -0 "$(cat "$SIM_PIDFILE")" 2>/dev/null; then
  write_pid "$SIM_PIDFILE" "$(listen_pid "$SIM_PORT")"
fi

SIM_PID="$(cat "$SIM_PIDFILE" 2>/dev/null || true)"
VOICE_PID="$(cat "$VOICE_PIDFILE" 2>/dev/null || true)"

echo "FOID_CONFIG=$FOID_CONFIG"
echo "backend=$([[ "$USE_SYSTEMD" -eq 1 ]] && echo systemd-user || echo double-fork)"
echo "logs: $SIM_LOG  $VOICE_LOG"
echo "pids: sim=${SIM_PID:-none} voice=${VOICE_PID:-none}"
echo "ports: ${VOICE_HOST}:${VOICE_PORT}  ${SIM_HOST}:${SIM_PORT}  tcp ${TCP_HOST}:${TCP_PORT}"
echo "listen pid 8080=$(listen_pid "$VOICE_PORT") 8090=$(listen_pid "$SIM_PORT") 9500=$(listen_pid "$TCP_PORT")"
echo "ppid sim=$(ppid_of "${SIM_PID:-0}") voice=$(ppid_of "${VOICE_PID:-0}")"
