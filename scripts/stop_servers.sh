#!/usr/bin/env bash
# Stop detached foid-sim / foid-voice started by scripts/run_servers.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FOID_VENV:-$ROOT/.venv}"
LOG_DIR="${FOID_LOG_DIR:-$ROOT/data/logs}"
TARGET="${1:-all}"

SIM_PIDFILE="$LOG_DIR/foid-sim.pid"
VOICE_PIDFILE="$LOG_DIR/foid-voice.pid"
SIM_BIN="$VENV/bin/foid-sim"
VOICE_BIN="$VENV/bin/foid-voice"
SIM_PORT="${FOID_SIM_PORT:-8090}"
VOICE_PORT="${FOID_VOICE_PORT:-8080}"
TCP_PORT="${FOID_TCP_PORT:-9500}"

kill_pid() {
  local pid="$1"
  [[ -n "$pid" && "$pid" != "0" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 0.2
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidfile="$1" name="$2"
  if [[ ! -f "$pidfile" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  echo "stopping $name pid=${pid:-none}"
  kill_pid "${pid:-}"
  rm -f "$pidfile"
}

stop_matching() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  local pid
  for pid in $pids; do
    # Do not kill this stop script.
    if [[ "$pid" -eq "$$" || "$pid" -eq "$PPID" ]]; then
      continue
    fi
    echo "stopping leftover $pattern pid=$pid"
    kill_pid "$pid"
  done
}

listen_pid() {
  local port="$1"
  ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

stop_port() {
  local port="$1"
  local pid
  pid="$(listen_pid "$port")"
  if [[ -n "$pid" ]]; then
    echo "stopping listener on :$port pid=$pid"
    kill_pid "$pid"
  fi
}

stop_unit() {
  local unit="$1"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
  fi
}

stop_sim() {
  stop_unit foid-sim
  stop_pidfile "$SIM_PIDFILE" "foid-sim"
  stop_matching "$SIM_BIN"
  stop_port "$SIM_PORT"
  stop_port "$TCP_PORT"
}

stop_voice() {
  stop_unit foid-voice
  stop_pidfile "$VOICE_PIDFILE" "foid-voice"
  stop_matching "$VOICE_BIN"
  stop_port "$VOICE_PORT"
}

case "$TARGET" in
  all | "")
    stop_sim
    stop_voice
    ;;
  foid-sim | sim)
    stop_sim
    ;;
  foid-voice | voice)
    stop_voice
    ;;
  *)
    echo "usage: $0 [all|foid-sim|foid-voice]" >&2
    exit 2
    ;;
esac
