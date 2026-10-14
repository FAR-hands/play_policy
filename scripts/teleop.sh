#!/usr/bin/env bash
# Live glove teleop: start the Manus glove ZMQ server + the Sharpa-hand Isaac Lab
# playground (isaac/teleop_isaac.py --source live). This is how human play demos are
# collected; see docs/TELEOP.md for the hardware + SDK setup.
#
# Starts the glove server in the background (building it on first run), waits for its
# ZMQ PUB port, then runs the teleop in the foreground. Ctrl-C the teleop and the glove
# server we started is shut down too. A server already publishing on the port is reused
# (and left running on exit).
#
# Run it from the Isaac Lab python env (or point PYTHON at that env's interpreter).
# Takes are written to data/teleop_takes/ ('s' start, 'e' end + save, 'r' reset).
# Any extra args are passed through to teleop_isaac.py, e.g.:
#   scripts/teleop.sh                         # live, right hand, sim-param sliders on
#   scripts/teleop.sh --no-sim-panel          # data collection: physics knobs locked
#   scripts/teleop.sh --out-dir data/my_takes # where takes go
#   MANUS_PORT=6000 scripts/teleop.sh         # glove server + teleop both on :6000
#   PYTHON=/path/to/envs/dsp/bin/python scripts/teleop.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- config (override via env) ------------------------------------------------
MANUS_HOST="${MANUS_HOST:-localhost}"      # where the teleop looks for the glove
MANUS_PORT="${MANUS_PORT:-5555}"           # glove ZMQ PUB port
PYTHON="${PYTHON:-python}"                 # interpreter of the Isaac Lab env
SERVER_SH="$ROOT/manus/server.sh"
TELEOP_PY="$ROOT/isaac/teleop_isaac.py"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "[teleop] python not found: $PYTHON (set PYTHON)" >&2; exit 1; }

# ---- helpers ------------------------------------------------------------------
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }   # TCP connect probe

TTY_STATE="$(stty -g 2>/dev/null || true)"   # save terminal mode; restored on exit
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[teleop] stopping glove server (pid $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # restore the terminal LAST -- Isaac/viser can leave it raw/no-echo on exit
  [[ -n "$TTY_STATE" ]] && stty "$TTY_STATE" 2>/dev/null || true
}
trap cleanup EXIT          # cleanup runs on every exit path ...
trap 'exit 130' INT        # ... and Ctrl-C / SIGTERM funnel through EXIT, so
trap 'exit 143' TERM       #     cleanup (incl. TTY restore) always fires once.

# ---- 1. glove ZMQ server ------------------------------------------------------
if [[ "$MANUS_HOST" != "localhost" && "$MANUS_HOST" != "127.0.0.1" ]]; then
  echo "[teleop] using the remote glove server at $MANUS_HOST:$MANUS_PORT"
elif port_open "$MANUS_PORT"; then
  echo "[teleop] glove server already up on :$MANUS_PORT -- reusing it."
else
  echo "[teleop] starting glove server ($SERVER_SH --pub-port $MANUS_PORT) ..."
  "$SERVER_SH" --pub-port "$MANUS_PORT" </dev/null &   # </dev/null keeps it off the TTY
  SERVER_PID=$!
  echo "[teleop] glove server pid $SERVER_PID; waiting for :$MANUS_PORT (first run builds the binary) ..."
  until port_open "$MANUS_PORT"; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[teleop] glove server exited before binding :$MANUS_PORT" >&2; exit 1; }
    sleep 0.5
  done
  echo "[teleop] glove server is publishing on :$MANUS_PORT."
fi

# ---- 2. Sharpa Rubik's-cube teleop (Isaac Lab) --------------------------------
export OMNI_KIT_ACCEPT_EULA=YES
# Isaac runs headless; the viser web view only binds http://localhost:8080 *after* Kit
# finishes booting (~30-60 s). The output is tee'd to a log so a traceback survives.
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/teleop_$(date +%Y%m%d_%H%M%S).log"
echo "[teleop] output -> $RUN_LOG"
echo "[teleop] 3D view at http://localhost:8080 once Isaac has booted (~30-60 s); --viz-port to change."
# NOT exec'd: keep this shell alive so the cleanup trap restores the terminal and stops
# the glove server when the teleop exits. -u keeps prints live through the tee pipe.
status=0
set +e
"$PYTHON" -u "$TELEOP_PY" --source live --side right \
  --manus-host "$MANUS_HOST" --manus-port "$MANUS_PORT" "$@" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}   # exit of python, not tee
set -e
exit "$status"
