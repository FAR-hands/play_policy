#!/usr/bin/env bash
# Browser replay of recorded takes / rollouts (isaac/viser_replay_server.py): a viser 3D
# view of the Sharpa hand + Rubik cube with play/pause, scrub, speed and, for MPC
# rollouts, a goal cube. Point it at one .npz or a directory (searched recursively; a
# dropdown switches between every recording found). No Isaac Sim / GPU needed.
#
#   scripts/replay.sh                                     # everything under data/
#   scripts/replay.sh --rollout data/teleop_takes
#   scripts/replay.sh --rollout <take>.npz --port 8090 --speed 2
#
# --sim switches to isaac/viser_sim_replay.py: the same scene, but the recording's joint
# commands are RE-SIMULATED in Isaac Lab, with the simulator's parameters (physics rate,
# substeps, hand gains, cube contact stiffness/damping, cube damping) on live sliders and
# a ghost cube marking the recorded pose. Boots Isaac Sim (~30-60 s), needs the GPU:
#
#   scripts/replay.sh --sim --rollout <take>.npz
#   scripts/replay.sh --sim --selftest --rollout <take>.npz     # knob check, no browser
#
# Run it from the Isaac Lab python env (viser, yourdfpy, trimesh live there) or set PYTHON.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python}"
PORT="${PORT:-8080}"                       # default port (an explicit --port wins)
SERVER_PY="$ROOT/isaac/viser_replay_server.py"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[replay] python not found: $PYTHON (set PYTHON)" >&2; exit 1; }

SIM=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --sim|--live) SIM=1 ;;
    *)            ARGS+=("$a") ;;
  esac
done

PORT_ARG=(--port "$PORT")
prev=""
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  [[ "$prev" == "--port" ]] && PORT="$a" && PORT_ARG=()      # an explicit --port wins
  prev="$a"
done

# (a relative --rollout resolves against the CWD, then the repo root)
if (( SIM )); then
  export OMNI_KIT_ACCEPT_EULA=YES
  LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
  RUN_LOG="$LOG_DIR/simreplay_$(date +%Y%m%d_%H%M%S).log"
  echo "[replay] LIVE PHYSICS: re-simulating the recording in Isaac Lab; sliders at http://localhost:${PORT} after boot (~30-60 s)"
  echo "[replay] output -> $RUN_LOG"
  status=0
  set +e
  "$PYTHON" -u "$ROOT/isaac/viser_sim_replay.py" "${PORT_ARG[@]}" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$RUN_LOG"
  status=${PIPESTATUS[0]}
  set -e
  exit "$status"
fi

echo "[replay] open http://localhost:${PORT} once it prints 'viser 3D view' (a few seconds)"
exec "$PYTHON" -u "$SERVER_PY" "${PORT_ARG[@]}" "${ARGS[@]+"${ARGS[@]}"}"
