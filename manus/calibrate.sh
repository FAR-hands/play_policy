#!/usr/bin/env bash
# Launch the Manus skeleton calibration UI. The server (./server.sh) must already
# be running -- calibration SDK calls execute inside it; this drives + visualizes.
# Open the printed http://localhost:8080 after it starts. Args are passed through.
#
#   ./calibrate.sh                # right hand, localhost
#   ./calibrate.sh --side left
#   PYTHON=/path/to/python ./calibrate.sh   # override the interpreter (needs viser + pyzmq)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [[ -z "$PYTHON" ]]; then
  echo "[calibrate.sh] no python found; set PYTHON=/path/to/python" >&2
  exit 1
fi

# Make `import manus_glove` resolve to this local copy (no pip install needed).
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

echo "[calibrate.sh] python: $PYTHON"
exec "$PYTHON" "$HERE/examples/calibrate_hand.py" "$@"
