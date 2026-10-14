#!/usr/bin/env bash
# Launch the Manus ZMQ server (owns the glove device).
# Builds the C++ binary on first run, then runs it. Any args are passed through.
#
#   ./server.sh
#   ./server.sh --pub-port 5555 --haptic-port 5556 --control-port 5557
#   MANUS_SDK_DIR=/path/to/ManusSDK ./server.sh     # SDK outside manus/ManusSDK
#
# The MANUS SDK is proprietary and not part of this repo -- see README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD:-$HERE/server/build}"
BIN="$BUILD/manus_zmq_server"
SDK="${MANUS_SDK_DIR:-$HERE/ManusSDK}"

if [[ ! -x "$BIN" ]]; then
  if [[ ! -f "$SDK/include/ManusSDK.h" || ! -f "$SDK/lib/libManusSDK_Integrated.so" ]]; then
    echo "[server.sh] MANUS SDK not found in '$SDK' (need include/ManusSDK.h and" >&2
    echo "[server.sh] lib/libManusSDK_Integrated.so). Download the MANUS Core SDK for Linux," >&2
    echo "[server.sh] copy its ManusSDK folder to manus/ManusSDK or set MANUS_SDK_DIR. See manus/README.md." >&2
    exit 1
  fi
  echo "[server.sh] binary missing -- building against $SDK ..."
  cmake -S "$HERE/server" -B "$BUILD" -DMANUS_SDK_DIR="$SDK"
  cmake --build "$BUILD" -j"$(nproc)"
fi

echo "[server.sh] launching: $BIN $*"
exec "$BIN" "$@"
