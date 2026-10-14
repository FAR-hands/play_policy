#!/usr/bin/env bash
# Fetch the released simulation checkpoints into checkpoints/ and verify their sha256.
#
#   quickstart/download_checkpoints.sh                    # download from $DSP_CKPT_URL
#   quickstart/download_checkpoints.sh --from <dir>       # copy from a local directory
#   quickstart/download_checkpoints.sh --from <dir> --link   # symlink instead of copying
#
#   dsp_sim_play_policy.pt  (30 MB)   DiT diffusion play policy, 7.5M params, 200-step chunks,
#                                     behaviour-cloned on the human teleop takes
#   dsp_sim_world_model.pt  (157 MB)  DiT diffusion world model, 39M params, 50-step horizon,
#                                     3-frame history; trained on 672k self-play frames
#
# Inference-only: optimizer state is stripped; the weights are the EMA weights used in the paper.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$HERE/../checkpoints}"
# Files are assets of the v1.0 GitHub release of FAR-hands/play_policy. While that repo is
# private, plain downloads fail: run `gh release download v1.0 -R FAR-hands/play_policy -D <dir>`
# and pass --from <dir>.
URL="${DSP_CKPT_URL:-https://github.com/FAR-hands/play_policy/releases/download/v1.0}"
FILES=(dsp_sim_play_policy.pt dsp_sim_world_model.pt)

FROM=""; LINK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift ;;
    --link) LINK=1 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  if [ -n "$FROM" ]; then
    src="$(cd "$FROM" && pwd)/$f"
    [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
    if [ $LINK = 1 ]; then ln -sf "$src" "$DEST/$f"; else cp "$src" "$DEST/$f"; fi
  elif [ ! -f "$DEST/$f" ]; then
    echo "[download] $URL/$f"
    if command -v wget >/dev/null; then wget -q --show-progress -O "$DEST/$f.part" "$URL/$f"
    else curl -fL --progress-bar -o "$DEST/$f.part" "$URL/$f"; fi
    mv "$DEST/$f.part" "$DEST/$f"
  fi
done

cd "$DEST"
sha256sum -c "$HERE/SHA256SUMS" && echo "[download] checkpoints ready in $(pwd)"
