#!/usr/bin/env bash
# Download + verify + extract the DSP simulation dataset (human teleop + 5 h autonomous play).
#
#   scripts/download_dataset.sh                          # download from $DSP_DATA_URL
#   scripts/download_dataset.sh --from /path/to/dsp_sim_dataset_v1.tar
#   scripts/download_dataset.sh --from /path/to/dir      # dir holding the tar (+ SHA256SUMS),
#                                                        #   or an already-extracted dsp_sim_dataset/
#   scripts/download_dataset.sh --dest /big/disk/data    # extract somewhere else (default: <repo>/data)
#
# Result: <dest>/dsp_sim_dataset/{teleop,play/paper,play/extension,splits,README.md}.
# Every .npz is checked against the SHA256SUMS shipped inside the dataset.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# TODO(authors): point this at the public dataset location before release.
DSP_DATA_URL="${DSP_DATA_URL:-https://huggingface.co/datasets/<ORG>/dsp-sim-dataset/resolve/main}"
TAR_NAME="dsp_sim_dataset_v1.tar"
DS_NAME="dsp_sim_dataset"
DEST="$REPO/data"
FROM=""

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "[download_dataset] unknown argument: $1" >&2; usage 1 ;;
  esac
done

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
if [[ -e "$DEST/$DS_NAME" ]]; then
  echo "[download_dataset] $DEST/$DS_NAME already exists -- remove it first to re-extract." >&2
  exit 1
fi

sha_check() {   # sha_check <SHA256SUMS file> <dir the listed paths are relative to>
  (cd "$2" && sha256sum --quiet -c "$1")
}

fetch() {       # fetch <url> <out>
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -O "$2" "$1"
  else echo "[download_dataset] need curl or wget" >&2; exit 1; fi
}

TAR=""; SUMS=""
if [[ -z "$FROM" ]]; then
  if [[ "$DSP_DATA_URL" == *"<ORG>"* ]]; then
    echo "[download_dataset] DSP_DATA_URL is still the placeholder ($DSP_DATA_URL)." >&2
    echo "[download_dataset] set DSP_DATA_URL=<base url> or use --from <local tar|dir>." >&2
    exit 1
  fi
  echo "[download_dataset] downloading $DSP_DATA_URL/$TAR_NAME -> $DEST"
  fetch "$DSP_DATA_URL/SHA256SUMS" "$DEST/SHA256SUMS"
  fetch "$DSP_DATA_URL/$TAR_NAME" "$DEST/$TAR_NAME"
  TAR="$DEST/$TAR_NAME"; SUMS="$DEST/SHA256SUMS"
elif [[ -f "$FROM" ]]; then
  TAR="$(cd "$(dirname "$FROM")" && pwd)/$(basename "$FROM")"
  [[ -f "$(dirname "$TAR")/SHA256SUMS" ]] && SUMS="$(dirname "$TAR")/SHA256SUMS"
elif [[ -d "$FROM/teleop" && -d "$FROM/play" ]]; then
  echo "[download_dataset] copying extracted dataset $FROM -> $DEST/$DS_NAME"
  cp -r "$FROM" "$DEST/$DS_NAME"
elif [[ -f "$FROM/$TAR_NAME" ]]; then
  TAR="$(cd "$FROM" && pwd)/$TAR_NAME"
  [[ -f "$FROM/SHA256SUMS" ]] && SUMS="$(cd "$FROM" && pwd)/SHA256SUMS"
else
  echo "[download_dataset] --from $FROM: expected $TAR_NAME, a dir containing it, or an extracted $DS_NAME/" >&2
  exit 1
fi

if [[ -n "$TAR" ]]; then
  if [[ -n "$SUMS" ]]; then
    echo "[download_dataset] verifying $(basename "$TAR") ..."
    (cd "$(dirname "$TAR")" && grep " $TAR_NAME\$" "$SUMS" | sha256sum --quiet -c -) \
      || { echo "[download_dataset] checksum MISMATCH for $TAR" >&2; exit 1; }
  else
    echo "[download_dataset] no SHA256SUMS next to the tar -- skipping the archive checksum" \
         "(per-file checksums are still verified after extraction)"
  fi
  echo "[download_dataset] extracting -> $DEST/$DS_NAME"
  tar -xf "$TAR" -C "$DEST"
fi

echo "[download_dataset] verifying $(grep -c . "$DEST/$DS_NAME/SHA256SUMS") files ..."
sha_check "$DEST/$DS_NAME/SHA256SUMS" "$DEST/$DS_NAME" \
  || { echo "[download_dataset] per-file checksum MISMATCH under $DEST/$DS_NAME" >&2; exit 1; }
if [[ -z "$FROM" ]]; then rm -f "$DEST/$TAR_NAME" "$DEST/SHA256SUMS"; fi   # downloaded copy no longer needed
n_tel=$(find "$DEST/$DS_NAME/teleop" -name '*.npz' | wc -l)
n_ply=$(find "$DEST/$DS_NAME/play" -name '*.npz' | wc -l)
echo "[download_dataset] OK: $n_tel teleop takes, $n_ply play rollouts in $DEST/$DS_NAME"
echo "[download_dataset] policy:      python -m play_policy.train --data_dir $DEST/$DS_NAME/teleop ..."
echo "[download_dataset] world model: python -m play_policy.train_worldmodel --data_dir $DEST/$DS_NAME/play ..."
