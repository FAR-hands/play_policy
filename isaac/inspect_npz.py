#!/usr/bin/env python3
"""Inspect the contents of .npz trajectory/demo files.

For each file it prints every stored array: name, dtype, shape, and a short
preview (numeric stats for float/int arrays, decoded values for string/object
and 0-d scalar arrays).

Usage:
    python isaac/inspect_npz.py data/*.npz
    python isaac/inspect_npz.py path/to/file.npz --full
    python isaac/inspect_npz.py data/                    # all .npz in a dir
"""
import argparse
import glob
import os
import sys

import numpy as np


def _preview(arr, full=False):
    """Return a one-line-ish preview string for an array."""
    # 0-d (scalar) arrays — e.g. glove_side="left"
    if arr.ndim == 0:
        return f"scalar = {arr.item()!r}"

    # string / object arrays — decode and show the values
    if arr.dtype.kind in ("U", "S", "O"):
        vals = arr.tolist()
        s = repr(vals)
        if not full and len(s) > 200:
            s = s[:200] + f" ... ({arr.size} items)"
        return f"values = {s}"

    # numeric arrays — show range / mean and a sample
    if arr.dtype.kind in ("f", "i", "u", "b"):
        flat = arr.ravel()
        finite = flat[np.isfinite(flat)] if arr.dtype.kind == "f" else flat
        stats = ""
        if finite.size:
            stats = (f"min={finite.min():.4g} max={finite.max():.4g} "
                     f"mean={finite.mean():.4g}")
            if arr.dtype.kind == "f" and finite.size != flat.size:
                stats += f" (+{flat.size - finite.size} non-finite)"
        sample = np.array2string(arr.ravel()[:6], precision=4,
                                 separator=", ", suppress_small=True)
        return f"{stats}  first={sample}"

    return f"(unhandled dtype {arr.dtype})"


def _sim_params_line(npz):
    """One-line summary of the recorded simulator config (isaac/sim_params.py stamps it).

    Imported lazily and defensively: sim_params only needs Isaac inside its reader, but
    inspecting an .npz should never depend on that import working."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sim_params import summarize
        line = summarize(str(npz["sim_params_json"]))
    except Exception:                                       # noqa: BLE001
        line = str(npz["sim_params_json"])
    if "sim_params_hash" in npz:
        line += f"   [{str(npz['sim_params_hash'])}]"
    if "sim_param_changes" in npz and int(npz["sim_param_changes"]) > 0:
        line += (f"\n  WARNING    {int(npz['sim_param_changes'])} knob change(s) DURING this "
                 f"take -- it mixes dynamics")
    return line


def inspect(path, full=False):
    print("=" * 78)
    print(f"FILE: {path}   ({os.path.getsize(path) / 1024:.1f} KiB)")
    print("=" * 78)
    with np.load(path, allow_pickle=True) as npz:
        keys = list(npz.keys())
        print(f"{len(keys)} array(s): {', '.join(keys)}\n")
        # The physics a recording was produced under is the field you actually compare
        # between datasets, so pull it out of the dump and show it decoded up front.
        if "sim_params_json" in keys:
            print(f"  SIMULATOR  {_sim_params_line(npz)}\n")
        # infer frame count from a leading-axis that repeats
        name_w = max((len(k) for k in keys), default=4)
        for k in keys:
            arr = npz[k]
            print(f"  {k:<{name_w}}  {str(arr.dtype):<10} "
                  f"{str(arr.shape):<16} {_preview(arr, full)}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".npz files, globs, or directories")
    ap.add_argument("--full", action="store_true",
                    help="do not truncate long string/object previews")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.npz")))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    if not files:
        print("No .npz files matched.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            inspect(f, full=args.full)
        except Exception as e:
            print(f"!! Failed to read {f}: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
