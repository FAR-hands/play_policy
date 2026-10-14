"""Strip a training checkpoint down to what inference needs.

  python -m play_policy.export_ckpt <run_dir>/ckpt_best.pt <out>.pt

Training checkpoints carry the optimizer state ('opt') and, for the world model, the raw
(non-EMA) training weights ('policy_train') so that ``--resume`` works; neither is used at
inference. Dropping them shrinks the paper play policy from 90 MB to 30 MB and the world model
from 626 MB to 157 MB. The exported file is re-loaded with the inference loaders before it is
reported, and can no longer be resumed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

TRAIN_ONLY_KEYS = ("opt", "policy_train")


def export(src: str, dst: str) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for k in TRAIN_ONLY_KEYS:
        ck.pop(k, None)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    return ck


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    ck = export(a.src, a.dst)
    if ck.get("world_model"):
        from .model import load_world_model_ckpt
        load_world_model_ckpt(a.dst)
        kind = "world model"
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
        from deploy_core import load_policy
        load_policy(a.dst)
        kind = "play policy"
    print(f"{kind} step {ck.get('step')}: {os.path.getsize(a.src) / 1e6:.1f} MB -> "
          f"{os.path.getsize(a.dst) / 1e6:.1f} MB  ({a.dst}, verified loadable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
