"""Train the diffusion play policy (behavior cloning) on human teleop recordings.

  python -m play_policy.train --data_dir <teleop dir>

State  = [hand_joint_pos(22), cube_pose(7)]   -> conditioning (normalized, 29-D)
Action = cmd_q22(22) future joint commands, predicted as an H-step chunk by a DiT denoiser
         (DDPM training, DDIM sampling), of which n_action_steps are executed before replanning.

The defaults ARE the paper recipe (H=200 / exec 50, DiT d256 depth 6 patch 4, 50 diffusion
steps / 10 DDIM, Adam 1e-4, batch 64, 40k steps; ~6 min on one RTX 5090). ``ckpt_best.pt``
tracks the lowest held-out action MSE (MC-averaged DDIM samples) and is the checkpoint to
deploy. TensorBoard logs train/val losses and figures (predicted-vs-GT chunks, samples,
per-joint MAE). --max_steps counts GRADIENT steps.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .data import CUBE_POSE, fit_normalizers, load_segments, Normalizer, split_segments, WindowDataset
from .model import build_policy
from . import viz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, nargs="+", required=True,
                   help="one or more dirs of recordings (searched recursively, pooled)")
    # Policy / horizon.
    p.add_argument("--H", type=int, default=200, help="action chunk horizon (predicted per call)")
    p.add_argument("--n_action_steps", type=int, default=50, help="executed before replanning")
    p.add_argument("--patch_size", type=int, default=4, help="DiT tokenizer: timesteps per token (must divide H)")
    p.add_argument("--d_model", type=int, default=256, help="DiT token width")
    p.add_argument("--depth", type=int, default=6, help="DiT transformer blocks")
    p.add_argument("--n_heads", type=int, default=8, help="DiT attention heads")
    p.add_argument("--mlp_ratio", type=float, default=4.0, help="DiT MLP expansion")
    p.add_argument("--dsed", type=int, default=256, help="diffusion-step embedding dim")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine", help="beta schedule")
    p.add_argument("--n_timesteps", type=int, default=50, help="DDPM training timesteps")
    p.add_argument("--n_sample_steps", type=int, default=10, help="DDIM inference steps")
    # Optimization.
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (constant)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=40000, help="GRADIENT steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15, help="fraction of SEGMENTS held out for val")
    p.add_argument("--resume", type=str, default=None,
                   help="resume from this checkpoint: load weights+optimizer+step and keep "
                        "training in the checkpoint's own run dir. Architecture flags are taken "
                        "from the checkpoint; --max_steps is the FINAL step to stop at.")
    # Cadence / IO.
    p.add_argument("--ckpt_dir", type=str, default="runs/play_policy")
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--val_mc_samples", type=int, default=4, help="DDIM samples averaged per val window")
    p.add_argument("--viz_every", type=int, default=5000, help="matplotlib-figure cadence (grad steps)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def seed_everything(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def crossed(prev: int, cur: int, every: int) -> bool:
    """True if a multiple of ``every`` lies in ``(prev, cur]``."""
    return every > 0 and cur // every > prev // every


@torch.no_grad()
def eval_val(policy, val_ds, args, device) -> dict:
    """Normalized action-MSE (MC-averaged DDIM) + the stable DDPM-loss curve on held-out windows."""
    policy.eval()
    rng = np.random.default_rng(args.split_seed)  # deterministic windows across calls
    mse_sum, n, ddpm_sum = 0.0, 0, 0.0
    for _ in range(args.val_batches):
        cond, chunk_gt = val_ds.sample_batch(args.batch_size, rng)
        cond, chunk_gt = cond.to(device), chunk_gt.to(device)
        preds = []
        for m in range(args.val_mc_samples):
            g = torch.Generator(device=device).manual_seed(10_000 * m)
            preds.append(policy(cond, generator=g))
        pred = torch.stack(preds, 0).mean(0)
        mse_sum += ((pred - chunk_gt) ** 2).mean().item() * cond.shape[0]
        n += cond.shape[0]
        ddpm_sum += policy.compute_loss(cond, chunk_gt).item()
    return {"val_action_mse_norm": mse_sum / max(1, n),
            "val_ddpm_loss": ddpm_sum / args.val_batches}


def save_checkpoint(path, step, policy, opt, args, norm_s, norm_a, meta, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step, "policy_type": "dit",
        "state_dim": policy.state_dim, "action_dim": policy.action_dim,
        "obj_dim": 0, "cube_dims": CUBE_POSE,
        "horizon": args.H, "n_action_steps": args.n_action_steps,
        "patch_size": args.patch_size, "d_model": args.d_model, "depth": args.depth,
        "n_heads": args.n_heads, "mlp_ratio": args.mlp_ratio, "dsed": args.dsed,
        "schedule": args.schedule,
        "n_timesteps": args.n_timesteps, "n_sample_steps": args.n_sample_steps,
        "norm_state": norm_s.state_dict(), "norm_action": norm_a.state_dict(),
        "meta": meta,  # joint-name legends etc.
        "policy": policy.state_dict(), "opt": opt.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    rng = seed_everything(args.seed)
    device = args.device
    print(f"device: {device}")

    # Resume: adopt the checkpoint's architecture BEFORE building the model so tensor shapes
    # match the saved weights exactly (the user only needs to pass --resume and --max_steps).
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("H", "n_action_steps", "patch_size", "d_model", "depth", "n_heads",
                    "mlp_ratio", "dsed", "schedule", "n_timesteps", "n_sample_steps"):
            ck_key = "horizon" if key == "H" else key
            if ck_key in resume:
                setattr(args, key, resume[ck_key])
        print(f"[resume] {args.resume}  (step {resume['step']} -> max_steps {args.max_steps}; H={args.H})")

    # teleport + t-gap segmentation, recursive, multi-dir (state=[hand22, cube7], action=cmd_q22)
    segments, meta = load_segments(args.data_dir, min_len=args.H)
    train_segs, val_segs = split_segments(segments, args.val_frac, args.split_seed)
    if resume is not None:
        # reuse the checkpoint's normalizers so inputs map exactly as the loaded policy expects
        norm_s = Normalizer.from_state_dict(resume["norm_state"])
        norm_a = Normalizer.from_state_dict(resume["norm_action"])
    else:
        norm_s, norm_a = fit_normalizers(train_segs)
    train_ds = WindowDataset(train_segs, args.H, norm_s, norm_a)
    val_ds = WindowDataset(val_segs, args.H, norm_s, norm_a) if val_segs else None
    state_dim = train_ds.state_dim
    print(f"data: {args.data_dir}  segments={len(segments)} (train {len(train_segs)} / val {len(val_segs)})")
    act_dim = train_ds.action_dim
    print(f"windows: train={len(train_ds)}  val={len(val_ds) if val_ds else 0}  "
          f"(state_dim={state_dim} = 22 hand + {CUBE_POSE} cube, action_dim={act_dim})")
    assert len(train_ds) > 0, "no training windows (segments shorter than H?)"
    if val_ds is None or len(val_ds) == 0:
        print("  WARNING: no val windows; validation/viz disabled")
        val_ds = None

    policy = build_policy("dit", state_dim, args, action_dim=act_dim).to(device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"policy: dit (H={args.H}, exec={args.n_action_steps}, "
          f"DDIM {args.n_sample_steps}/{args.n_timesteps})  {n_params:.2f}M params")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    start_step = 0
    best_val = float("inf")
    if resume is not None:
        policy.load_state_dict(resume["policy"])
        opt.load_state_dict(resume["opt"])
        start_step = int(resume["step"])
        # continue in the checkpoint's own run dir so tb + checkpoint numbering carry on
        run_dir = Path(args.resume).resolve().parent
        # keep the existing best so a worse early val doesn't clobber ckpt_best.pt
        best_ckpt = run_dir / "ckpt_best.pt"
        if best_ckpt.exists():
            bd = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            best_val = float(bd.get("best_val_action_mse_norm", float("inf")))
            print(f"[resume] existing best val action_mse_norm = {best_val:.5f}")
    else:
        run_dir = Path(args.ckpt_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(f"run dir: {run_dir}")
    print(f"tensorboard: tensorboard --logdir {Path(args.ckpt_dir).resolve()}")

    def _save(path, extra=None):
        save_checkpoint(path, step, policy, opt, args, norm_s, norm_a, meta, extra=extra)

    stop = {"flag": False}
    def _on_sigint(_s, _f):
        if stop["flag"]:
            raise KeyboardInterrupt
        print("\n[signal] Ctrl-C received, finishing current step and saving...")
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    t_start = time.time()
    step = start_step
    try:
        while not stop["flag"] and step < args.max_steps:
            prev = step
            step += 1
            policy.train()
            cond, chunk = train_ds.sample_batch(args.batch_size, rng)
            loss = policy.compute_loss(cond.to(device), chunk.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            writer.add_scalar("train/loss", float(loss.item()), step)

            if crossed(prev, step, args.log_every):
                rate = (step - start_step) / max(time.time() - t_start, 1e-6)
                print(f"[step {step}] loss={loss.item():.5f}  {rate:.1f} steps/s")

            if val_ds is not None and crossed(prev, step, args.val_every):
                m = eval_val(policy, val_ds, args, device)
                for k, v in m.items():
                    writer.add_scalar(f"val/{k}", v, step)
                if m["val_action_mse_norm"] < best_val:
                    best_val = m["val_action_mse_norm"]
                    _save(run_dir / "ckpt_best.pt", extra={"best_val_action_mse_norm": best_val})
                    note = "  *new best -> ckpt_best.pt"
                else:
                    note = ""
                writer.add_scalar("val/val_action_mse_norm_best", best_val, step)
                print(f"  [val @ {step}] action_mse_norm={m['val_action_mse_norm']:.5f} "
                      f"(best {best_val:.5f}){note}  ddpm_loss={m['val_ddpm_loss']:.4f}")

            if val_ds is not None and crossed(prev, step, args.viz_every):
                writer.add_figure("viz/pred_vs_gt", viz.fig_pred_vs_gt(policy, val_ds, norm_a, device), step)
                writer.add_figure("viz/multimodal", viz.fig_multimodal(policy, val_ds, norm_a, device), step)
                fig_mae, mae = viz.fig_joint_mae(policy, val_ds, norm_a, device)
                writer.add_figure("viz/joint_mae", fig_mae, step)
                writer.add_scalar("val/action_mae_rad", mae, step)
                print(f"  [viz @ {step}] logged figures  (per-joint MAE={mae:.4f} rad)")

            if crossed(prev, step, args.ckpt_every):
                _save(run_dir / f"ckpt_{step:08d}.pt")
                print(f"  -> checkpoint saved at step {step}")
    finally:
        _save(run_dir / "ckpt_final.pt")
        print(f"final checkpoint saved  (step {step})")
        writer.flush()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
