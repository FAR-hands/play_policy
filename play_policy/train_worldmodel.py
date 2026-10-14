"""Train the action-conditioned diffusion WORLD MODEL on autonomous play data.

  python -m play_policy.train_worldmodel --data_dir <play dir> [<more dirs> ...]

Conditioning : state-history stack [s_t, s_{t-3}, s_{t-6}], s = [hand_joint_pos(22), cube_pose(7)]
               (87-D global conditioning)  +  future actions cmd_q22[t:t+H]  (H x 22, clean input)
Target       : future cube pose cube_state[t:t+H, :7]  (H x 7 = pos3 + quat4)

The defaults ARE the paper recipe (H=50, 3-frame history at stride 3, DiT d512 depth 8 patch 5,
100 diffusion steps / 10 DDIM, AdamW 1e-4 fixed LR, EMA, batch 128, 150k steps; ~40 min on one
RTX 5090). ``ckpt_best.pt`` holds the EMA weights with the lowest held-out normalized cube MSE
and is the checkpoint MPC uses. --max_steps counts grad steps.
"""
from __future__ import annotations

import argparse
import copy
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .data import (
    CUBE_POSE, HAND_DIM, fit_normalizers, load_segments, Normalizer, split_segments,
    WorldModelWindowDataset,
)
from .model import build_world_model
from . import viz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, nargs="+", required=True,
                   help="one or more dirs of play recordings (searched recursively, pooled)")
    p.add_argument("--val_data_dir", type=str, nargs="*", default=[],
                   help="take the validation set EXCLUSIVELY from these dirs instead of holding "
                        "out a --val_frac slice of --data_dir")
    p.add_argument("--H", type=int, default=50, help="future horizon (cube states / actions)")
    p.add_argument("--n_hist", type=int, default=3,
                   help="number of conditioning state frames: [s_t, s_{t-stride}, ...] -> the "
                        "global conditioning is n_hist*29 wide (1 = current state only)")
    p.add_argument("--hist_stride", type=int, default=3,
                   help="frame spacing between the conditioning history states")
    p.add_argument("--rot_weight", type=float, default=None,
                   help="optional ROTATION-to-POSITION loss mass ratio inside the 7-D cube target "
                        "(pos_mass + rot_mass = 7, rot/pos = this). Default: every channel "
                        "weighted 1.0 (= 4/3), which is the paper recipe.")
    # ---- DiT denoiser ----
    p.add_argument("--patch_size", type=int, default=5, help="DiT tokenizer: timesteps per token (H must divide)")
    p.add_argument("--d_model", type=int, default=512, help="DiT token width")
    p.add_argument("--depth", type=int, default=8, help="DiT transformer blocks")
    p.add_argument("--n_heads", type=int, default=8, help="DiT attention heads")
    p.add_argument("--mlp_ratio", type=float, default=4.0, help="DiT MLP expansion")
    p.add_argument("--dsed", type=int, default=256, help="diffusion-step embedding dim")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--n_timesteps", type=int, default=100)
    p.add_argument("--n_sample_steps", type=int, default=10, help="DDIM steps at sampling time")
    # optimization -- FIXED learning rate
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=150000)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1, help="fraction of SEGMENTS held out for val")
    p.add_argument("--resume", type=str, default=None,
                   help="resume from this checkpoint (weights + EMA + optimizer + step) in its own "
                        "run dir; --max_steps is the FINAL step to stop at")
    p.add_argument("--ckpt_dir", type=str, default="runs/world_model")
    p.add_argument("--ckpt_every", type=int, default=25000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=6)
    p.add_argument("--train_eval_every", type=int, default=5000,
                   help="also compute the val metrics on TRAIN windows this often (0=off)")
    p.add_argument("--val_mc_samples", type=int, default=2)
    p.add_argument("--viz_every", type=int, default=5000, help="matplotlib-figure cadence (grad steps)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def crossed(prev, cur, every):
    return every > 0 and cur // every > prev // every


@torch.no_grad()
def eval_val(wm, val_ds, args, device, out_center, out_scale):
    """Normalized cube MSE (MC-averaged DDIM), DDPM loss, and denormalized cube position (mm)
    and orientation (deg) MAE over the horizon."""
    wm.eval()
    rng = np.random.default_rng(args.split_seed)
    cs = torch.as_tensor(out_center, device=device)
    sc = torch.as_tensor(out_scale, device=device)
    cube_mse = ddpm = pos_mm = rot_deg = 0.0
    nb = 0
    for _ in range(args.val_batches):
        cond, act, obj_gt = val_ds.sample_batch(args.batch_size, rng)
        cond, act, obj_gt = cond.to(device), act.to(device), obj_gt.to(device)
        preds = []
        for m in range(args.val_mc_samples):
            g = torch.Generator(device=device).manual_seed(9000 + m)
            preds.append(wm(cond, act, generator=g))
        pred = torch.stack(preds, 0).mean(0)                       # (B,H,7)
        cube_mse += ((pred - obj_gt) ** 2).mean().item()
        ddpm += wm.compute_loss(cond, obj_gt, act).item()
        P = pred * sc + cs                                         # physical units
        G = obj_gt * sc + cs
        pos_mm += (P[..., :3] - G[..., :3]).abs().mean().item() * 1000.0
        qp = torch.nn.functional.normalize(P[..., 3:7], dim=-1)
        qg = torch.nn.functional.normalize(G[..., 3:7], dim=-1)
        dot = (qp * qg).sum(-1).abs().clamp(max=1.0)
        rot_deg += torch.rad2deg(2 * torch.arccos(dot)).mean().item()
        nb += 1
    return {"val_obj_mse_norm": cube_mse / nb, "val_ddpm_loss": ddpm / nb,
            "val_pos_mae_mm": pos_mm / nb, "val_rot_mae_deg": rot_deg / nb}


def save_ckpt(path, step, wm, opt, args, norm_s, norm_a, meta, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world_model": True, "step": step, "policy_type": "dit_worldmodel",
        "state_dim": wm.state_dim, "act_dim": wm.act_dim, "obj_dim": wm.obj_dim,
        "cube_dim": wm.obj_dim, "rot_repr": "quat",
        # conditioning-history stack: cond = [s_t, s_{t-stride}, ...] (n_hist frames, current-first).
        # state_dim above is the FULL cond width (n_hist * frame_dim); frame_dim is per-frame.
        "n_hist": int(args.n_hist), "hist_stride": int(args.hist_stride),
        "frame_dim": int(args.frame_dim),
        "out_cols": [int(c) for c in args.out_cols],
        "horizon": args.H, "backbone": "dit", "objective": "ddpm",
        "patch_size": int(args.patch_size), "d_model": int(args.d_model),
        "depth": int(args.depth), "n_heads": int(args.n_heads),
        "mlp_ratio": float(args.mlp_ratio), "dsed": int(args.dsed),
        "schedule": args.schedule, "n_timesteps": args.n_timesteps, "n_sample_steps": args.n_sample_steps,
        "out_weight": list(args.out_weight or []),
        "norm_state": norm_s.state_dict(), "norm_action": norm_a.state_dict(), "meta": meta,
        "policy": wm.state_dict(), "opt": opt.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    print(f"device: {device}")

    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        for a, k in (("H", "horizon"), ("n_hist", "n_hist"), ("hist_stride", "hist_stride"),
                     ("patch_size", "patch_size"), ("d_model", "d_model"), ("depth", "depth"),
                     ("n_heads", "n_heads"), ("mlp_ratio", "mlp_ratio"), ("dsed", "dsed"),
                     ("schedule", "schedule"), ("n_timesteps", "n_timesteps"),
                     ("n_sample_steps", "n_sample_steps")):
            if k in resume:
                setattr(args, a, resume[k])
        print(f"[resume] {args.resume} (step {resume['step']} -> {args.max_steps})")

    segments, meta = load_segments(args.data_dir, min_len=args.H)
    if args.val_data_dir:                       # fixed val corpus; --data_dir is train-only
        val_segs, _vmeta = load_segments(args.val_data_dir, min_len=args.H)
        train_segs = segments
        print(f"val corpus (held out whole): {args.val_data_dir}  {len(val_segs)} segments  "
              f"(--val_frac ignored)")
    else:
        train_segs, val_segs = split_segments(segments, args.val_frac, args.split_seed)
    if resume is not None:
        norm_s = Normalizer.from_state_dict(resume["norm_state"])
        norm_a = Normalizer.from_state_dict(resume["norm_action"])
    else:
        norm_s, norm_a = fit_normalizers(train_segs)
    hist_kw = dict(n_hist=args.n_hist, hist_stride=args.hist_stride)
    train_ds = WorldModelWindowDataset(train_segs, args.H, norm_s, norm_a, **hist_kw)
    val_ds = WorldModelWindowDataset(val_segs, args.H, norm_s, norm_a, **hist_kw) if val_segs else None
    state_dim, act_dim, out_dim = train_ds.cond_dim, train_ds.action_dim, train_ds.obj_dim
    meta["cube_dim"] = int(out_dim)
    meta["train_dirs"] = [str(d) for d in args.data_dir]
    print(f"data: {args.data_dir}  segments={len(segments)} (train {len(train_segs)} / val {len(val_segs)})")
    print(f"windows: train={len(train_ds)}  val={len(val_ds) if val_ds else 0}  "
          f"(cond={state_dim}={args.n_hist}x(22hand+{train_ds.state_dim - HAND_DIM}cube)"
          f"@stride{args.hist_stride}, act={act_dim}, out={out_dim}=[cube{out_dim}], H={args.H})")
    assert len(train_ds) > 0, "no training windows (segments shorter than H?)"
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None
    # the target lives in the state normalizer's space: out_cols = the cube-pose state columns
    out_center = norm_s.center[train_ds.tgt_cols].copy()
    out_scale = norm_s.scale[train_ds.tgt_cols].copy()
    args.out_cols = [int(c) for c in train_ds.tgt_cols]
    args.frame_dim = int(train_ds.state_dim)                       # per-frame state width (cond/n_hist)
    # Optional rotation/position split of the cube loss (group-normalized: pos_mass + rot_mass
    # = 7 and rot/pos = rot_weight). Off by default: all channels at 1.0, as in the paper.
    args.out_weight = None
    if args.rot_weight is not None:
        n_rot = CUBE_POSE - 3
        pos_mass = CUBE_POSE / (1.0 + args.rot_weight)
        rot_mass = CUBE_POSE - pos_mass
        args.out_weight = [pos_mass / 3.0] * 3 + [rot_mass / n_rot] * n_rot
        print(f"cube loss split: pos 3 ch @{pos_mass / 3.0:.4f} | rot {n_rot} ch @{rot_mass / n_rot:.4f}"
              f"  (rot:pos = {args.rot_weight:.3f}:1)")

    wm = build_world_model(state_dim, args, act_dim=act_dim, obj_dim=out_dim).to(device)
    n_params = sum(p.numel() for p in wm.parameters()) / 1e6
    print(f"world model: dit(patch={args.patch_size} d={args.d_model} depth={args.depth} "
          f"heads={args.n_heads}) (H={args.H}, DDIM {args.n_sample_steps}/{args.n_timesteps})  "
          f"{n_params:.2f}M params  fixed lr={args.lr}")
    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay)

    ema = None
    if args.ema:
        ema = copy.deepcopy(wm)
        for pe in ema.parameters():
            pe.requires_grad_(False)
        ema.eval()
    eval_wm = ema if ema is not None else wm

    start_step, best_val = 0, float("inf")
    if resume is not None:
        DERIVED = ("out_w", "cube_sel", "rot_sel")      # constant buffers some checkpoints carry
        raw = resume.get("policy_train", resume["policy"])
        wm.load_state_dict({k: v for k, v in raw.items() if k not in DERIVED})
        opt.load_state_dict(resume["opt"])
        if ema is not None:
            ema.load_state_dict({k: v for k, v in resume["policy"].items() if k not in DERIVED})
        start_step = int(resume["step"])
        run_dir = Path(args.resume).resolve().parent
        bpath = run_dir / "ckpt_best.pt"
        if bpath.exists():
            best_val = float(torch.load(bpath, map_location="cpu", weights_only=False)
                             .get("best_val_obj_mse_norm", float("inf")))
    else:
        run_dir = Path(args.ckpt_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(f"run dir: {run_dir}\ntensorboard: tensorboard --logdir {Path(args.ckpt_dir).resolve()}")

    def _save(path, extra=None):
        ex = dict(extra or {})
        if ema is not None:
            ex["ema"] = True; ex["policy_train"] = wm.state_dict()   # raw weights, for --resume
        save_ckpt(path, step, eval_wm, opt, args, norm_s, norm_a, meta, extra=ex)

    stop = {"flag": False}
    def _sig(_s, _f):
        if stop["flag"]:
            raise KeyboardInterrupt
        print("\n[signal] Ctrl-C, finishing step + saving..."); stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)

    t0 = time.time(); step = start_step
    try:
        while not stop["flag"] and step < args.max_steps:
            prev = step; step += 1
            wm.train()
            cond, act, obj = train_ds.sample_batch(args.batch_size, rng)
            loss = wm.compute_loss(cond.to(device), obj.to(device), act.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if ema is not None:
                # EMA warmup schedule (power 0.75, capped at 0.9999)
                decay = min(0.9999, 1.0 - float(step) ** -0.75)
                with torch.no_grad():
                    for pe, pt in zip(ema.parameters(), wm.parameters()):
                        pe.lerp_(pt, 1.0 - decay)
            writer.add_scalar("train/loss", float(loss.item()), step)
            if crossed(prev, step, args.log_every):
                rate = (step - start_step) / max(time.time() - t0, 1e-6)
                print(f"[step {step}] loss={loss.item():.5f}  {rate:.1f} steps/s")
            if val_ds is not None and crossed(prev, step, args.val_every):
                m = eval_val(eval_wm, val_ds, args, device, out_center, out_scale)
                for k, v in m.items():
                    writer.add_scalar(f"val/{k}", v, step)
                note = ""
                if m["val_obj_mse_norm"] < best_val:
                    best_val = m["val_obj_mse_norm"]
                    _save(run_dir / "ckpt_best.pt", extra={"best_val_obj_mse_norm": best_val})
                    note = "  *new best -> ckpt_best.pt"
                print(f"  [val @ {step}] obj_mse_norm={m['val_obj_mse_norm']:.5f} "
                      f"(best {best_val:.5f}){note}  pos_mae={m['val_pos_mae_mm']:.1f}mm  "
                      f"rot_mae={m['val_rot_mae_deg']:.1f}deg  ddpm={m['val_ddpm_loss']:.4f}")
            if args.train_eval_every > 0 and crossed(prev, step, args.train_eval_every):
                mt = eval_val(eval_wm, train_ds, args, device, out_center, out_scale)
                for k, v in mt.items():
                    writer.add_scalar(f"train/{k.replace('val_', '')}", v, step)
                print(f"  [train-eval @ {step}] obj_mse_norm={mt['val_obj_mse_norm']:.5f}  "
                      f"pos_mae={mt['val_pos_mae_mm']:.1f}mm  rot_mae={mt['val_rot_mae_deg']:.1f}deg")
            if val_ds is not None and crossed(prev, step, args.viz_every):
                writer.add_figure("viz/pred_vs_gt",
                                  viz.fig_wm_pred_vs_gt(eval_wm, val_ds, device, out_center, out_scale), step)
                writer.add_figure("viz/samples",
                                  viz.fig_wm_samples(eval_wm, val_ds, device, out_center, out_scale), step)
                fig_err, pmm, rdeg = viz.fig_wm_error_vs_horizon(eval_wm, val_ds, device, out_center, out_scale)
                writer.add_figure("viz/error_vs_horizon", fig_err, step)
                print(f"  [viz @ {step}] logged figures  (pos={pmm:.1f}mm rot={rdeg:.1f}deg)")

            if crossed(prev, step, args.ckpt_every):
                _save(run_dir / f"ckpt_{step:08d}.pt")
                print(f"  -> checkpoint saved at step {step}")
    finally:
        _save(run_dir / "ckpt_final.pt")
        print(f"final checkpoint saved (step {step})")
        writer.flush(); writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
