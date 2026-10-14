"""Training-time visualizations logged to TensorBoard (no simulator needed).

All figures plot in *physical* units (radians for joints) by denormalizing the
policy's normalized outputs with the action ``Normalizer``.

  - ``fig_pred_vs_gt``:  predicted vs ground-truth action chunks over the horizon
                         for a few held-out windows and a few joints.
  - ``fig_multimodal``:  many DDIM samples from SEVERAL held-out states (one row per
                         state), to show the policy's action distribution per joint.
  - ``fig_joint_mae``:   per-joint mean abs error (rad) over a val batch.
  - ``fig_wm_*``:        world-model analogs (predicted vs GT cube trajectory, samples,
                         error vs horizon) logged by ``train_worldmodel.py``.
"""

from __future__ import annotations

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


@torch.no_grad()
def fig_pred_vs_gt(policy, val_ds, norm_a, device, n_windows=3, n_joints=6, seed=0):
    """Sample ``n_windows`` held-out windows; overlay GT vs predicted chunk for ``n_joints`` joints."""
    policy.eval()
    rng = np.random.default_rng(seed)
    cond, chunk_gt = val_ds.sample_batch(n_windows, rng)
    pred = policy(cond.to(device)).cpu().numpy()            # (W, H, A) normalized
    gt = chunk_gt.numpy()
    pred = norm_a.denormalize(pred)                         # -> radians
    gt = norm_a.denormalize(gt)
    H = pred.shape[1]
    joints = np.arange(min(n_joints, pred.shape[2]))
    fig, axes = plt.subplots(n_windows, 1, figsize=(7, 2.4 * n_windows), squeeze=False)
    x = np.arange(H)
    for w in range(n_windows):
        ax = axes[w][0]
        for ji, j in enumerate(joints):
            c = plt.cm.tab10(ji % 10)
            ax.plot(x, gt[w, :, j], color=c, lw=2.0, label=f"j{j}" if w == 0 else None)
            ax.plot(x, pred[w, :, j], color=c, lw=1.4, ls="--")
        ax.set_ylabel("rad")
        ax.set_title(f"window {w}  (solid=GT, dashed=pred)", fontsize=9)
    axes[-1][0].set_xlabel("horizon step")
    if n_windows:
        axes[0][0].legend(ncol=len(joints), fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_multimodal(policy, val_ds, norm_a, device, n_samples=16, n_joints=6, seed=1, n_windows=4):
    """Draw ``n_samples`` independent DDIM samples from EACH of ``n_windows`` held-out states.

    One row per state, one column per joint: shows the policy's action distribution (spread /
    multimodality) and how it varies across states, not just at a single lucky one. All
    ``n_windows * n_samples`` chunks are sampled in one batched forward."""
    policy.eval()
    rng = np.random.default_rng(seed)
    cond, chunk_gt = val_ds.sample_batch(n_windows, rng)
    W = cond.shape[0]
    state = cond.to(device).repeat_interleave(n_samples, dim=0)          # (W*S, D)
    g = torch.Generator(device=device).manual_seed(1000)
    x0 = torch.randn(W * n_samples, policy.horizon, policy.action_dim,
                     device=device, generator=g)
    samples = policy(state, x_init=x0).cpu().numpy()                     # (W*S, H, A)
    samples = norm_a.denormalize(samples).reshape(W, n_samples, policy.horizon, -1)
    gt = norm_a.denormalize(chunk_gt.numpy())                            # (W, H, A)
    H = samples.shape[2]
    joints = np.arange(min(n_joints, samples.shape[3]))
    fig, axes = plt.subplots(W, len(joints), figsize=(2.4 * len(joints), 2.2 * W), squeeze=False)
    x = np.arange(H)
    for w in range(W):
        for ji, j in enumerate(joints):
            ax = axes[w][ji]
            for s in range(n_samples):
                ax.plot(x, samples[w, s, :, j], color="tab:blue", alpha=0.25, lw=0.8)
            ax.plot(x, gt[w, :, j], color="k", lw=2.0)
            if w == 0:
                ax.set_title(f"joint {j}", fontsize=9)
            if w == W - 1:
                ax.set_xlabel("step")
            if ji == 0:
                ax.set_ylabel(f"state {w}\nrad (black=GT)", fontsize=8)
    fig.suptitle(f"{n_samples} DDIM samples per state, {W} held-out states", fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_wm_pred_vs_gt(wm, val_ds, device, obj_center, obj_scale, n_windows=3, seed=0):
    """World-model analog of ``fig_pred_vs_gt``: predicted vs GT future OBJECT trajectory
    over the horizon for a few held-out windows (position mm | quaternion), solid=GT."""
    wm.eval()
    rng = np.random.default_rng(seed)
    cond, act, obj_gt = val_ds.sample_batch(n_windows, rng)
    pred = wm(cond.to(device), act.to(device)).cpu().numpy()          # (W,H,7) normalized
    P = pred * obj_scale + obj_center
    G = obj_gt.numpy() * obj_scale + obj_center
    H = pred.shape[1]; x = np.arange(H)
    fig, axes = plt.subplots(n_windows, 2, figsize=(10, 2.3 * n_windows), squeeze=False)
    pos_lab, q_lab = ["x", "y", "z"], ["qw", "qx", "qy", "qz"]
    for w in range(n_windows):
        axp, axq = axes[w][0], axes[w][1]
        for d in range(3):
            c = plt.cm.tab10(d)
            axp.plot(x, G[w, :, d] * 1000, color=c, lw=2.0, label=pos_lab[d] if w == 0 else None)
            axp.plot(x, P[w, :, d] * 1000, color=c, lw=1.4, ls="--")
        for d in range(min(4, pred.shape[2] - 3)):
            c = plt.cm.tab10(d)
            axq.plot(x, G[w, :, 3 + d], color=c, lw=2.0, label=q_lab[d] if w == 0 else None)
            axq.plot(x, P[w, :, 3 + d], color=c, lw=1.4, ls="--")
        axp.set_ylabel("pos (mm)"); axq.set_ylabel("quat")
        axp.set_title(f"window {w}: position  (solid=GT, dashed=pred)", fontsize=9)
        axq.set_title(f"window {w}: orientation", fontsize=9)
    axes[-1][0].set_xlabel("horizon step"); axes[-1][1].set_xlabel("horizon step")
    axes[0][0].legend(ncol=3, fontsize=7); axes[0][1].legend(ncol=4, fontsize=7)
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_wm_samples(wm, val_ds, device, obj_center, obj_scale, n_samples=12, seed=1):
    """World-model analog of ``fig_multimodal``: many samples from ONE (state, action-seq)
    condition -> the model's predictive spread per object dim (tight = confident)."""
    wm.eval()
    rng = np.random.default_rng(seed)
    cond, act, obj_gt = val_ds.sample_batch(1, rng)
    state = cond.to(device).repeat(n_samples, 1)
    acts = act.to(device).repeat(n_samples, 1, 1)
    rows = [torch.randn(wm.horizon, wm.obj_dim, device=device,
                        generator=torch.Generator(device=device).manual_seed(1000 + s))
            for s in range(n_samples)]
    S = wm(state, acts, x_init=torch.stack(rows, 0)).cpu().numpy() * obj_scale + obj_center
    G = obj_gt.numpy()[0] * obj_scale + obj_center
    labs = ["x (mm)", "y (mm)", "z (mm)", "qw", "qx", "qy", "qz"][:S.shape[2]]
    x = np.arange(S.shape[1])
    fig, axes = plt.subplots(1, len(labs), figsize=(2.2 * len(labs), 2.6), squeeze=False)
    for d in range(len(labs)):
        ax = axes[0][d]; sc = 1000 if d < 3 else 1
        for s in range(S.shape[0]):
            ax.plot(x, S[s, :, d] * sc, color="tab:blue", alpha=0.3, lw=0.8)
        ax.plot(x, G[:, d] * sc, color="k", lw=2.0)
        ax.set_title(labs[d], fontsize=9); ax.set_xlabel("step")
    fig.suptitle(f"{S.shape[0]} world-model samples from one (state, action seq)   black=GT", fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_wm_error_vs_horizon(wm, val_ds, device, obj_center, obj_scale, batch=128, mc=2, seed=2):
    """World-model analog of ``fig_joint_mae``: how position (mm) and rotation (deg) error
    grow across the prediction horizon (compounding). Returns ``(fig, mean_pos_mm, mean_rot_deg)``."""
    wm.eval()
    rng = np.random.default_rng(seed)
    cond, act, obj_gt = val_ds.sample_batch(min(batch, len(val_ds)), rng)
    cond, act = cond.to(device), act.to(device)
    preds = []
    for m in range(mc):
        g = torch.Generator(device=device).manual_seed(7 + 100 * m)
        preds.append(wm(cond, act, generator=g).cpu().numpy())
    pred = np.mean(preds, 0)
    P = pred * obj_scale + obj_center
    G = obj_gt.numpy() * obj_scale + obj_center
    pos_mm = np.linalg.norm(P[:, :, :3] - G[:, :, :3], axis=-1).mean(0) * 1000.0   # (H,)
    qp = P[:, :, 3:7] / (np.linalg.norm(P[:, :, 3:7], axis=-1, keepdims=True) + 1e-9)
    qg = G[:, :, 3:7] / (np.linalg.norm(G[:, :, 3:7], axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.abs((qp * qg).sum(-1)), 0, 1)
    rot_deg = np.degrees(2 * np.arccos(dot)).mean(0)                                # (H,)
    x = np.arange(len(pos_mm))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(x, pos_mm, color="tab:blue", label="position (mm)")
    ax.set_xlabel("horizon step"); ax.set_ylabel("pos MAE (mm)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(x, rot_deg, color="tab:red", label="rotation (deg)")
    ax2.set_ylabel("rot MAE (deg)", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title(f"world-model error vs horizon  (mean pos={pos_mm.mean():.1f}mm  rot={rot_deg.mean():.1f}deg)",
                 fontsize=9)
    fig.tight_layout()
    return fig, float(pos_mm.mean()), float(rot_deg.mean())


@torch.no_grad()
def fig_joint_mae(policy, val_ds, norm_a, device, batch=256, mc=4, seed=2):
    """Per-joint MAE (rad) between MC-averaged predicted chunks and GT over one val batch."""
    policy.eval()
    rng = np.random.default_rng(seed)
    cond, chunk_gt = val_ds.sample_batch(min(batch, len(val_ds)), rng)
    cond = cond.to(device)
    preds = []
    for m in range(mc):
        g = torch.Generator(device=device).manual_seed(7 + 100 * m)
        preds.append(policy(cond, generator=g).cpu().numpy())
    pred = np.mean(preds, axis=0)
    pred = norm_a.denormalize(pred)
    gt = norm_a.denormalize(chunk_gt.numpy())
    mae = np.abs(pred - gt).mean(axis=(0, 1))   # (A,)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.bar(np.arange(len(mae)), mae, color="tab:purple")
    ax.set_xlabel("joint index (cmd_q22 order)")
    ax.set_ylabel("MAE (rad)")
    ax.set_title(f"per-joint action MAE over {cond.shape[0]} val windows  (mean={mae.mean():.4f} rad)", fontsize=9)
    fig.tight_layout()
    return fig, float(mae.mean())
