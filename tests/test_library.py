"""Smoke tests for the play_policy library (no simulator needed).

  python tests/test_library.py          # or: python -m pytest tests/

Builds tiny models, runs training losses and sampling, checks the data loader on synthetic
recordings, and -- when the release checkpoints are present under checkpoints/ (or
$DSP_CKPT_DIR) -- loads both paper checkpoints and runs one proposal + prediction batch.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "isaac"))

from play_policy.data import (Normalizer, WindowDataset, WorldModelWindowDataset,  # noqa: E402
                              fit_normalizers, load_segments, quat_geo_deg, split_segments,
                              stack_history)
from play_policy.model import PlayDiTPolicy, build_world_model  # noqa: E402

CKPT_DIR = Path(os.environ.get("DSP_CKPT_DIR", ROOT / "checkpoints"))


def _fake_recording(path, T=400, teleport_at=250, seed=0):
    """A recording in the collector's schema with one cube respawn (teleport) at ``teleport_at``."""
    rng = np.random.default_rng(seed)
    hand = np.cumsum(rng.normal(0, 0.01, (T, 22)), 0).astype(np.float32)
    cube = np.zeros((T, 13), np.float32)
    cube[:, :3] = [0.0, 0.07, 0.36]
    ang = np.cumsum(rng.normal(0, 0.01, T))
    cube[:, 3] = np.cos(ang / 2); cube[:, 4] = np.sin(ang / 2)          # rotation about x (wxyz)
    cube[teleport_at:, 2] += 0.1                                         # respawn jump
    names = np.array([f"j{i}" for i in range(22)])
    np.savez(path, t=np.arange(T, dtype=np.float32) / 60.0, hand_joint_pos=hand, cmd_q22=hand + 0.05,
             cube_state=cube, hand_joint_names=names, cmd_q22_names=names)


def test_loader_and_windows():
    with tempfile.TemporaryDirectory() as d:
        _fake_recording(f"{d}/roll_0000.npz", seed=0)
        _fake_recording(f"{d}/roll_0001.npz", seed=1)
        np.savez(f"{d}/traj_0000.npz", geo=np.zeros(5))                  # sidecar: must be skipped
        segs, meta = load_segments(d, min_len=20)
        assert len(segs) == 4, len(segs)                                 # 2 files x (teleport split)
        assert segs[0]["states"].shape[1] == 29 and segs[0]["actions"].shape[1] == 22
        assert meta["cmd_q22_names"][0] == "j0"
        tr, va = split_segments(segs, 0.25, 0)
        ns, na = fit_normalizers(tr)
        ds = WindowDataset(tr, 16, ns, na)
        cond, chunk = ds.sample_batch(8, np.random.default_rng(0))
        assert cond.shape == (8, 29) and chunk.shape == (8, 16, 22)
        assert float(cond.abs().max()) <= 1.0 + 1e-5
        wds = WorldModelWindowDataset(tr, 10, ns, na, n_hist=3, hist_stride=3)
        c, a, o = wds.sample_batch(8, np.random.default_rng(0))
        assert c.shape == (8, 87) and a.shape == (8, 10, 22) and o.shape == (8, 10, 7)
        # the dataset's history stack equals what the controller builds online with stack_history
        st = int(wds.starts[0])
        online = stack_history(wds.flat_s[:st + 1], 3, 3)
        assert np.array_equal(online, wds.flat_s[[st, st - 3, st - 6]].reshape(-1))


def test_policy_tiny():
    torch.manual_seed(0)
    pol = PlayDiTPolicy(29, horizon=8, n_action_steps=4, patch_size=4, d_model=32, depth=1, n_heads=4)
    s, x = torch.rand(5, 29) * 2 - 1, torch.rand(5, 8, 22) * 2 - 1
    loss = pol.compute_loss(s, x)
    loss.backward()
    pol.eval()
    out = pol(s, eta=1.0, init_noise_scale=1.1)
    assert out.shape == (5, 8, 22) and float(out.abs().max()) <= 1.0
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(pol(s, generator=g1), pol(s, generator=g2))       # seeded => repeatable


def test_world_model_tiny():
    torch.manual_seed(0)
    args = types.SimpleNamespace(H=10, n_timesteps=20, n_sample_steps=4, patch_size=5,
                                 d_model=32, depth=1, n_heads=4)
    wm = build_world_model(87, args, act_dim=22, obj_dim=7)
    s, a, o = torch.rand(4, 87), torch.rand(4, 10, 22), torch.rand(4, 10, 7)
    wm.compute_loss(s, o, a).backward()
    wm.eval()
    assert wm(s, a).shape == (4, 10, 7)


def test_release_checkpoints():
    pp, wp = CKPT_DIR / "dsp_sim_play_policy.pt", CKPT_DIR / "dsp_sim_world_model.pt"
    if not (pp.exists() and wp.exists()):
        print(f"  [skip] release checkpoints not found under {CKPT_DIR}")
        return
    from deploy_core import load_policy
    from play_policy.model import load_world_model_ckpt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol, ns_p, na_p, meta_p, cfg = load_policy(str(pp), dev)
    wm, ck = load_world_model_ckpt(str(wp), device=dev)
    assert cfg["horizon"] == 200 and cfg["n_action_steps"] == 50 and cfg["cube_dims"] == 7
    assert ck["n_hist"] == 3 and ck["hist_stride"] == 3 and wm.horizon == 50
    torch.manual_seed(0)
    state = torch.zeros(16, 29, device=dev)                              # mid-range normalized state
    ch = pol(state, eta=1.0, init_noise_scale=1.1)                       # (16, 200, 22) proposals
    ns_w, na_w = Normalizer.from_state_dict(ck["norm_state"]), Normalizer.from_state_dict(ck["norm_action"])
    perm = [list(meta_p["cmd_q22_names"]).index(n) for n in ck["meta"]["cmd_q22_names"]]
    raw = ch.cpu().numpy() * na_p.scale + na_p.center
    a_wm = torch.as_tensor((raw[:, :50, perm] - na_w.center) / na_w.scale, dtype=torch.float32, device=dev)
    pred = wm(torch.zeros(16, 87, device=dev), a_wm)                      # (16, 50, 7)
    assert ch.shape == (16, 200, 22) and pred.shape == (16, 50, 7)
    q = pred.cpu().numpy()[..., 3:7] * ns_w.scale[25:29] + ns_w.center[25:29]
    assert np.isfinite(quat_geo_deg(q[:, 0], q[:, -1])).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
