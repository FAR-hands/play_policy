"""Isaac-free core for closed-loop deployment of the diffusion play policy.

Checkpoint loading, observation assembly, action decoding, and the name-based joint remap
between the policy and the articulation live here, so they can be used (and tested) without
a simulator. ``isaac/wm_mpc_eval_batch.py`` imports ``load_policy`` from this file.

Ordering contract (everything is remapped *by joint name* via the checkpoint's legends, so it
is robust to articulation order differences):

  state  = [ hand_joint_pos reordered to meta['hand_joint_names'] (22),
             cube_state[:cube_dims] ]                      -> normalized -> policy
  action = cmd_q22 in meta['cmd_q22_names'] order (22)     <- policy, denormalized
           applied to articulation columns ``act_col``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Importable with the repo root on sys.path (python -m / isaac scripts) or standalone.
try:
    from play_policy.model import PlayDiTPolicy
    from play_policy.data import Normalizer
except ModuleNotFoundError:  # pragma: no cover - path bootstrap
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from play_policy.model import PlayDiTPolicy
    from play_policy.data import Normalizer


def load_policy(ckpt_path: str, device: str = "cpu"):
    """Rebuild the policy + normalizers from a training checkpoint.

    Returns ``(policy, norm_s, norm_a, meta, cfg)`` where ``cfg`` carries the
    fields the Isaac loop needs (cube_dims, horizon, n_action_steps).
    """
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if sd.get("policy_type") != "dit":
        raise ValueError(f"{ckpt_path}: policy_type={sd.get('policy_type')!r}; "
                         f"this release loads DiT play policies only")
    unsupported = {k: sd[k] for k in ("latent_dim", "obs_encoder", "predict_obj")
                   if sd.get(k)}
    if unsupported:
        raise ValueError(f"{ckpt_path}: checkpoint uses features outside this release: {unsupported}")
    policy = PlayDiTPolicy(
        sd["state_dim"], action_dim=sd["action_dim"],
        horizon=sd["horizon"], n_action_steps=sd["n_action_steps"],
        patch_size=int(sd.get("patch_size", 4)), d_model=int(sd.get("d_model", 256)),
        depth=int(sd.get("depth", 6)), n_heads=int(sd.get("n_heads", 8)),
        mlp_ratio=float(sd.get("mlp_ratio", 4.0)), dsed=int(sd.get("dsed", 256)),
        n_timesteps=sd["n_timesteps"], n_sample_steps=sd["n_sample_steps"],
        schedule=sd.get("schedule", "cosine"))
    policy.load_state_dict(sd["policy"])
    policy.to(device).eval()
    norm_s = Normalizer.from_state_dict(sd["norm_state"])
    norm_a = Normalizer.from_state_dict(sd["norm_action"])
    cfg = {
        "cube_dims": int(sd.get("cube_dims", 7)),
        "horizon": int(sd["horizon"]),
        "n_action_steps": int(sd["n_action_steps"]),
        "policy_type": sd["policy_type"],
        "step": int(sd.get("step", -1)),
        "obj_dim": 0,             # the whole chunk is executable joint commands
    }
    return policy, norm_s, norm_a, sd["meta"], cfg


def joint_perms(art_names: list[str], meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Name-based index maps between the live articulation and the checkpoint.

    ``hand_perm``: gather articulation joint_pos into the policy's hand-state order
                   (``meta['hand_joint_names']``).
    ``act_col``:   scatter the policy's 22 action dims (``meta['cmd_q22_names']``
                   order) into articulation target columns.
    """
    hand_names = list(meta["hand_joint_names"])
    cmd_names = list(meta["cmd_q22_names"])
    missing_h = [n for n in hand_names if n not in art_names]
    missing_c = [n for n in cmd_names if n not in art_names]
    if missing_h or missing_c:
        raise KeyError(f"joints missing from articulation: hand={missing_h} cmd={missing_c}")
    hand_perm = np.array([art_names.index(n) for n in hand_names], dtype=np.int64)
    act_col = np.array([art_names.index(n) for n in cmd_names], dtype=np.int64)
    return hand_perm, act_col


def make_obs(hand_joint_pos_art: np.ndarray, cube_state13: np.ndarray,
             hand_perm: np.ndarray, cube_dims: int, norm_s: Normalizer,
             device: str = "cpu") -> torch.Tensor:
    """Assemble + normalize the policy conditioning state. Returns ``(1, state_dim)``.

    ``hand_joint_pos_art`` (22,) is in articulation order; ``cube_state13`` (13,) is
    ``root_state_w`` = pos[3]+quat_wxyz[4]+linvel[3]+angvel[3] (env origin removed).
    """
    hand = np.asarray(hand_joint_pos_art, np.float32)[hand_perm]
    cube = np.asarray(cube_state13, np.float32)[:cube_dims]
    state = np.concatenate([hand, cube], axis=0)
    state_n = norm_s.normalize(state).astype(np.float32)
    return torch.from_numpy(state_n[None]).to(device)


@torch.no_grad()
def decode_chunk(policy, cond: torch.Tensor, norm_a: Normalizer,
                 generator: torch.Generator | None = None,
                 eta: float = 0.0, init_noise_scale: float = 1.0) -> np.ndarray:
    """Sample one chunk and denormalize. Returns ``(H, 22)`` joint targets (radians,
    ``meta['cmd_q22_names']`` order).

    ``eta``>0 enables stochastic reverse sampling and ``init_noise_scale``>1 scales the
    initial latent -- both widen the sampled trajectory (play collection uses 1.0 / 1.1)."""
    chunk_n = policy(cond, generator=generator, eta=eta,
                     init_noise_scale=init_noise_scale).cpu().numpy()[0]  # (H, 22) normalized
    return norm_a.denormalize(chunk_n).astype(np.float32)
