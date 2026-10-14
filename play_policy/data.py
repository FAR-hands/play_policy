"""Load Sharpa-hand recordings (teleop takes and autonomous play) and sample training windows.

Every ``*.npz`` recording (written by ``isaac/teleop_isaac.py`` or by the batched collector
``isaac/wm_mpc_eval_batch.py --mode policy --record``) carries, per control frame:

    t               (T,)        timestamps (s); teleop takes have gaps where resets were trimmed
    hand_joint_pos  (T, 22)     observed hand joints (articulation order)
    cmd_q22         (T, 22)     commanded joint targets (retarget order)  <- the ACTION
    cube_state      (T, 13)     pos[3] + quat_wxyz[4] + lin_vel[3] + ang_vel[3]
    hand_joint_names, cmd_q22_names   the two joint legends (stored in every checkpoint)

Files without those keys (e.g. the collector's ``traj_*.npz`` planner sidecars) are skipped.

State = [hand_joint_pos(22), cube_pose(7)] (29-D); action = cmd_q22 (22-D). A recording is not
one contiguous trajectory: the cube is respawned when it drops, so each file is split into
contiguous SEGMENTS at every reset (cube teleport, ``t`` going backwards, ``t`` gap) and no
training window ever straddles one. Everything is min-max normalized to [-1, 1] by a
``Normalizer`` fit on the TRAIN split only.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

HAND_DIM = 22       # hand_joint_pos / cmd_q22
ACTION_DIM = 22     # cmd_q22
CUBE_FULL = 13      # pos3 + quat4 + linvel3 + angvel3 (as recorded)
CUBE_POSE = 7       # pos3 + quat4 (what the models see)


# ---------------------------------------------------------------------------
# Rotation helpers (quaternions are wxyz throughout)
# ---------------------------------------------------------------------------

def quat_wxyz_to_R9(q):
    """(...,4) wxyz quaternion -> (...,9) row-major-flattened 3x3 rotation matrix.
    Non-finite / zero-norm rows pass through as NaN rows."""
    from scipy.spatial.transform import Rotation as _Rot
    q = np.asarray(q, np.float64)
    xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    norm = np.linalg.norm(xyzw, axis=-1, keepdims=True)
    bad = ~np.isfinite(norm[..., 0]) | (norm[..., 0] < 1e-6)          # (...,)
    xyzw = np.where(bad[..., None], np.array([0.0, 0.0, 0.0, 1.0]), xyzw)
    xyzw = xyzw / (np.linalg.norm(xyzw, axis=-1, keepdims=True) + 1e-12)
    R = _Rot.from_quat(xyzw.reshape(-1, 4)).as_matrix()               # (n,3,3)
    out = R.reshape(*q.shape[:-1], 9).astype(np.float32)              # row-major flatten
    out[bad] = np.nan
    return out


def R9_to_matrix(x):
    """(...,9) -> (...,3,3) nearest rotation via SVD special-orthogonalization
    (U @ diag(1,1,det(U Vt)) @ Vt). Handles arbitrary (off-manifold) 9-vectors."""
    M = np.asarray(x, np.float64).reshape(*np.shape(x)[:-1], 3, 3)
    U, _, Vt = np.linalg.svd(M)
    D = np.zeros(M.shape)
    D[..., 0, 0] = 1.0; D[..., 1, 1] = 1.0
    D[..., 2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def geodesic_deg_from_matrices(Ra, Rb):
    """(...,3,3),(...,3,3) -> geodesic angle (deg) = arccos((tr(Ra^T Rb)-1)/2)."""
    Rr = np.einsum('...ij,...ik->...jk', Ra, Rb)                      # Ra^T Rb
    tr = np.clip((np.trace(Rr, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def quat_geo_deg(qa, qb):
    """Geodesic angle (deg) between wxyz quaternions; broadcasts over leading dims.
    Uses |dot| so it is sign-agnostic (q and -q are the same rotation)."""
    qa = np.asarray(qa, np.float64); qb = np.asarray(qb, np.float64)
    qa = qa / (np.linalg.norm(qa, axis=-1, keepdims=True) + 1e-12)
    qb = qb / (np.linalg.norm(qb, axis=-1, keepdims=True) + 1e-12)
    d = np.clip(np.abs((qa * qb).sum(-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def goal_quat_from_npz(path):
    """Goal orientation + position from a short HELD recording.

    Returns (quat_wxyz (4,), pos (3,)): the sign-aligned unit-mean of ``cube_state[:, 3:7]``
    and the mean of ``cube_state[:, :3]`` over every frame. The frames are hemisphere-aligned
    to the first one before averaging so q/-q double cover cannot cancel out. A one-frame file
    ``np.savez(p, cube_state=[[0, 0, 0, w, x, y, z]])`` pins an exact goal."""
    d = np.load(path)
    cs = np.asarray(d["cube_state"], np.float64)
    q = cs[:, 3:7]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    q = np.where((q @ q[0])[:, None] < 0.0, -q, q)
    g = q.mean(0); g = g / np.linalg.norm(g)
    return g, cs[:, :3].mean(0)


# ---------------------------------------------------------------------------
# Normalization (min-max to [-1, 1], parameterized as center/scale per dim)
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-dim affine map ``x_n = (x - center) / scale`` into ~[-1, 1].

    ``center = (min + max) / 2``, ``scale = (max - min) / 2``. Constant dims
    (scale < eps) get scale=1 so they map to 0 instead of exploding.
    """

    def __init__(self, center: np.ndarray, scale: np.ndarray):
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        lo = x.min(axis=0)
        hi = x.max(axis=0)
        center = (lo + hi) * 0.5
        scale = (hi - lo) * 0.5
        scale = np.where(scale < eps, 1.0, scale)
        return cls(center, scale)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center) / self.scale

    def denormalize(self, x_n: np.ndarray) -> np.ndarray:
        return x_n * self.scale + self.center

    def state_dict(self) -> dict:
        return {"center": self.center.copy(), "scale": self.scale.copy()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["center"], np.float32), np.asarray(d["scale"], np.float32))


# ---------------------------------------------------------------------------
# Loading + segmentation
# ---------------------------------------------------------------------------

def _quat_geo_deg(q):
    """Per-consecutive-frame geodesic angle (deg) for a (T,4) wxyz quat array -> (T-1,)."""
    d = np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def load_segments(
    data_dir,
    min_len: int = 1,
    pos_jump: float = 0.03,
    ang_jump: float = 45.0,
    gap_factor: float = 5.0,
    min_gap_s: float = 0.5,
) -> tuple[list[dict], dict]:
    """Load every recording under ``data_dir`` (one dir or a list; searched recursively) and
    split it into contiguous segments.

    Returns ``(segments, meta)``: each segment is ``{"states": (L, 29), "actions": (L, 22),
    "file": relative path, "seg": i}`` in raw units, and ``meta`` carries the joint legends
    stored in every checkpoint. Segments shorter than ``min_len`` frames are dropped.

    A segment boundary is placed before frame i when
      * the cube TELEPORTS between i-1 and i (position jump > ``pos_jump`` m or orientation
        jump > ``ang_jump`` deg): the collector respawns a dropped cube without a ``t`` gap;
      * ``t`` goes backwards (a new episode inside one file) or jumps by more than both
        ``gap_factor`` x the median step and ``min_gap_s`` (teleop takes trim the frames around
        each reset at save time);
      * the ``episode`` index changes (when the file carries one), or a non-finite row starts
        or ends (non-finite spans are dropped).
    """
    dirs = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
    files = [(Path(dd), fp) for dd in dirs
             for fp in sorted(glob.glob(str(Path(dd) / "**/*.npz"), recursive=True))]
    if not files:
        raise FileNotFoundError(f"no .npz under {dirs}")
    segments, meta = [], {}
    for base, f in files:
        d = np.load(f, allow_pickle=True)
        if not all(k in d for k in ("hand_joint_pos", "cmd_q22", "cube_state")):
            continue                                                     # e.g. traj_*.npz sidecars
        if not meta:
            meta = {
                "hand_joint_names": d["hand_joint_names"].tolist() if "hand_joint_names" in d else None,
                "cmd_q22_names": d["cmd_q22_names"].tolist() if "cmd_q22_names" in d else None,
                "cube_dims": CUBE_POSE, "obj_dim": CUBE_POSE, "rot_repr": "quat",
                "hz": float(d["hz"]) if "hz" in d else None,
            }
        hand = np.asarray(d["hand_joint_pos"], np.float32)
        cube = np.asarray(d["cube_state"], np.float32)
        act = np.asarray(d["cmd_q22"], np.float32)
        states = np.concatenate([hand, cube[:, :CUBE_POSE]], axis=1)    # (T, 29)
        T = len(states)
        if T < 2:
            continue
        act = act[:T]
        finite = np.isfinite(states).all(axis=1)                          # (T,)
        # boundary before frame i (1..T-1) if a reset happened between i-1 and i
        dpos = np.linalg.norm(np.diff(cube[:, :3], axis=0), axis=1)       # (T-1,)
        q = cube[:, 3:7]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        dang = _quat_geo_deg(q)                                           # (T-1,)
        brk = (dpos > pos_jump) | (dang > ang_jump)                       # NaN compares False: ok,
        brk |= finite[1:] != finite[:-1]                                  # toggles cut here anyway
        if "episode" in d:
            brk |= (np.diff(np.asarray(d["episode"]).astype(np.int64)[:T]) != 0)
        if "t" in d:
            dt = np.diff(np.asarray(d["t"], np.float64)[:T])
            brk |= (dt < 0)                                               # per-episode t reset
            pos_dt = dt[dt > 0]
            if pos_dt.size:                                               # trimmed-reset t gap
                brk |= (dt > gap_factor * float(np.median(pos_dt))) & (dt > min_gap_s)
        starts = sorted(set([0] + [i + 1 for i in np.where(brk)[0]] + [T]))
        name = str(Path(f).relative_to(base))
        for si, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            if b - a >= min_len and finite[a]:  # spans are uniformly valid; drop the NaN ones
                segments.append({"states": states[a:b], "actions": act[a:b],
                                 "file": name, "seg": si})
    if not segments:
        raise RuntimeError(f"no usable segments under {data_dir}")
    return segments, meta


# backwards-compatible name used by earlier versions of the training scripts
load_worldmodel_segments = load_segments


def split_segments(
    segments: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Hold out a fraction of whole SEGMENTS for validation (so adjacent windows don't leak)."""
    n = len(segments)
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, round(val_frac * n))
    val_ids = set(order[:n_val].tolist())
    train = [segments[i] for i in range(n) if i not in val_ids]
    val = [segments[i] for i in range(n) if i in val_ids]
    return train, val


def fit_normalizers(segments: list[dict]) -> tuple[Normalizer, Normalizer]:
    """Fit state/action normalizers on the pooled frames of ``segments`` (use the TRAIN split)."""
    s = np.concatenate([seg["states"] for seg in segments], axis=0)
    a = np.concatenate([seg["actions"] for seg in segments], axis=0)
    return Normalizer.fit(s), Normalizer.fit(a)


# ---------------------------------------------------------------------------
# Conditioning history (world model)
# ---------------------------------------------------------------------------

def history_offsets(n_hist: int, hist_stride: int) -> np.ndarray:
    """Frame offsets (>=0, current-first) of the conditioning state stack.

    ``n_hist=3, hist_stride=3`` -> ``[0, 3, 6]``, i.e. cond = [s_t, s_{t-3}, s_{t-6}]."""
    n_hist = max(1, int(n_hist))
    stride = max(1, int(hist_stride))
    return (np.arange(n_hist, dtype=np.int64) * stride)


def stack_history(frames, n_hist: int, hist_stride: int) -> np.ndarray:
    """Build the flattened conditioning vector from a per-frame state buffer.

    ``frames`` is (T, state_dim) ordered oldest -> newest (``frames[-1]`` = current). Offsets
    that run off the start of the buffer clamp to the oldest available frame (episode warm-up).
    Returns (n_hist * state_dim,), current-first -- matching ``WorldModelWindowDataset``."""
    f = np.asarray(frames)
    idx = np.clip(len(f) - 1 - history_offsets(n_hist, hist_stride), 0, len(f) - 1)
    return f[idx].reshape(-1)


# ---------------------------------------------------------------------------
# Window datasets
# ---------------------------------------------------------------------------

class WorldModelWindowDataset:
    """Pools ``(cond_t, actions[t:t+H], cube_pose[t:t+H])`` windows for world-model training.

    The prediction target is the cube-pose slice (``[22:29]`` = pos3 + quat4) of the NORMALIZED
    future state, so it lives in the state normalizer's space and target[k=0] equals the cube
    pose of the current state.

    ``n_hist``/``hist_stride`` widen the conditioning from the single current state to a STACK
    of ``n_hist`` frames spaced ``hist_stride`` apart, current-first:
    ``cond = [s_t, s_{t-stride}, ..., s_{t-(n_hist-1)*stride}]`` (``cond_dim = n_hist*state_dim``).
    Window starts are pushed ``(n_hist-1)*stride`` frames into their segment so neither the
    history nor the horizon ever crosses a reset.
    """

    def __init__(self, segments, horizon, norm_s, norm_a, n_hist=1, hist_stride=1):
        import torch
        self.torch = torch
        self.horizon = int(horizon)
        self.n_hist = max(1, int(n_hist))
        self.hist_stride = max(1, int(hist_stride))
        self._hist_off = history_offsets(self.n_hist, self.hist_stride)      # (n_hist,) current-first
        pad = int(self._hist_off.max())                    # frames of history needed before a start
        self.tgt_cols = np.arange(HAND_DIM, HAND_DIM + CUBE_POSE, dtype=np.int64)
        flat_s, flat_a, starts, off = [], [], [], 0
        self.n_skipped = 0
        for seg in segments:
            sN = norm_s.normalize(seg["states"]).astype(np.float32)
            aN = norm_a.normalize(seg["actions"]).astype(np.float32)
            L = sN.shape[0]
            flat_s.append(sN); flat_a.append(aN)
            n_valid = L - self.horizon + 1 - pad
            if n_valid > 0:
                starts.append(off + pad + np.arange(n_valid, dtype=np.int64))
            else:
                self.n_skipped += 1
            off += L
        self.state_dim = segments[0]["states"].shape[1] if segments else 0
        self.action_dim = segments[0]["actions"].shape[1] if segments else 0
        self.cond_dim = self.state_dim * self.n_hist       # width of the global conditioning
        self.obj_dim = int(self.tgt_cols.shape[0])         # predicted width = cube pose (7)
        self.flat_s = np.concatenate(flat_s, 0) if flat_s else np.zeros((0, self.state_dim), np.float32)
        self.flat_a = np.concatenate(flat_a, 0) if flat_a else np.zeros((0, self.action_dim), np.float32)
        self.starts = np.concatenate(starts) if starts else np.empty(0, np.int64)
        self._h_off = np.arange(self.horizon, dtype=np.int64)

    def __len__(self):
        return int(self.starts.shape[0])

    def sample_batch(self, batch_size, rng):
        """Returns ``(cond (B, cond_dim), act (B, H, 22), target (B, H, 7))`` normalized
        torch tensors; ``cond`` is the current-first history stack."""
        st = self.starts[rng.integers(0, len(self), size=batch_size)]
        hidx = st[:, None] - self._hist_off[None, :]                      # (B, n_hist) current-first
        s0 = self.flat_s[hidx].reshape(len(st), -1)                       # (B, n_hist*state_dim)
        idx = st[:, None] + self._h_off[None, :]                          # (B, H)
        act = self.flat_a[idx]                                            # (B, H, act_dim)
        obj = self.flat_s[idx][:, :, self.tgt_cols]                       # (B, H, 7)
        return (self.torch.from_numpy(s0), self.torch.from_numpy(act), self.torch.from_numpy(obj))


class WindowDataset:
    """Pools valid ``(state_t -> actions[t:t+H])`` windows across segments and samples uniformly.

    Windows never cross a segment boundary. States/actions are pre-normalized at construction,
    so ``sample_batch`` is just a gather. Returns float32 torch tensors.
    """

    def __init__(self, segments: list[dict], horizon: int, norm_s: Normalizer, norm_a: Normalizer):
        import torch  # local import so data.py is importable without torch for inspection
        self.torch = torch
        self.horizon = int(horizon)
        flat_s, flat_a, starts = [], [], []
        off = 0
        self.n_skipped = 0
        for seg in segments:
            sN = norm_s.normalize(seg["states"]).astype(np.float32)
            aN = norm_a.normalize(seg["actions"]).astype(np.float32)
            L = sN.shape[0]
            flat_s.append(sN)
            flat_a.append(aN)
            n_valid = L - self.horizon + 1
            if n_valid > 0:
                starts.append(off + np.arange(n_valid, dtype=np.int64))
            else:
                self.n_skipped += 1
            off += L
        self.state_dim = segments[0]["states"].shape[1] if segments else 0
        self.action_dim = segments[0]["actions"].shape[1] if segments else 0
        self.flat_s = np.concatenate(flat_s, axis=0) if flat_s else np.zeros((0, self.state_dim), np.float32)
        self.flat_a = np.concatenate(flat_a, axis=0) if flat_a else np.zeros((0, self.action_dim), np.float32)
        self.starts = np.concatenate(starts) if starts else np.empty(0, np.int64)
        self._h_off = np.arange(self.horizon, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        """Returns ``(cond (B, state_dim), chunk (B, H, action_dim))`` normalized torch tensors."""
        picks = rng.integers(0, len(self), size=batch_size)
        s0 = self.flat_s[self.starts[picks]]                                  # (B, state_dim)
        idx = self.starts[picks][:, None] + self._h_off[None, :]              # (B, H)
        chunk = self.flat_a[idx]                                             # (B, H, action_dim)
        return self.torch.from_numpy(s0), self.torch.from_numpy(chunk)
