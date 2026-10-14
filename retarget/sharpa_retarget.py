"""Manus glove -> Sharpa hand retargeting core (viser-free, reusable).

Shared by the viser viewer (retarget_viz.py) and the Isaac Lab teleop
(isaac/teleop_isaac.py). Contains: skeleton normalization, the dex-retargeting
"shape" optimizer wrapper, per-finger scaling, and distal-joint coupling.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from manus_client import GloveData, RawNode  # vendored ZMQ client

_ROOT = Path(__file__).resolve().parent.parent      # repo root (this file lives in retarget/)
_SHARPA_URDF = _ROOT / "assets" / "urdf" / "sharpa" / "right_fixed.urdf"
_ASSETS = _ROOT / "assets"

# ── Sharpa joint order (SDK / URDF) ───────────────────────────────────────────
_URDF_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]


def get_urdf_joint_names(side="right"):
    return [f"{side}_{s}" for s in _URDF_JOINT_SUFFIXES]


# ── Manus skeleton -> normalized robot-frame keypoints ────────────────────────
OPERATOR2ROBOT = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Uniform abduction-lean fix: one rotation of the normalized hand about the palm
# normal (robot x-axis) de-skews every finger's sideways pointing at once.
# Robot frame: x=palm normal, y=lateral (radial-ulnar), z=fingers forward.
_AB_FIX = {"deg": 0.0}


def _rot_about_palm_normal(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def set_abduction_fix(deg):
    """Set the uniform abduction-lean correction (rotation about palm normal, deg)."""
    _AB_FIX["deg"] = float(deg)


def common_mode_lean_deg(skel):
    """Mean sideways angle (deg) of the 4 long fingers' axes -- the common lean to
    cancel (a symmetric natural spread averages ~0). Pass to set_abduction_fix()."""
    if skel is None or len(skel) < 25:
        return 0.0
    angs = [np.degrees(np.arctan2((skel[tip] - skel[mcp])[1],
                                  (skel[tip] - skel[mcp])[2]))
            for mcp, tip in ((5, 9), (10, 14), (15, 19), (20, 24))]
    return float(np.mean(angs))


def _extract_hand_positions(data: GloveData):
    wrist, joints = None, {}
    for node in data.raw_nodes:
        chain = str(node.chain_type).split(".")[-1]
        joint = str(node.joint_type).split(".")[-1]
        if chain == "Hand" and wrist is None:
            wrist = node.position
        elif chain in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            joints[(chain, joint)] = node.position
    if wrist is None:
        return None
    out = np.zeros((25, 3), dtype=np.float64)
    out[0] = wrist
    layout = [("Thumb", 1, ["MCP", "PIP", "DIP", "TIP"]),
              ("Index", 5, ["MCP", "PIP", "IP", "DIP", "TIP"]),
              ("Middle", 10, ["MCP", "PIP", "IP", "DIP", "TIP"]),
              ("Ring", 15, ["MCP", "PIP", "IP", "DIP", "TIP"]),
              ("Pinky", 20, ["MCP", "PIP", "IP", "DIP", "TIP"])]
    for finger, base, jtypes in layout:
        for j, jt in enumerate(jtypes):
            key = (finger, jt)
            if key in joints:
                out[base + j] = joints[key]
            elif jt == "IP":
                continue
            else:
                return None
    return out


def _estimate_wrist_frame(wrist, index_mcp, middle_mcp):
    pts = np.array([wrist, index_mcp, middle_mcp])
    pts = pts - pts.mean(axis=0, keepdims=True)
    x_vec = wrist - middle_mcp
    _, _, v = np.linalg.svd(pts)
    normal = v[2, :]
    x = x_vec - np.dot(x_vec, normal) * normal
    if np.linalg.norm(x) < 1e-8:
        return np.eye(3)
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.dot(z, index_mcp - middle_mcp) < 0:
        normal, z = -normal, -z
    return np.stack([x, normal, z], axis=1)


def normalize_hand_positions(data: GloveData, side="right"):
    hp = _extract_hand_positions(data)
    if hp is None:
        return None
    wrist = hp[0]
    frame = _estimate_wrist_frame(wrist, hp[5], hp[10])
    out = (hp - wrist[None, :]) @ frame @ OPERATOR2ROBOT
    if side == "left":
        out[:, 0] *= -1
        out[:, 1] *= -1
    if _AB_FIX["deg"]:
        out = out @ _rot_about_palm_normal(_AB_FIX["deg"]).T
    return out


def compute_retarget_input(data: GloveData, indices, side="right"):
    jp = normalize_hand_positions(data, side=side)
    if jp is None:
        return None
    return jp[indices[1], :] - jp[indices[0], :]


# ── Distal-joint coupling ─────────────────────────────────────────────────────
def build_distal_coupling(joint_names):
    idx = {n: i for i, n in enumerate(joint_names)}
    side = joint_names[0].split("_")[0]
    pairs = []
    for f in ("index", "middle", "ring", "pinky"):
        dip, pip = f"{side}_{f}_DIP", f"{side}_{f}_PIP"
        if dip in idx and pip in idx:
            pairs.append((idx[dip], idx[pip], 1.3963))
    ip, src = f"{side}_thumb_IP", f"{side}_thumb_MCP_FE"
    if ip in idx and src in idx:
        pairs.append((idx[ip], idx[src], 1.7453))
    return pairs


def apply_distal_coupling(qpos, pairs, ratio):
    if ratio <= 0.0:
        return qpos
    for dst, src, upper in pairs:
        qpos[dst] = float(np.clip(ratio * qpos[src], 0.0, upper))
    return qpos


# ── Direct thumb tracking from Manus ergonomics (thumb_mode "joint") ──────────
# Drive the Sharpa thumb joints straight from the glove's reported thumb joint
# angles instead of the fingertip-pose optimizer. Each ergonomics value's observed
# range auto-calibrates to the URDF joint limits (units-agnostic; sweep the thumb
# once). Inverts come from the shared params. Mirror of the copy in retarget_viz.py.
#   (ergo suffix, sharpa joint suffix, is_spread, invert)
_THUMB_ERGO_MAP = [
    ("ThumbMCPSpread", "thumb_CMC_AA", True, False),
    ("ThumbMCPStretch", "thumb_CMC_FE", False, False),
    ("ThumbPIPStretch", "thumb_MCP_FE", False, False),
    ("ThumbDIPStretch", "thumb_IP", False, False),
]


def _read_joint_limits(urdf_path):
    from xml.etree import ElementTree as ET
    lim = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        lo = joint.find("limit")
        if lo is not None and lo.get("lower") is not None:
            lim[joint.get("name")] = (float(lo.get("lower")), float(lo.get("upper")))
    return lim


def _find_ergo(ergo, suffix):
    s = suffix.lower()
    for k, v in ergo.items():
        if k.lower().endswith(s):
            return float(v)
    return None


class ThumbErgoTracker:
    """Map the 4 Manus thumb ergonomics to Sharpa thumb joints (auto-ranged + EMA)."""

    def __init__(self, joint_names, urdf_path, ema_alpha=0.3):
        idx = {n: i for i, n in enumerate(joint_names)}
        side = joint_names[0].split("_")[0]
        limits = _read_joint_limits(urdf_path)
        self.entries = []   # [suffix, jidx, lo, hi, is_spread, invert, vmin, vmax]
        self.invert_targets = {}
        for suf, jsuf, is_spread, invert in _THUMB_ERGO_MAP:
            jn = f"{side}_{jsuf}"
            if jn in idx and jn in limits:
                lo, hi = limits[jn]
                entry = [suf, idx[jn], lo, hi, is_spread, invert, None, None]
                self.entries.append(entry)
                self.invert_targets[jsuf.replace("thumb_", "")] = entry
        self.mcp_aa_idx = idx.get(f"{side}_thumb_MCP_AA")
        self.ema_alpha = ema_alpha
        self._prev = {}
        self._announced = False

    def angles(self, glove_data):
        ergo = glove_data.get_ergonomics_dict() if glove_data is not None else {}
        if not ergo:
            return None
        out, found = {}, []
        for e in self.entries:
            suf, jidx, lo, hi, is_spread, invert = e[0], e[1], e[2], e[3], e[4], e[5]
            v = _find_ergo(ergo, suf)
            if v is None:
                continue
            found.append(suf)
            e[6] = v if e[6] is None else min(e[6], v)
            e[7] = v if e[7] is None else max(e[7], v)
            rng = e[7] - e[6]
            t = (0.5 if is_spread else 0.0) if rng < 1e-6 else (v - e[6]) / rng
            if invert:
                t = 1.0 - t
            ang = (lo + t * (hi - lo)) if is_spread else (max(lo, 0.0)
                   + t * (hi - max(lo, 0.0)))
            out[jidx] = float(np.clip(ang, lo, hi))
        if self.mcp_aa_idx is not None:
            out[self.mcp_aa_idx] = 0.0
        a = float(np.clip(self.ema_alpha, 1e-3, 1.0))
        for jidx, val in out.items():
            sm = a * val + (1.0 - a) * self._prev.get(jidx, val)
            self._prev[jidx] = sm
            out[jidx] = sm
        if not self._announced:
            self._announced = True
            print(f"[thumb] joint mode tracking ergonomics {found}" if found else
                  "[thumb] joint mode: NO thumb ergonomics in stream "
                  f"(sample keys {list(ergo)[:3]}); keeping optimizer thumb")
        return out if found else None


def apply_thumb_joint(qpos, glove_data, tracker, weight=1.0):
    """Blend the optimizer's thumb pose (already in qpos) with the glove's measured
    thumb joint angles. weight=1 -> pure direct joint (overwrite, the old behavior);
    0 -> pure fingertip-pose optimizer; in between -> hybrid."""
    if tracker is None:
        return qpos
    ang = tracker.angles(glove_data)
    if ang:
        w = float(np.clip(weight, 0.0, 1.0))
        for jidx, a in ang.items():
            qpos[jidx] = (1.0 - w) * qpos[jidx] + w * a
    return qpos


def make_thumb_tracker(joint_names, urdf_path, params):
    """Build a ThumbErgoTracker configured from shared params (ema + inverts)."""
    tr = ThumbErgoTracker(joint_names, urdf_path,
                          ema_alpha=params.get("thumb_ema", 0.3))
    for short, inv in (params.get("thumb_invert") or {}).items():
        if short in tr.invert_targets:
            tr.invert_targets[short][5] = bool(inv)
    return tr


# ── Per-finger size scaling ───────────────────────────────────────────────────
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def compute_task_scales(master, finger_scales, task_fingers, base_scale):
    return base_scale * master * np.array(
        [finger_scales[f] for f in task_fingers], dtype=np.float64)


# ── Shape (multi-keypoint) retargeter ─────────────────────────────────────────
_SHAPE_HJ = {"index": (5, 6, 8, 9), "middle": (10, 11, 13, 14),
             "ring": (15, 16, 18, 19), "pinky": (20, 21, 23, 24)}


class ShapeRetargeter:
    """Per finger: fingertip + proximal/distal bone-direction vectors."""

    def __init__(self, side="right"):
        import tempfile
        import yaml
        import yourdfpy
        from dex_retargeting.retargeting_config import RetargetingConfig
        if side != "right":
            raise ValueError("shape strategy is wired for the right hand only")

        tasks = []  # (origin_link, task_link, h_origin, h_task, finger)
        for f, (mcp, pip, dip, tip) in _SHAPE_HJ.items():
            PP, MP, DP = f"right_{f}_PP", f"right_{f}_MP", f"right_{f}_DP"
            tasks.append(("right_hand_C_MC", DP, 0, tip, f))
            tasks.append((PP, MP, mcp, pip, f))
            tasks.append((MP, DP, pip, tip, f))
        tasks.append(("right_hand_C_MC", "right_thumb_DP", 0, 4, "thumb"))
        tasks.append(("right_thumb_PP", "right_thumb_DP", 2, 4, "thumb"))
        self.tasks = tasks
        self.task_fingers = [t[4] for t in tasks]
        h_org, h_task = [t[2] for t in tasks], [t[3] for t in tasks]

        cfg = {"retargeting": {
            "type": "vector", "urdf_path": "urdf/sharpa/right_fixed.urdf",
            "target_origin_link_names": [t[0] for t in tasks],
            "target_task_link_names": [t[1] for t in tasks],
            "target_link_human_indices": [h_org, h_task],
            "scaling_factor": 1.0, "low_pass_alpha": 1.0}}
        fd, p = tempfile.mkstemp(suffix=".yml")
        os.close(fd)
        with open(p, "w") as fh:
            yaml.dump(cfg, fh)
        try:
            RetargetingConfig.set_default_urdf_dir(str(_ASSETS))
            self._retargeting = RetargetingConfig.load_from_file(p).build()
        finally:
            os.unlink(p)

        self._side = side
        self._indices = np.array([h_org, h_task])
        std = get_urdf_joint_names(side)
        self._reorder = np.array(
            [self._retargeting.joint_names.index(n) for n in std])
        self._last_qpos = np.zeros(len(std), dtype=np.float32)

        robot = yourdfpy.URDF.load(str(_SHARPA_URDF))
        robot.update_cfg(np.zeros(len(std)))

        def rp(link):
            return robot.get_transform(link)[:3, 3]

        self._robot_ext = np.array(
            [np.linalg.norm(rp(t[1]) - rp(t[0])) for t in tasks])
        self._human_max = self._robot_ext / 0.9
        self.base_scale = self._robot_ext / self._human_max

    def update_calibration(self, skel_pts):
        if skel_pts is None or len(skel_pts) < 25:
            return
        o, t = self._indices[0], self._indices[1]
        d = np.linalg.norm(skel_pts[t] - skel_pts[o], axis=1)
        self._human_max = np.maximum(self._human_max, d)
        self.base_scale = self._robot_ext / np.maximum(self._human_max, 1e-6)

    def retarget(self, glove_data, task_scales=None):
        ref = compute_retarget_input(glove_data, self._indices, self._side)
        if ref is None:
            return self._last_qpos.copy()
        scales = self.base_scale if task_scales is None else task_scales
        q = self._retargeting.retarget(
            np.asarray(ref, dtype=np.float64) * scales[:, None])
        q = q[self._reorder].astype(np.float32)
        self._last_qpos = q
        return q


# ── Replay (no hardware): rebuild GloveData from recorded manus_node_pos ───────
_REPLAY_LAYOUT = [("Hand", "Wrist")]
for _chain, _joints in [("Thumb", ["MCP", "PIP", "DIP", "TIP"]),
                        ("Index", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Middle", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Ring", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Pinky", ["MCP", "PIP", "IP", "DIP", "TIP"])]:
    _REPLAY_LAYOUT += [(_chain, _j) for _j in _joints]


def glove_data_from_nodes(node_pos):
    nodes = [RawNode(node_id=i, parent_node_id=max(i - 1, -1), joint_type=j,
                     chain_type=c, position=np.asarray(node_pos[i], dtype=np.float64),
                     orientation=np.array([0.0, 0.0, 0.0, 1.0]))
             for i, (c, j) in enumerate(_REPLAY_LAYOUT)]
    return GloveData(glove_id=0, side="Right", timestamp=0.0, raw_nodes=nodes)


def make_scale_fn(retargeter, master=1.0, finger_scales=None):
    """Return scale(skel_pts) -> per-task scales (with running size calibration)."""
    fs = finger_scales or {f: (1.1 if f == "pinky" else 1.0) for f in FINGER_ORDER}

    def scale(skel_pts):
        retargeter.update_calibration(skel_pts)
        return compute_task_scales(master, fs, retargeter.task_fingers,
                                   retargeter.base_scale)
    return scale
