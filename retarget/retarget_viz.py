#!/usr/bin/env python3
"""Live Manus glove -> Sharpa hand retargeting visualizer (shape strategy).

Self-contained: launches a Viser server showing the 25-point Manus hand skeleton
(green) and the retargeted Sharpa hand (URDF), with the Sharpa fingers following
the glove. The wrist root is FIXED (no Vive tracker) -- this visualizes finger
retargeting only.

Strategy = "shape" (DexPilot-style multi-keypoint vector retargeting): per finger
it matches the fingertip position AND the proximal/distal bone directions, so the
curl distribution (and fingertip pointing) tracks the hand. Per-finger size is
auto-calibrated from the running max bone length. DIP/thumb-IP are coupled to
their proximal joint (the distal bone is too short for the optimizer to actuate).

Pipeline:
    ManusGlove (ZMQ) -> normalized skeleton -> per-task target vectors
      -> dex-retargeting vector optimizer -> 22 Sharpa joints -> ViserUrdf

Usage
-----
Live (needs the Manus glove + manus_zmq_server running):
    python retarget_viz.py

Replay a recording (no hardware; relabels manus_node_pos through the optimizer):
    python retarget_viz.py --source replay --npz <recording with manus_node_pos>.npz

Open the printed http://0.0.0.0:8080 URL.
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

import retarget_params as rp                          # shared YAML param store
from manus_client import GloveData, ManusGlove, RawNode  # vendored ZMQ client

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
# Maps wrist-frame operator coords to the Sharpa URDF palm convention.
OPERATOR2ROBOT = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Uniform abduction-lean fix: one rotation of the normalized hand about the palm
# normal (robot x-axis) de-skews every finger's sideways pointing at once -- the
# whole-hand, geometrically-consistent version of "nudge all the abductions".
# In the robot frame: x=palm normal, y=lateral (radial-ulnar), z=fingers forward.
_AB_FIX = {"deg": 0.0}


def _rot_about_palm_normal(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _common_mode_lean_deg(skel):
    """Mean sideways angle (deg) of the 4 long fingers' axes. A symmetric natural
    spread averages ~0, so a nonzero mean is the common lean to cancel."""
    if skel is None or len(skel) < 25:
        return 0.0
    angs = [np.degrees(np.arctan2((skel[tip] - skel[mcp])[1],
                                  (skel[tip] - skel[mcp])[2]))
            for mcp, tip in ((5, 9), (10, 14), (15, 19), (20, 24))]
    return float(np.mean(angs))


def _extract_hand_positions(data: GloveData):
    """(25,3) array: wrist + thumb(MCP,PIP,DIP,TIP) + 4 fingers(MCP,PIP,IP,DIP,TIP)."""
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
    """(25,3) keypoints centered at the wrist, in the Sharpa palm convention."""
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
    """Vector-retargeting ref: human (task - origin) vectors per task."""
    jp = normalize_hand_positions(data, side=side)
    if jp is None:
        return None
    return jp[indices[1], :] - jp[indices[0], :]


# ── Manus skeleton viser overlay ──────────────────────────────────────────────
class ManusSkeleton:
    BONE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4),
                  (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                  (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                  (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
                  (0, 20), (20, 21), (21, 22), (22, 23), (23, 24)]

    def __init__(self, server, prefix, color=(80, 220, 80)):
        self._kp = [server.scene.add_icosphere(f"/{prefix}/kp_{i}", radius=0.004,
                    color=color, visible=False) for i in range(25)]
        self._ln = [server.scene.add_line_segments(
            f"/{prefix}/bone_{i}", points=np.zeros((1, 2, 3)),
            colors=np.array([[list(color), list(color)]]), line_width=2.0,
            visible=False) for i in range(len(self.BONE_PAIRS))]

    def update(self, pos):
        n = len(pos)
        for i, h in enumerate(self._kp):
            if i < n:
                h.position = pos[i].astype(np.float32); h.visible = True
            else:
                h.visible = False
        for i, (a, b) in enumerate(self.BONE_PAIRS):
            if a < n and b < n:
                self._ln[i].points = np.array([[pos[a], pos[b]]], dtype=np.float32)
                self._ln[i].visible = True
            else:
                self._ln[i].visible = False

    def hide(self):
        for h in self._kp:
            h.visible = False
        for h in self._ln:
            h.visible = False


def patch_urdf_remove_base_cylinder(urdf_path: Path) -> Path:
    """Temp URDF with the base_link cylinder replaced by a tiny sphere.

    Written to the system temp dir (mesh paths made absolute so they still resolve) and
    removed at exit, so nothing is left behind in the asset folder."""
    import tempfile
    from xml.etree import ElementTree as ET
    ET.register_namespace("", "")
    tree = ET.parse(urdf_path)
    for link in tree.getroot().findall("link"):
        if link.get("name") == "base_link":
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is not None and geom.find("cylinder") is not None:
                    geom.remove(geom.find("cylinder"))
                    ET.SubElement(geom, "sphere", radius="0.001")
    for mesh in tree.getroot().iter("mesh"):
        fn = mesh.get("filename", "")
        if fn and "://" not in fn and not os.path.isabs(fn):
            mesh.set("filename", str((urdf_path.parent / fn).resolve()))
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="wb",
                                      prefix="sharpa_patched_")
    tree.write(tmp.name)
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))
    return Path(tmp.name)


# ── Distal-joint coupling ─────────────────────────────────────────────────────
# The vector objective leaves the distal joint (DIP / thumb IP) effectively
# unactuated (the distal bone is too short to register), so couple it to the
# proximal joint like a tendon-driven finger. (dst_idx, src_idx, upper_rad)
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


# ── Direct thumb tracking from Manus ergonomics (thumb-mode "joint") ──────────
# Instead of matching the thumb fingertip pose through the optimizer, drive the
# Sharpa thumb joints straight from the glove's reported thumb joint angles. Each
# ergonomics value's observed range is auto-calibrated to the URDF joint limits,
# so the (undocumented) units don't matter -- after launch, sweep the thumb once
# through its full range. Mapping:
#   ThumbMCPSpread  -> thumb_CMC_AA   (spread, neutral-centered)
#   ThumbMCPStretch -> thumb_CMC_FE   (flexion)
#   ThumbPIPStretch -> thumb_MCP_FE   (flexion)
#   ThumbDIPStretch -> thumb_IP       (flexion)
#   thumb_MCP_AA    -> 0
# (ergo suffix, sharpa joint suffix, is_spread, invert)
# `invert` flips a joint's direction; each is also a live checkbox in the viser
# panel ("Invert thumb <joint>") so you can fix whichever axis reads backwards.
_THUMB_ERGO_MAP = [
    ("ThumbMCPSpread", "thumb_CMC_AA", True, False),
    ("ThumbMCPStretch", "thumb_CMC_FE", False, True),
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
    """Lookup an ergonomics value by name suffix (robust to the SDK's prefixing)."""
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
        self.invert_targets = {}   # short joint name -> entry (for live toggles)
        for suf, jsuf, is_spread, invert in _THUMB_ERGO_MAP:
            jn = f"{side}_{jsuf}"
            if jn in idx and jn in limits:
                lo, hi = limits[jn]
                entry = [suf, idx[jn], lo, hi, is_spread, invert, None, None]
                self.entries.append(entry)
                self.invert_targets[jsuf.replace("thumb_", "")] = entry
        self.mcp_aa_idx = idx.get(f"{side}_thumb_MCP_AA")
        self.ema_alpha = ema_alpha      # 1.0 = no smoothing; lower = smoother
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
        # exponential moving average per joint (alpha=1 -> passthrough)
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


# ── Target overlay (what the optimizer tracks) ────────────────────────────────
class TargetMarkers:
    def __init__(self, server, colors):
        colors = np.asarray(colors)
        self.n = len(colors)
        self._sph = [server.scene.add_icosphere(
            f"/targets/t{i}", radius=0.006,
            color=tuple(int(c) for c in colors[i]), visible=False)
            for i in range(self.n)]
        self._lines = server.scene.add_line_segments(
            "/targets/lines", points=np.zeros((self.n, 2, 3)),
            colors=np.stack([colors, colors], axis=1), line_width=2.5, visible=False)

    def update(self, origins, targets):
        for i in range(self.n):
            self._sph[i].position = targets[i].astype(np.float32)
            self._sph[i].visible = True
        self._lines.points = np.stack([origins, targets], axis=1).astype(np.float32)
        self._lines.visible = True

    def hide(self):
        for h in self._sph:
            h.visible = False
        self._lines.visible = False


def update_targets(targets, retargeter, skel_pts, task_scales, offset):
    if targets is None or skel_pts is None or task_scales is None:
        if targets is not None:
            targets.hide()
        return
    o, t = retargeter._indices[0], retargeter._indices[1]
    origins = skel_pts[o] + offset
    goals = skel_pts[o] + task_scales[:, None] * (skel_pts[t] - skel_pts[o]) + offset
    targets.update(origins, goals)


# ── Scene ─────────────────────────────────────────────────────────────────────
def build_scene(server):
    if not _SHARPA_URDF.exists():
        raise FileNotFoundError(f"Sharpa URDF not found: {_SHARPA_URDF}")
    patched = patch_urdf_remove_base_cylinder(_SHARPA_URDF)
    sharpa_vis = ViserUrdf(server, patched, root_node_name="/sharpa")
    server.scene.add_frame("/sharpa_root", axes_length=0.05, axes_radius=0.003)
    server.scene.add_label("/sharpa_label", text="Retargeted Sharpa",
                           position=(0.0, 0.0, 0.12))
    skel = ManusSkeleton(server, prefix="manus", color=(80, 220, 80))
    server.scene.add_label("/manus_label", text="Manus glove",
                           position=(0.0, 0.0, 0.16))
    server.scene.add_grid("/floor", width=0.6, height=0.6, cell_size=0.05)
    return sharpa_vis, skel


def update_hand(sharpa_vis, skel, joint_names, qpos, skel_pts, offset):
    sharpa_vis.update_cfg({n: float(qpos[j]) for j, n in enumerate(joint_names)})
    if skel_pts is not None:
        skel.update((skel_pts + offset).astype(np.float32))
    else:
        skel.hide()


# ── Replay support (no hardware) ──────────────────────────────────────────────
_REPLAY_LAYOUT = [("Hand", "Wrist")]
for _chain, _joints in [("Thumb", ["MCP", "PIP", "DIP", "TIP"]),
                        ("Index", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Middle", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Ring", ["MCP", "PIP", "IP", "DIP", "TIP"]),
                        ("Pinky", ["MCP", "PIP", "IP", "DIP", "TIP"])]:
    _REPLAY_LAYOUT += [(_chain, _j) for _j in _joints]


def _glove_data_from_nodes(node_pos):
    nodes = [RawNode(node_id=i, parent_node_id=max(i - 1, -1), joint_type=j,
                     chain_type=c, position=np.asarray(node_pos[i], dtype=np.float64),
                     orientation=np.array([0.0, 0.0, 0.0, 1.0]))
             for i, (c, j) in enumerate(_REPLAY_LAYOUT)]
    return GloveData(glove_id=0, side="Right", timestamp=0.0, raw_nodes=nodes)


# ── Ctrl-S (terminal) -> save params ──────────────────────────────────────────
class _CtrlSPoller:
    """Detect Ctrl-S from the launching terminal (XON/XOFF flow control disabled
    so 0x13 reaches us; ISIG kept so Ctrl-C still quits)."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.fd = self.old = None
        if self.enabled:
            import termios
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            new = termios.tcgetattr(self.fd)
            new[3] &= ~(termios.ICANON | termios.ECHO)
            new[0] &= ~termios.IXON
            termios.tcsetattr(self.fd, termios.TCSANOW, new)

    def hit_ctrl_s(self):
        if not self.enabled:
            return False
        import select
        got = False
        while select.select([sys.stdin], [], [], 0)[0]:
            got = sys.stdin.read(1) == "\x13" or got
        return got

    def restore(self):
        if self.enabled and self.old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ── Run loops ─────────────────────────────────────────────────────────────────
def run_live(args, retargeter, joint_names, scene, scale, coupling, offset,
             thumb_tracker=None, latest=None, on_save=None, keys=None):
    sharpa_vis, skel, targets = scene
    print(f"[live] connecting to Manus server {args.manus_host}:{args.manus_port} ...")
    dt = 1.0 / args.hz
    n, last = 0, time.monotonic()
    with ManusGlove(host=args.manus_host, pub_port=args.manus_port,
                    sides=[args.side]) as glove:
        while True:
            t0 = time.monotonic()
            if keys is not None and keys.hit_ctrl_s() and on_save:
                on_save()
            data = glove.get_data(args.side)
            if data is not None and data.raw_nodes:
                skel_pts = normalize_hand_positions(data, side=args.side)
                if latest is not None:
                    latest["skel"] = skel_pts
                ts = scale(skel_pts)
                qpos = apply_distal_coupling(
                    retargeter.retarget(data, ts), coupling, args.dip_coupling)
                if args.thumb_mode == "joint":
                    qpos = apply_thumb_joint(qpos, data, thumb_tracker,
                                             args.thumb_joint_weight)
                update_hand(sharpa_vis, skel, joint_names, qpos, skel_pts, offset)
                update_targets(targets, retargeter, skel_pts, ts, offset)
                n += 1
            if time.monotonic() - last >= 5.0:
                print(f"[live] {n / (time.monotonic() - last):.1f} Hz"
                      + ("" if n else "  (no glove data yet)"))
                n, last = 0, time.monotonic()
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)


def run_replay(args, retargeter, joint_names, scene, scale, coupling, offset, server,
               thumb_tracker=None, latest=None, on_save=None, keys=None):
    sharpa_vis, skel, targets = scene
    path = Path(args.npz)
    if not path.exists():
        raise FileNotFoundError(f"--npz not found: {path}")
    node_pos = np.load(path, allow_pickle=True)["manus_node_pos"]
    T = len(node_pos)
    n_nodes = np.asarray(node_pos[0]).shape[0] if T else 0
    ok = n_nodes == len(_REPLAY_LAYOUT)
    print(f"[replay] {path.name}: {T} frames, {n_nodes} nodes/frame"
          + ("" if ok else "  (WARNING: unexpected node count)"))
    play = server.gui.add_checkbox("Play", initial_value=True)
    frame = server.gui.add_slider("Frame", min=0, max=max(T - 1, 0), step=1,
                                  initial_value=0)
    dt = 1.0 / args.hz
    while True:
        if keys is not None and keys.hit_ctrl_s() and on_save:
            on_save()
        i = int(frame.value)
        pts = np.asarray(node_pos[i], dtype=np.float64)
        if ok:
            gd = _glove_data_from_nodes(pts)
            skel_pts = normalize_hand_positions(gd, side=args.side)
            if latest is not None:
                latest["skel"] = skel_pts
            ts = scale(skel_pts)
            qpos = apply_distal_coupling(
                retargeter.retarget(gd, ts), coupling, args.dip_coupling)
            if args.thumb_mode == "joint":
                qpos = apply_thumb_joint(qpos, gd, thumb_tracker,
                                         args.thumb_joint_weight)
            update_targets(targets, retargeter, skel_pts, ts, offset)
        else:
            qpos = np.zeros(len(joint_names))
            skel_pts = pts if pts.ndim == 2 and len(pts) else None
        update_hand(sharpa_vis, skel, joint_names, qpos, skel_pts, offset)
        if play.value and T > 1:
            frame.value = (i + 1) % T
        time.sleep(dt)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = rp.load()                     # shared retargeting params (YAML defaults)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["live", "replay"], default="live")
    p.add_argument("--npz", default=None, help="recording for --source replay")
    p.add_argument("--side", choices=["right"], default="right")
    p.add_argument("--manus-host", default="localhost")
    p.add_argument("--manus-port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=60.0)
    p.add_argument("--port", type=int, default=8080, help="viser web port")
    p.add_argument("--skeleton-offset", type=float, default=0.0,
                   help="Y offset (m) to place the skeleton beside the hand")
    p.add_argument("--dip-coupling", type=float, default=cfg["dip_coupling"],
                   help="Couple DIP/thumb-IP to the proximal joint at this ratio "
                        "(0 disables). Tunable live.")
    p.add_argument("--scaling", type=float, default=cfg["scaling"],
                   help="Master size multiplier across all fingers. Tunable live.")
    p.add_argument("--finger-scales", default=rp.finger_scales_str(cfg),
                   help="Per-finger size (thumb,index,middle,ring,pinky) on top of "
                        "the auto-calibrated size. Tunable live.")
    p.add_argument("--show-targets", action=argparse.BooleanOptionalAction,
                   default=True, help="Draw the optimizer target markers.")
    p.add_argument("--thumb-mode", choices=["shape", "joint"],
                   default=cfg["thumb_mode"],
                   help="thumb retargeting: 'shape' (fingertip-pose optimizer, "
                        "default) or 'joint' (direct from Manus thumb ergonomics, "
                        "auto-ranged). Tunable live.")
    p.add_argument("--thumb-ema", type=float, default=cfg["thumb_ema"],
                   help="EMA smoothing for thumb joint mode (0<a<=1; 1=off, "
                        "lower=smoother). Tunable live.")
    p.add_argument("--thumb-joint-weight", type=float,
                   default=cfg["thumb_joint_weight"],
                   help="In thumb 'joint' mode, blend toward the measured thumb "
                        "joint angles vs the fingertip-pose optimizer (1=pure "
                        "joint/old behavior, 0=pure pose, in between=hybrid). "
                        "Tunable live.")
    p.add_argument("--ab-correct-deg", type=float, default=cfg["ab_correct_deg"],
                   help="rotate the whole hand about the palm normal by this many "
                        "degrees to cancel a uniform abduction lean. Tunable live "
                        "(slider + 'Auto abduction fix' button).")
    args = p.parse_args()

    joint_names = get_urdf_joint_names(args.side)
    coupling = build_distal_coupling(joint_names)
    thumb_tracker = ThumbErgoTracker(joint_names, str(_SHARPA_URDF),
                                     ema_alpha=args.thumb_ema)
    for short, inv in cfg.get("thumb_invert", {}).items():
        if short in thumb_tracker.invert_targets:
            thumb_tracker.invert_targets[short][5] = bool(inv)
    _AB_FIX["deg"] = args.ab_correct_deg
    latest = {"skel": None}   # most recent normalized skeleton (for Auto abduction fix)
    offset = np.array([0.0, args.skeleton_offset, 0.0], dtype=np.float64)
    vals = [float(x) for x in args.finger_scales.split(",")]
    if len(vals) != len(FINGER_ORDER):
        p.error(f"--finger-scales needs {len(FINGER_ORDER)} values")
    finger_scales = dict(zip(FINGER_ORDER, vals))

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    sharpa_vis, skel = build_scene(server)
    print(f"[viz] Viser server -> http://0.0.0.0:{args.port}")

    retargeter = ShapeRetargeter(side=args.side)
    print(f"[retarget] shape strategy ready ({len(retargeter.tasks)} tasks); "
          f"master={args.scaling}, finger={finger_scales}, "
          f"dip_coupling={args.dip_coupling}")

    sliders = {}
    try:
        sliders["master"] = server.gui.add_slider("Size: master", min=0.3, max=1.6,
                                                  step=0.01, initial_value=args.scaling)
        for f in FINGER_ORDER:
            sliders[f] = server.gui.add_slider(f"Size: {f}", min=0.3, max=1.6,
                                               step=0.01, initial_value=finger_scales[f])
        dc = server.gui.add_slider("DIP coupling", min=0.0, max=1.0, step=0.01,
                                   initial_value=args.dip_coupling)
        dc.on_update(lambda _: setattr(args, "dip_coupling", float(dc.value)))
        tm = server.gui.add_checkbox("Thumb: direct joint (ergo)",
                                     initial_value=(args.thumb_mode == "joint"))
        tm.on_update(lambda _: setattr(args, "thumb_mode",
                                       "joint" if tm.value else "shape"))
        te = server.gui.add_slider("Thumb EMA", min=0.02, max=1.0, step=0.02,
                                   initial_value=args.thumb_ema)
        te.on_update(lambda _: setattr(thumb_tracker, "ema_alpha", float(te.value)))
        tw = server.gui.add_slider("Thumb joint weight", min=0.0, max=1.0, step=0.02,
                                   initial_value=args.thumb_joint_weight)
        tw.on_update(lambda _: setattr(args, "thumb_joint_weight", float(tw.value)))
        for short, entry in thumb_tracker.invert_targets.items():
            cb = server.gui.add_checkbox(f"Invert thumb {short}",
                                         initial_value=bool(entry[5]))
            cb.on_update(lambda _, e=entry, c=cb: e.__setitem__(5, bool(c.value)))
        ab = server.gui.add_slider("Abduction fix (deg)", min=-25.0, max=25.0,
                                   step=0.5, initial_value=args.ab_correct_deg)
        ab.on_update(lambda _: _AB_FIX.__setitem__("deg", float(ab.value)))
        ab_auto = server.gui.add_button("Auto abduction fix")

        @ab_auto.on_click
        def _ab_auto(_):  # noqa: ANN001
            skel = latest["skel"]
            if skel is None:
                print("[ab-fix] no skeleton yet -- move the glove first")
                return
            # skel already has the current fix applied; add what cancels the residual
            new = float(np.clip(_AB_FIX["deg"] + _common_mode_lean_deg(skel),
                                -25.0, 25.0))
            _AB_FIX["deg"] = new
            ab.value = new
            print(f"[ab-fix] auto-set abduction fix -> {new:+.1f} deg")
    except Exception as exc:
        print(f"[viz] GUI sliders unavailable: {exc}")

    def scale(skel_pts):
        retargeter.update_calibration(skel_pts)
        master = sliders["master"].value if "master" in sliders else args.scaling
        fvals = {f: (sliders[f].value if f in sliders else finger_scales[f])
                 for f in FINGER_ORDER}
        return compute_task_scales(master, fvals, retargeter.task_fingers,
                                   retargeter.base_scale)

    def save_params():
        get = lambda k, d: (sliders[k].value if k in sliders else d)  # noqa: E731
        params = {
            "scaling": float(get("master", args.scaling)),
            "finger_scales": {f: float(get(f, finger_scales[f]))
                              for f in FINGER_ORDER},
            "dip_coupling": float(args.dip_coupling),
            "ab_correct_deg": float(_AB_FIX["deg"]),
            "thumb_mode": args.thumb_mode,
            "thumb_ema": float(thumb_tracker.ema_alpha),
            "thumb_joint_weight": float(args.thumb_joint_weight),
            "thumb_invert": {s: bool(e[5])
                             for s, e in thumb_tracker.invert_targets.items()},
        }
        print(f"[params] saved current retargeting params -> {rp.save(params)}")

    try:
        server.gui.add_button("Save params (Ctrl-S)").on_click(
            lambda _: save_params())
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] save button unavailable: {exc}")

    targets = None
    if args.show_targets:
        cols = np.array([[235, 40, 40] if a == 0 else [255, 150, 30]
                         for a in retargeter._indices[0]])
        targets = TargetMarkers(server, cols)
    scene = (sharpa_vis, skel, targets)

    keys = _CtrlSPoller()
    print(f"[params] Ctrl-S (terminal) or the 'Save params' button -> {rp.PARAMS_PATH}")
    try:
        if args.source == "live":
            run_live(args, retargeter, joint_names, scene, scale, coupling, offset,
                     thumb_tracker, latest, on_save=save_params, keys=keys)
        else:
            if args.npz is None:
                p.error("--source replay requires --npz")
            run_replay(args, retargeter, joint_names, scene, scale, coupling, offset,
                       server, thumb_tracker, latest, on_save=save_params, keys=keys)
    finally:
        keys.restore()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[viz] shutting down.")
