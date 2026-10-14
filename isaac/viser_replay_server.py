"""Browser replay server for saved rollouts (play-policy OR world-model MPC).

Serves the SAME viser 3D scene used during data collection -- Sharpa's URDF
(solid link meshes via ViserUrdf) plus a textured Rubik cube -- and REPLAYS a
recorded rollout .npz with full in-browser playback controls: play/pause, a
scrub timeline, speed, loop, and single-frame stepping. Point it at one .npz or
a whole directory and a dropdown switches between every rollout it finds.

Handles both recording schemas transparently (they share ``t``):
  * teleop takes and self-play recordings (the dataset's teleop/ and play/ files,
    data/teleop_takes/*.npz); files with ``episode`` + ``hz`` get a short pause at
    episode boundaries.
  * world-model MPC rollouts (<mpc out>/roll_XXXX.npz): carry a GOAL
    orientation + per-step geodesic. A second Rubik cube is shown beside the
    hand at the goal orientation (match the stickers = solved) and the live
    to-goal angle / running min / final is read out in the GUI.

If a sibling ``cand_XXXX.npz`` exists (a dump of the MPC candidates), the
256 candidate action chunks the world model scored at each replan are overlaid as
fingertip trajectories -- one hue family per finger (thumb red ... pinky grey),
brightness by WM score (vivid = low loss, fading to white), the executed pick bold
gold. Toggle / thin them from the "Candidates" GUI folder.

The in-sim MPC cube is a plain cuboid; drawing it Rubik-textured here is a viz
aid so orientation (and goal alignment) is legible -- only the recorded pose is
used. No Isaac Sim / GPU needed; runs headless and stays up until Ctrl-C, so you can
connect any time.

  python isaac/viser_replay_server.py --rollout data/teleop_takes      # dir -> dropdown of all
  python isaac/viser_replay_server.py --rollout <mpc out dir> --port 8080

Needs viser + yourdfpy + trimesh (all in the Isaac Lab env), or just use scripts/replay.sh.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# same Rubik face scheme as teleop_isaac
_RUBIK_FACE_FILES = ["rubik_face_white.png", "rubik_face_yellow.png",
                     "rubik_face_red.png", "rubik_face_orange.png",
                     "rubik_face_green.png", "rubik_face_blue.png"]


def build_rubik_mesh(size):
    """Textured box (edge=`size` m), each face UV-mapped to a 3x3 sticker image
    (identical to teleop_isaac.build_rubik_mesh)."""
    import trimesh
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    imgs = [Image.open(os.path.join(_HERE, "assets", f)).convert("RGB") for f in _RUBIK_FACE_FILES]
    tile = imgs[0].width
    atlas = Image.new("RGB", (tile * 6, tile))
    for i, im in enumerate(imgs):
        atlas.paste(im.resize((tile, tile)), (i * tile, 0))
    h = size / 2.0
    faces_xyz = [
        [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)],
        [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)],
        [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)],
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],
        [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)],
    ]
    verts, tris, uvs = [], [], []
    for i, corners in enumerate(faces_xyz):
        b = len(verts)
        verts += corners
        u0, u1 = i / 6.0, (i + 1) / 6.0
        uvs += [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)]
        tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    mesh = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64), faces=np.array(tris), process=False)
    mesh.visual = TextureVisuals(uv=np.array(uvs), material=SimpleMaterial(image=atlas))
    return mesh


def _resolve(path):
    """Accept a path relative to CWD or to the repo root (so it works anywhere)."""
    if os.path.exists(path):
        return path
    alt = os.path.join(_ROOT, path)
    return alt if os.path.exists(alt) else path


def discover_rollouts(path):
    """[(label, abspath)] for a rollout .npz file or a dir of them (recursively).

    A file counts as a rollout only if it holds ``hand_joint_pos`` -- this skips
    the sibling ``traj_XXXX.npz`` (metrics-only) that MPC runs write alongside.
    """
    path = _resolve(path)
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return [(os.path.basename(p), p)]
    hits = []
    for f in sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)):
        try:
            with np.load(f, allow_pickle=True) as d:
                if "hand_joint_pos" not in d.files:
                    continue
        except Exception:
            continue
        hits.append((os.path.relpath(f, p), f))
    return hits


def load_rollout(path):
    """Load one rollout into a plain dict, filling optional fields with None."""
    d = np.load(path, allow_pickle=True)
    files = set(d.files)
    hand = np.asarray(d["hand_joint_pos"], float)              # (T, 22)
    names = [str(n) for n in d["hand_joint_names"]]
    cube = np.asarray(d["cube_state"], float)                  # (T, >=7): pos3 + quat4(wxyz)
    T = len(hand)
    cube_size = float(d["cube_size"]) if "cube_size" in files else 0.05
    hroot_pos = np.asarray(d["hand_root_pos"], float) if "hand_root_pos" in files else np.zeros(3)
    hroot_quat = np.asarray(d["hand_root_quat"], float) if "hand_root_quat" in files else np.array([1.0, 0, 0, 0])
    if "t" in files:
        tarr = np.asarray(d["t"], float).ravel()
    elif "hz" in files:
        tarr = np.arange(T) / max(float(d["hz"]), 1e-6)
    else:
        tarr = np.arange(T) / 60.0
    episode = np.asarray(d["episode"]).ravel().astype(int) if "episode" in files else None
    goal_quat = np.asarray(d["goal_quat"], float).ravel() if "goal_quat" in files else None
    geo = np.asarray(d["geo"], float).ravel() if "geo" in files else None
    # nominal seconds-per-frame for real-time pacing (robust to gaps / episode resets)
    frame_dt = 1.0 / 60.0
    if T > 1:
        dts = np.diff(tarr)
        dts = dts[(dts > 0) & np.isfinite(dts)]
        if len(dts):
            frame_dt = float(np.median(dts))
    # closest approach to the goal (MPC rollouts): the min-geodesic frame
    best_i = best_p = best_geo = None
    if geo is not None and len(geo):
        best_i = int(np.argmin(geo))
        best_geo = float(geo[best_i])
        best_p = best_i / max(T - 1, 1)
    # optional candidate overlay: sibling cand_XXXX.npz (an MPC candidate dump)
    cand = None
    mo = re.search(r"roll_(\d+)\.npz$", os.path.basename(path))
    if mo:
        cpath = os.path.join(os.path.dirname(path), f"cand_{mo.group(1)}.npz")
        if os.path.exists(cpath):
            cd = np.load(cpath, allow_pickle=True)
            cf = set(cd.files)
            hstr = int(cd["h_stride"]) if "h_stride" in cf else 5
            cand = {"fingertips": cd["fingertips"],                       # (K,N,P,5,3) fp16, base frame
                    "scores": np.asarray(cd["scores"], np.float64),       # (K,N) loss
                    "exec_idx": np.asarray(cd["exec_idx"], int),          # (K,)
                    "plan_frames": np.asarray(cd["plan_frames"], int),    # (K,)
                    "h_stride": hstr,                                     # horizon steps between path points
                    "horizon": int(cd["horizon"]) if "horizon" in cf else cd["fingertips"].shape[2] * hstr}
    return dict(hand=hand, names=names, cube=cube, cube_size=cube_size,
                hroot_pos=hroot_pos, hroot_quat=hroot_quat, tarr=tarr, T=T,
                episode=episode, goal_quat=goal_quat, geo=geo, frame_dt=frame_dt,
                best_i=best_i, best_p=best_p, best_geo=best_geo, cand=cand, path=path)


def info_text(d, fi):
    T = d["T"]
    t = d["tarr"][min(fi, len(d["tarr"]) - 1)]
    lines = [f"**{os.path.basename(d['path'])}**", "",
             f"frame&nbsp;&nbsp;**{fi} / {T - 1}**&nbsp;&nbsp;·&nbsp;&nbsp;t&nbsp;=&nbsp;{t:.2f}s"]
    if d["geo"] is not None:
        g = d["geo"]
        cur = g[min(fi, len(g) - 1)]
        lines += ["", f"to-goal&nbsp;&nbsp;**{cur:.1f}°**"]
        if d.get("best_i") is not None:
            bi = d["best_i"]
            tb = d["tarr"][min(bi, len(d["tarr"]) - 1)]
            here = "&nbsp;&nbsp;◀&nbsp;here" if fi == bi else ""
            lines.append(f"◆&nbsp;closest&nbsp;**{d['best_geo']:.1f}°**&nbsp;@&nbsp;frame&nbsp;{bi}"
                         f"&nbsp;(t&nbsp;{tb:.1f}s){here}")
        lines.append(f"final&nbsp;{float(g[-1]):.1f}°")
    if d["episode"] is not None:
        ep = d["episode"]
        lines += ["", f"episode&nbsp;{int(ep[min(fi, len(ep) - 1)])} / {int(ep.max())}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate-action overlay (world-model MPC): the 256 proposed action chunks per
# replan, FK'd to fingertip paths offline (cand_XXXX.npz next to the rollout).
# ---------------------------------------------------------------------------
# One hue family per finger; a candidate's score (geodesic-to-goal LOSS, lower is
# better) sets brightness -- best candidates show the vivid hue, worst fade to white.
FINGER_HUES = np.array([
    [214, 39, 40],     # thumb  - red
    [31, 119, 180],    # index  - blue
    [44, 160, 44],     # middle - green
    [255, 127, 14],    # ring   - orange
    [127, 127, 127],   # pinky  - grey
], dtype=np.float32)
EXEC_COLOR = np.array([255, 215, 0], dtype=np.uint8)   # bright gold, distinct from every finger hue
FINGER_LEGEND = "thumb&nbsp;red · index&nbsp;blue · middle&nbsp;green · ring&nbsp;orange · pinky&nbsp;grey"


def build_candidate_segments(fingertips, scores, exec_idx, s_min=None, s_max=None, eps=1e-9):
    """(N,P,5,3) fingertip paths + (N,) LOSS scores -> viser line segments (vectorized).

    Returns candidate segments (executed excluded) coloured by finger x reversed-loss
    (best = vivid hue, worst = white), plus the executed candidate's segments alone
    (drawn bold gold). viser wants per-ENDPOINT colours of shape (M,2,3) uint8 --
    per-segment (M,3) is rejected. Pass the full replan's ``s_min``/``s_max`` so the
    colormap stays stable when only a top-K subset is shown.
    """
    fingertips = np.asarray(fingertips, np.float32)
    scores = np.asarray(scores, np.float64)
    N = fingertips.shape[0]
    s_min = float(scores.min()) if s_min is None else float(s_min)
    s_max = float(scores.max()) if s_max is None else float(s_max)
    denom = s_max - s_min
    w = (np.ones(N, np.float32) if denom < eps
         else np.clip((s_max - scores) / (denom + eps), 0.0, 1.0).astype(np.float32))   # 1 = best
    wc = w[:, None, None]
    cf = np.clip(np.rint((1.0 - wc) * 255.0 + wc * FINGER_HUES[None, :, :]), 0, 255).astype(np.uint8)  # (N,5,3)

    seg = np.stack([fingertips[:, :-1], fingertips[:, 1:]], axis=-2)          # (N,P-1,5,2,3)
    seg_col = np.broadcast_to(cf[:, None, :, None, :], seg.shape)             # (N,P-1,5,2,3)
    keep = np.ones(N, bool)
    has_exec = 0 <= exec_idx < N
    if has_exec:
        keep[exec_idx] = False
    return dict(
        cand_points=np.ascontiguousarray(seg[keep].reshape(-1, 2, 3), np.float32),
        cand_colors=np.ascontiguousarray(seg_col[keep].reshape(-1, 2, 3), np.uint8),
        exec_points=(np.ascontiguousarray(seg[exec_idx].reshape(-1, 2, 3), np.float32)
                     if has_exec else np.zeros((0, 2, 3), np.float32)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", default="data",
                    help="rollout .npz OR a directory of them, searched recursively (default: data/)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed multiplier")
    ap.add_argument("--ep-pause", type=float, default=0.4, help="seconds paused at episode / loop boundaries")
    ap.add_argument("--fps", type=float, default=30.0, help="render tick-rate cap")
    ap.add_argument("--goal-offset", type=float, nargs=3, default=[0.15, 0.0, 0.07],
                    help="goal cube offset from the hand root (MPC rollouts)")
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    rollouts = discover_rollouts(args.rollout)
    if not rollouts:
        raise SystemExit(f"[replay] no rollout .npz (with hand_joint_pos) under: {args.rollout}")
    labels = [lab for lab, _ in rollouts]
    by_label = dict(rollouts)
    print(f"[replay] {len(rollouts)} rollout(s) from {args.rollout}")

    data = load_rollout(rollouts[0][1])
    print(f"[replay] first: {os.path.basename(data['path'])}  frames={data['T']}  "
          f"goal={'yes' if data['goal_quat'] is not None else 'no'}  "
          f"episodes={0 if data['episode'] is None else len(np.unique(data['episode']))}")

    srv = viser.ViserServer(host="0.0.0.0", port=args.port)
    srv.gui.configure_theme(control_width="large", brand_color=(0, 128, 128))  # wide controls; teal slider fill
    urdf_path = Path(os.path.join(_ROOT, "assets", "urdf", "sharpa", "right_fixed.urdf"))
    hand_frame = srv.scene.add_frame("/hand", position=tuple(map(float, data["hroot_pos"])),
                                     wxyz=tuple(map(float, data["hroot_quat"])), show_axes=False)
    vurdf = ViserUrdf(srv, urdf_path, root_node_name="/hand")
    srv.scene.add_grid("/floor", width=1.0, height=1.0, cell_size=0.05)

    # Initial camera: a CLOSE view framed from the real scene geometry (center
    # [0.06,0.07,0.31], radius 0.17 m, ~0.36 m back). Viewed from the +Y/+X
    # hemisphere (az +60, el 20) so it faces the palm / working face -- from the
    # -Y side the SOLID hand meshes occlude the cube. Fires for every client.
    @srv.on_client_connect
    def _(client):
        client.camera.position = (0.232, 0.364, 0.435)
        client.camera.up_direction = (0.0, 0.0, 1.0)   # Z-up; set before look_at
        client.camera.look_at = (0.061, 0.068, 0.311)

    scene = {}  # live scene nodes, rebuilt when cube size changes

    def place_goal(d):
        if d["goal_quat"] is None:
            scene["goal"].visible = False
            return
        gp = np.asarray(d["hroot_pos"], float) + np.asarray(args.goal_offset, float)
        scene["goal"].position = tuple(map(float, gp))
        scene["goal"].wxyz = tuple(map(float, d["goal_quat"]))
        scene["goal"].visible = True

    def build_cubes(d):
        # add_mesh_trimesh with an existing name replaces the node -> no manual remove
        scene["cube"] = srv.scene.add_mesh_trimesh("/cube", build_rubik_mesh(d["cube_size"]))
        scene["goal"] = srv.scene.add_mesh_trimesh("/goal", build_rubik_mesh(d["cube_size"]))
        place_goal(d)

    build_cubes(data)

    # ---- shared playback state (mutated from viser's callback thread + main loop) ----
    st = {"p": 0.0, "playing": True, "speed": float(args.speed), "loop": True,
          "reload_to": None, "data": data}

    def set_p(p):
        st["p"] = min(max(float(p), 0.0), 1.0)

    ui = {}                                        # widgets rebuilt per rollout (timeline, jump)
    folder_pb = srv.gui.add_folder("Playback")
    with folder_pb:
        play_cb = srv.gui.add_checkbox("Play", True, order=1)
        # The Timeline slider IS the progress bar: its filled (teal) track shows playback
        # position and it's draggable to seek. It + the "◆ jump to closest" button carry a
        # per-rollout mark, so both are (re)built in rebuild_seek_ui() -- marks are fixed
        # at slider creation.
        speed_sl = srv.gui.add_slider("Speed", 0.1, 4.0, 0.1, float(args.speed), order=5)
        loop_cb = srv.gui.add_checkbox("Loop", True, order=6)
        btn_prev = srv.gui.add_button("◀ −1 frame", order=7)
        btn_next = srv.gui.add_button("+1 frame ▶", order=8)

    dd = None
    if len(rollouts) > 1:
        with srv.gui.add_folder("Rollout"):
            dd = srv.gui.add_dropdown("File", labels, initial_value=labels[0])

    with srv.gui.add_folder("Info"):
        info_md = srv.gui.add_markdown(info_text(data, 0))

    # candidate-action overlay controls (active only when a cand_XXXX.npz was found)
    cst = {"k": -1, "dirty": True}                 # active replan shown + rebuild flag
    _c0 = data.get("cand")
    n_cand = int(_c0["scores"].shape[1]) if _c0 is not None else 256
    cand_horizon = int(_c0["horizon"]) if _c0 is not None else 200
    cand_hstride = int(_c0["h_stride"]) if _c0 is not None else 5
    with srv.gui.add_folder("Candidates"):
        cand_show = srv.gui.add_checkbox("Show candidates", True)
        cand_exec = srv.gui.add_checkbox("Show executed (gold)", True)
        cand_topk = srv.gui.add_slider("Top-K by score", 1, max(n_cand, 2), 1, n_cand)
        cand_steps = srv.gui.add_slider("Traj steps", 0, cand_horizon, cand_hstride, cand_horizon)
        cand_md = srv.gui.add_markdown("")

    # ---- callbacks: ignore server-originated updates (event.client is None) ----
    @play_cb.on_update
    def _(e):
        if e.client is None:
            return
        st["playing"] = play_cb.value
        if play_cb.value and st["p"] >= 1.0:   # replay from the top if pressing play at the end
            set_p(0.0)

    @speed_sl.on_update
    def _(e):
        if e.client is None:
            return
        st["speed"] = float(speed_sl.value)

    @loop_cb.on_update
    def _(e):
        if e.client is None:
            return
        st["loop"] = bool(loop_cb.value)

    def step(delta):
        set_p(st["p"] + delta / max(st["data"]["T"] - 1, 1))
        st["playing"] = False
        play_cb.value = False

    btn_prev.on_click(lambda e: step(-1))
    btn_next.on_click(lambda e: step(+1))

    if dd is not None:
        @dd.on_update
        def _(e):
            if e.client is None:
                return
            st["reload_to"] = by_label[dd.value]

    def rebuild_seek_ui(d):
        """(Re)create the Timeline scrubber + jump-to-closest button for rollout `d`.

        The Timeline carries a ◆ mark at the closest-approach frame (min geodesic-
        to-goal); the button seeks straight there, snapping the live cube next to
        the goal cube so you can eyeball how close it got. Both are rebuilt on every
        rollout switch because slider marks are fixed at creation time.
        """
        for k in ("timeline", "jump"):
            if ui.get(k) is not None:
                ui[k].remove()
                ui[k] = None
        if d.get("best_p") is not None:
            marks = ((0.0, "start"), (float(d["best_p"]), f"◆ {d['best_geo']:.0f}°"), (1.0, "end"))
        else:
            marks = ((0.0, "start"), (1.0, "end"))
        with folder_pb:
            tl = srv.gui.add_slider("Timeline", 0.0, 1.0, 0.001, float(st["p"]), marks=marks, order=2)
            ui["timeline"] = tl

            @tl.on_update
            def _(e):
                if e.client is None:
                    return
                set_p(tl.value)

            if d.get("best_p") is not None:
                jb = srv.gui.add_button(f"◆ Jump to closest ({d['best_geo']:.0f}°)",
                                        color="teal", order=3)
                ui["jump"] = jb

                @jb.on_click
                def _(e, bp=float(d["best_p"])):
                    set_p(bp)
                    st["playing"] = False
                    play_cb.value = False

    rebuild_seek_ui(data)

    def frame_index():
        return int(round(st["p"] * (st["data"]["T"] - 1)))

    def push(fi):
        d = st["data"]
        vurdf.update_cfg({n: float(d["hand"][fi, i]) for i, n in enumerate(d["names"])})
        c = d["cube"][fi]
        scene["cube"].position = (float(c[0]), float(c[1]), float(c[2]))
        scene["cube"].wxyz = (float(c[3]), float(c[4]), float(c[5]), float(c[6]))

    def apply_reload(path):
        nd = load_rollout(path)
        st["data"] = nd
        hand_frame.position = tuple(map(float, nd["hroot_pos"]))
        hand_frame.wxyz = tuple(map(float, nd["hroot_quat"]))
        build_cubes(nd)
        set_p(0.0)
        rebuild_seek_ui(nd)
        cst["k"] = -1
        cst["dirty"] = True
        st["playing"] = True
        play_cb.value = True

    @cand_show.on_update
    def _(e):
        if e.client is not None:
            cst["dirty"] = True

    @cand_exec.on_update
    def _(e):
        if e.client is not None:
            cst["dirty"] = True

    @cand_topk.on_update
    def _(e):
        if e.client is not None:
            cst["dirty"] = True

    @cand_steps.on_update
    def _(e):
        if e.client is not None:
            cst["dirty"] = True

    def active_replan(fi):
        cand = st["data"].get("cand")
        if cand is None:
            return -1
        pf = cand["plan_frames"]
        return int(np.clip(np.searchsorted(pf, fi, side="right") - 1, 0, len(pf) - 1))

    def render_candidates(k):
        """(Re)build the candidate + executed line-segment nodes for replan k.

        Drawn under /hand (base frame -> inherits the hand mount). Fingertip paths
        are static per replan; this fires when the active replan changes or a toggle
        is flipped. Cheap: ~50k segments build + upload in a few ms.
        """
        cand = st["data"].get("cand")
        if cand is None or not cand_show.value or k < 0:
            for nm in ("cand", "exec"):
                if scene.get(nm) is not None:
                    scene[nm].visible = False
            cand_md.content = ("_no candidate data (no cand_XXXX.npz next to this rollout)_"
                               if cand is None else "_candidates hidden_")
            return
        ft = np.asarray(cand["fingertips"][k], np.float32)       # (N,P,5,3) base frame
        sc = cand["scores"][k]                                   # (N,) loss (lower better)
        ex = int(cand["exec_idx"][k])
        N, P = ft.shape[0], ft.shape[1]
        hstr = int(cand.get("h_stride", 5))
        horizon = int(cand.get("horizon", (P - 1) * hstr))
        # "Traj steps" slider -> keep the first n_pts path points (each = hstr horizon steps)
        n_pts = int(np.clip(int(cand_steps.value) // max(hstr, 1) + 1, 1, P))
        if n_pts < 2:                                            # 0 steps -> draw nothing
            for nm in ("cand", "exec"):
                if scene.get(nm) is not None:
                    scene[nm].visible = False
            cand_md.content = f"_showing 0 / {horizon} steps — raise “Traj steps”_"
            return
        m = int(min(max(int(cand_topk.value), 1), N))
        sel = np.argsort(sc)[:m]                                 # keep the best (lowest-loss) m
        if ex not in sel:
            sel = np.append(sel, ex)
        ex_local = int(np.where(sel == ex)[0][0])
        out = build_candidate_segments(ft[sel][:, :n_pts], sc[sel], ex_local,
                                       s_min=float(sc.min()), s_max=float(sc.max()))
        scene["cand"] = srv.scene.add_line_segments("/hand/candidates", out["cand_points"],
                                                    out["cand_colors"], line_width=1.5)
        scene["exec"] = srv.scene.add_line_segments("/hand/executed", out["exec_points"],
                                                    EXEC_COLOR, line_width=5.0)
        scene["cand"].visible = True
        scene["exec"].visible = bool(cand_exec.value)
        cand_md.content = (f"**{N} candidates** · replan {k + 1}/{len(cand['plan_frames'])}"
                           f"{'' if m >= N else f' · top&nbsp;{m}'}"
                           f" · steps&nbsp;{min(int(cand_steps.value), horizon)}/{horizon}\n\n"
                           f"{FINGER_LEGEND}\n\n"
                           f"brightness = WM score: vivid = low loss (better) → white = worse\n\n"
                           f"**gold** = executed (WM pick) · best&nbsp;{float(sc.min()):.1f}°")

    print(f"[replay] viser 3D view -> http://0.0.0.0:{args.port}   (Ctrl-C to stop; loops forever)")

    tick = 1.0 / max(args.fps, 1.0)
    prev_fi = -1
    force = True
    last = time.time()
    try:
        while True:
            now = time.time()
            elapsed = now - last
            last = now

            if st["reload_to"] is not None:
                apply_reload(st["reload_to"])
                st["reload_to"] = None
                prev_fi = -1
                force = True
                last = time.time()
                continue

            d = st["data"]
            T = d["T"]
            if st["playing"] and T > 1:
                newp = st["p"] + st["speed"] * elapsed / (d["frame_dt"] * (T - 1))
                if newp >= 1.0:
                    if st["loop"]:
                        set_p(1.0)
                        push(frame_index())
                        ui["timeline"].value = 1.0
                        time.sleep(args.ep_pause)      # brief hold on the last frame, then wrap
                        set_p(0.0)
                        last = time.time()
                    else:
                        set_p(1.0)
                        st["playing"] = False
                        play_cb.value = False
                else:
                    set_p(newp)

            fi = frame_index()
            if fi != prev_fi or force:
                ep = d["episode"]
                if (not force) and ep is not None and prev_fi >= 0 and fi > prev_fi \
                        and ep[min(fi, len(ep) - 1)] != ep[min(prev_fi, len(ep) - 1)]:
                    time.sleep(args.ep_pause)
                    last = time.time()
                push(fi)
                # server-side .value set -> callback sees client=None -> ignored (no feedback loop)
                ui["timeline"].value = float(st["p"])
                info_md.content = info_text(d, fi)
                prev_fi = fi
                force = False

            k = active_replan(fi)
            if k != cst["k"] or cst["dirty"]:
                render_candidates(k)
                cst["k"] = k
                cst["dirty"] = False

            rest = tick - (time.time() - now)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        print("\n[replay] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
