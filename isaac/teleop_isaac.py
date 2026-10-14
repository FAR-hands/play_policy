#!/usr/bin/env python3
"""Manus -> Sharpa teleop playground in Isaac Lab (free play with a Rubik's cube).

A fixed Sharpa right hand (palm up) holds a rigid Rubik's-textured cube; the Manus
glove drives the fingers via the shape retargeter (retarget/sharpa_retarget.py).
Reorientation free-play -- push the cube around in-hand. This is how the human play
demos (the play policy's training data) are collected; see docs/TELEOP.md.

Runs in the Isaac Lab python env. Sources:
  --source live    : live Manus glove (needs glove + license dongle + manus_zmq_server on :5555)
  --source demo    : no hardware -- fingers follow a slow open/close sinusoid
  --source replay  : drive from a recording: a dataset take (replays its cmd_q22 joint
                     commands) or a raw glove recording with manus_node_pos (re-retargeted)

    OMNI_KIT_ACCEPT_EULA=YES python isaac/teleop_isaac.py --source demo
    OMNI_KIT_ACCEPT_EULA=YES python isaac/teleop_isaac.py --source live --side right
    OMNI_KIT_ACCEPT_EULA=YES python isaac/teleop_isaac.py --source replay --npz <take>.npz

Takes are written to data/teleop_takes/ (--out-dir). The hand USD ships in
isaac/assets/sharpa_right/; isaac/convert_sharpa_usd.py regenerates it from the URDF.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)         # repo root
sys.path.insert(0, os.path.join(_ROOT, "retarget"))   # sharpa_retarget, manus_client, retarget_params
from sim_params import CONTROL_HZ      # noqa: E402  (isaac/ is the script dir; no Isaac imports at module level)
import retarget_params as _rp           # noqa: E402  shared retargeting params (YAML)
_CFG = _rp.load()
_USD = os.path.join(_HERE, "assets", "sharpa_right", "sharpa_right.usd")
_RUBIK_TEX = os.path.join(_HERE, "assets", "rubik_sticker.png")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--source", choices=["live", "demo", "replay"], default="demo")
parser.add_argument("--npz", default=None,
                    help="recording for --source replay: a take (.npz with cmd_q22, e.g. from the "
                         "dataset) or a raw glove recording (.npz with manus_node_pos)")
parser.add_argument("--side", choices=["right"], default="right")
parser.add_argument("--manus-host", default="localhost")
parser.add_argument("--manus-port", type=int, default=5555)
parser.add_argument("--glove-timeout", type=float, default=20.0,
                    help="--source live: seconds to wait for the first glove skeleton before giving "
                         "up (checked BEFORE Isaac boots, so a missing server fails fast)")
parser.add_argument("--hz", type=float, default=CONTROL_HZ,
                    help="control-loop rate. Physics advances 1/240 s per frame whatever this is; "
                         "the rate sets how much hand motion lands in each frame, so keep it at "
                         "the value the training pool was recorded at (sim_params.CONTROL_HZ)")
parser.add_argument("--usd", default=_USD)
parser.add_argument("--cube-size", type=float, default=0.05,
                    help="cube edge (m)")
parser.add_argument("--scaling", type=float, default=_CFG["scaling"],
                    help="retarget master size (default from retarget_params.yaml)")
parser.add_argument("--dip-coupling", type=float, default=_CFG["dip_coupling"])
parser.add_argument("--ab-correct-deg", type=float, default=_CFG["ab_correct_deg"],
                    help="rotate the hand about the palm normal by this many degrees "
                         "to cancel a uniform abduction lean (see retarget_viz.py)")
parser.add_argument("--thumb-mode", choices=["shape", "joint"],
                    default=_CFG["thumb_mode"],
                    help="thumb retargeting: 'shape' (optimizer) or 'joint' (direct "
                         "from Manus thumb ergonomics). Default from retarget_params.yaml")
parser.add_argument("--thumb-joint-weight", type=float,
                    default=_CFG["thumb_joint_weight"],
                    help="thumb 'joint' mode blend weight: 1=pure ergo (old), "
                         "0=pure fingertip-pose optimizer, in between=hybrid. "
                         "Default from retarget_params.yaml")
# Hand mounting: position (m) and orientation (deg, XYZ euler) of the URDF root.
# Default rpy mounts the hand palm-up, fingers forward (computed from the URDF).
parser.add_argument("--hand-pos", type=float, nargs=3, default=[0.0, 0.0, 0.30])
parser.add_argument("--hand-rpy", type=float, nargs=3, default=[-173.746, -59.607, 94.102],
                    help="hand root euler XYZ in degrees (palm-up, rotated -30deg "
                         "about world X from the original [108.8, -87.9, 171.0])")
parser.add_argument("--cube-offset", type=float, nargs=3, default=[0.0, 0.07, 0.06],
                    help="cube spawn offset from the hand root (m), in world axes")
parser.add_argument("--stiffness", type=float, default=3.0)
parser.add_argument("--damping", type=float, default=0.1)
parser.add_argument("--viz", choices=["viser", "none"], default="viser",
                    help="viser: stream the headless physics to a browser 3D view "
                         "(Isaac's RTX renderer is broken on this driver). "
                         "none: physics only, no view.")
parser.add_argument("--viz-port", type=int, default=8080)
parser.add_argument("--sim-panel", action=argparse.BooleanOptionalAction, default=True,
                    help="show the Simulator folder in viser: live sliders for physics rate, "
                         "substeps, hand stiffness/damping/armature and the cube's contact and "
                         "linear/angular damping (isaac/sim_params.py). Untouched sliders "
                         "change nothing; every take records the values it was collected at. "
                         "--no-sim-panel locks the knobs for a data-collection run.")
parser.add_argument("--record", action="store_true",
                    help="auto-start a recording take at launch (same as pressing 's')")
parser.add_argument("--out-dir", default=os.path.join(_ROOT, "data", "teleop_takes"),
                    help="directory takes are written to (play_<timestamp>_take<n>.npz); point "
                         "`python -m play_policy.train --data_dir` at it")
parser.add_argument("--record-path", default=None,
                    help="explicit output .npz for the first take (later takes get _take<n>); "
                         "overrides --out-dir")
parser.add_argument("--record-secs", type=float, default=0.0,
                    help="end + save the take automatically after this many seconds (0 = off, "
                         "stop with 'e'). With --record this makes an unattended fixed-length "
                         "collection run: the take is saved and the script exits.")
parser.add_argument("--reset-trim", type=float, default=1.5,
                    help="seconds to drop before AND after each reset from a take")
parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True,
                    help="auto-reset when the cube is dropped; while recording this marks a "
                         "reset too, so +/-reset-trim seconds around it are trimmed on save")
parser.add_argument("--randomize-orientation", action=argparse.BooleanOptionalAction, default=True,
                    help="spawn the cube at a UNIFORM SO(3) orientation on every reset (and at "
                         "startup). On by default: the task is any-pose-to-any-pose, and always "
                         "starting from identity leaves most of SO(3) undemonstrated. Pass "
                         "--no-randomize-orientation for the pre-2026-09-05 behaviour.")
parser.add_argument("--drop-z", type=float, default=0.15,
                    help="cube center height (m) below which it counts as dropped (spawn ~0.36)")
AppLauncher.add_app_launcher_args(parser)
# NOTE on --device: the 2026-07 pool was recorded on cuda:0 and the PhysX backend is part of
# a take's identity (sim_params_json stores it), so the default stays cuda. One hand + one
# cube is latency-bound there -- ~7 ms per physics step -- which holds the loop near 40 Hz
# even with physics-only substeps. `--device cpu` runs a substep in ~0.3 ms but changes the
# solver backend: do not mix such takes into the cuda pool.
args = parser.parse_args()
# Isaac's RTX renderer crashes on this GPU/driver; always run sim headless and
# visualize via viser instead.
args.headless = True

# dex-retargeting's backend is pinocchio (boost-python / eigenpy). It MUST be built
# BEFORE the Isaac/Omniverse Kit app boots: once Kit loads its own boost libraries,
# pinocchio's C++ type registry is shadowed and the first model access fails with
#   TypeError: No Python class registered for C++ class std::vector<int, ...>
# Building it here, in the clean process context, registers pinocchio's converters up
# front; the retargeter then works fine for the whole run after Kit is up.
# --source replay takes either kind of recording: a raw glove log (manus_node_pos, run back
# through the retargeter) or a take/dataset file (cmd_q22, the joint commands themselves).
_REPLAY_TAKE = None
if args.source == "replay":
    if not args.npz or not os.path.exists(args.npz):
        sys.exit(f"[teleop] --source replay needs --npz <recording>.npz (got {args.npz!r})")
    with np.load(args.npz, allow_pickle=True) as _z:
        if "manus_node_pos" not in _z.files:
            if "cmd_q22" not in _z.files:
                sys.exit(f"[teleop] {args.npz} has neither manus_node_pos nor cmd_q22")
            _REPLAY_TAKE = {k: _z[k] for k in ("cmd_q22", "cmd_q22_names", "cube_state",
                                                "hand_joint_pos", "hand_joint_names")
                            if k in _z.files}
            print(f"[teleop] replaying the cmd_q22 joint commands of {args.npz} "
                  f"({len(_REPLAY_TAKE['cmd_q22'])} frames)")


def _wait_for_glove():
    """Connect to the glove server and wait for a NON-EMPTY skeleton, before Isaac boots.

    Two failure modes look alike from the sim (a hand that never moves), so they are told
    apart here: nothing at all on the port = no server; messages with an empty skeleton =
    the server runs but the MANUS SDK has no license (it lives on the glove's USB dongle)."""
    from manus_client import ManusGlove
    glove = ManusGlove(host=args.manus_host, pub_port=args.manus_port, sides=[args.side])
    glove.connect()
    t0, data = time.time(), None
    while time.time() - t0 < args.glove_timeout:
        data = glove.get_data(args.side)
        if data is not None and data.raw_nodes:
            print(f"[teleop] glove streaming: {len(data.raw_nodes)} skeleton nodes")
            return glove
        time.sleep(0.1)
    glove.disconnect()
    where = f"tcp://{args.manus_host}:{args.manus_port}"
    if data is None:
        sys.exit(f"[teleop] no {args.side} glove data on {where} after {args.glove_timeout:g} s. "
                 "Start the glove server first (manus/server.sh -- scripts/teleop.sh does it for "
                 "you) and check that the glove AND its license dongle are plugged in.")
    sys.exit(f"[teleop] the glove server on {where} publishes, but the skeleton is EMPTY: the MANUS "
             "SDK found no license. Plug in the glove's license dongle (the license lives on the "
             "dongle, not the glove) and restart the server. See docs/TELEOP.md.")


_PREBUILT_RETARGETER = None
_GLOVE = _wait_for_glove() if args.source == "live" else None
if args.source == "live" or (args.source == "replay" and _REPLAY_TAKE is None):
    import sharpa_retarget as _sr_prebuild
    _PREBUILT_RETARGETER = _sr_prebuild.ShapeRetargeter(args.side)
    print(f"[teleop] pre-built retargeter before Isaac boot (source={args.source})")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- sim app live below ----
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import (  # noqa: E402
    Articulation, ArticulationCfg, RigidObject, RigidObjectCfg)
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, _HERE)                       # sim_params (Isaac-only, imported post-boot)
from sharpa_retarget import get_urdf_joint_names  # noqa: E402
from sim_params import (PHYS_BASE_HZ, PHYSICS_DT, SUBSTEPS, SimParamPanel,  # noqa: E402
                        cube_physics_material, params_hash, params_json, read_sim_params)


def _quat_wxyz_from_rpy_deg(rpy):
    q = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()  # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _apply_rubik_texture(cube_prim_path, tex_path):
    """Best-effort: bind a UsdPreviewSurface with the Rubik sticker texture."""
    try:
        import omni.usd
        from pxr import Sdf, UsdShade
        stage = omni.usd.get_context().get_stage()
        mat_path = f"{cube_prim_path}/RubikMat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        tex = UsdShade.Shader.Define(stage, f"{mat_path}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).Set((1.0, 1.0))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = stage.GetPrimAtPath(cube_prim_path)
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        print("[viz] bound Rubik texture to cube")
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] Rubik texture bind skipped ({exc}); using flat color")


def build_scene():
    # Finer dt + stabilization keep the in-hand cube from skittering/popping; the
    # global bounce threshold suppresses tiny restitution jitters at rest.
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT, device=args.device,
        physx=sim_utils.PhysxCfg(
            enable_stabilization=True, bounce_threshold_velocity=0.2),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.5, 0.5, 0.6), target=(0.0, 0.0, 0.25))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    if not os.path.exists(args.usd):
        raise FileNotFoundError(
            f"Sharpa USD not found: {args.usd}\nRun convert_sharpa_usd.py first.")

    robot_cfg = ArticulationCfg(
        prim_path="/World/Sharpa",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(args.hand_pos),
            rot=_quat_wxyz_from_rpy_deg(args.hand_rpy),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=args.stiffness,
                damping=args.damping, effort_limit_sim=2.0, velocity_limit_sim=10.0),
        },
    )
    robot = Articulation(robot_cfg)

    h = args.cube_size
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(h, h, h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Cap the depenetration speed so PhysX nudges (not launches) the cube
                # out of finger overlap; more solver iters resolve the grasp contacts;
                # a little damping bleeds off energy so it settles instead of skitters.
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.005, stabilization_threshold=0.0025),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=cube_physics_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.85, 0.85), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(args.hand_pos[0] + args.cube_offset[0],
                 args.hand_pos[1] + args.cube_offset[1],
                 args.hand_pos[2] + args.cube_offset[2])),
    )
    cube = RigidObject(cube_cfg)
    return sim, robot, cube


# Standard ("Western") Rubik's cube faces, opposite-pairs match a real cube:
# white/yellow, red/orange, green/blue -- the same scheme OpenAI's Dactyl cube uses.
_RUBIK_FACE_FILES = ["rubik_face_white.png", "rubik_face_yellow.png",
                     "rubik_face_red.png", "rubik_face_orange.png",
                     "rubik_face_green.png", "rubik_face_blue.png"]


def build_rubik_mesh(size):
    """A textured box (edge=`size` m) with each face UV-mapped to a 3x3 sticker image.

    viser renders this scene (Isaac's RTX is headless here) and add_box is flat-color
    only, so we hand the real Rubik look to viser as a textured trimesh instead.
    """
    import trimesh
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    imgs = [Image.open(os.path.join(_HERE, "assets", f)).convert("RGB")
            for f in _RUBIK_FACE_FILES]
    tile = imgs[0].width
    atlas = Image.new("RGB", (tile * 6, tile))
    for i, im in enumerate(imgs):
        atlas.paste(im.resize((tile, tile)), (i * tile, 0))

    h = size / 2.0
    # 4 outward-CCW corners per face: +X, -X, +Y, -Y, +Z, -Z
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
    mesh = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64),
                           faces=np.array(tris), process=False)
    mesh.visual = TextureVisuals(uv=np.array(uvs), material=SimpleMaterial(image=atlas))
    return mesh


# ── viser visualization (replaces Isaac's broken RTX renderer) ────────────────
def setup_viser(port, cube_size, hand_pos, hand_wxyz):
    from pathlib import Path

    import viser
    from viser.extras import ViserUrdf
    srv = viser.ViserServer(host="0.0.0.0", port=port)
    # fixed hand mount frame, so viser world == Isaac world
    srv.scene.add_frame("/hand", position=tuple(float(x) for x in hand_pos),
                        wxyz=tuple(float(x) for x in hand_wxyz), show_axes=False)
    urdf_path = os.path.join(_ROOT, "assets", "urdf", "sharpa", "right_fixed.urdf")
    # viser's ViserUrdf takes a yourdfpy.URDF or a pathlib.Path -- NOT a str
    # (a str fails its `isinstance(urdf, yourdfpy.URDF)` assertion).
    vurdf = ViserUrdf(srv, Path(urdf_path), root_node_name="/hand")
    cube = srv.scene.add_mesh_trimesh("/cube", build_rubik_mesh(cube_size))
    srv.scene.add_grid("/floor", width=1.0, height=1.0, cell_size=0.05)
    print(f"[viz] viser 3D view -> http://0.0.0.0:{port}")
    return srv, vurdf, cube


def update_viser(vurdf, cube, art_joint_names, joint_pos, cube_pos, cube_wxyz):
    vurdf.update_cfg({n: float(joint_pos[i]) for i, n in enumerate(art_joint_names)})
    cube.position = tuple(float(x) for x in cube_pos)
    cube.wxyz = tuple(float(x) for x in cube_wxyz)


# ── reset + key handling ──────────────────────────────────────────────────────
def _rand_quat_wxyz(rng):
    """Uniform random unit quaternion (Shoemake). Isotropic, so the component assignment is
    arbitrary."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1.0 - u1), math.sqrt(u1)
    return np.array([a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2),
                     b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)], np.float64)


def reset_sim(robot, cube, default_joint_pos, rand_rng=None):
    """Drop the cube back at its spawn pose (zero velocity) and restore the hand.

    With ``rand_rng`` the cube gets a fresh uniform-SO(3) orientation instead of the spawn one,
    so consecutive takes demonstrate reorientation from all over the group rather than from a
    single pose."""
    root = cube.data.default_root_state.clone()
    pose = root[:, :7].clone()
    if rand_rng is not None:
        pose[:, 3:7] = torch.as_tensor(_rand_quat_wxyz(rand_rng), dtype=pose.dtype,
                                       device=pose.device)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    cube.reset()
    robot.write_joint_state_to_sim(
        default_joint_pos.clone(), torch.zeros_like(default_joint_pos))
    robot.reset()
    print("[teleop] reset: cube + hand restored to spawn pose")


class TerminalKeyPoller:
    """Non-blocking single-key reader on the launching terminal (cbreak mode)."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.fd = None
        self.old = None
        if self.enabled:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def poll(self):
        if not self.enabled:
            return None
        import select
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if dr else None

    def restore(self):
        if self.enabled and self.old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class Recorder:
    """Take-based log of hand joints + cube state, saved to a compressed .npz.

    Flow: 's' starts a take, 'e' ends it and writes a file; each pressed 'r' during
    a take marks a reset time, and on save every frame within +/-`trim` seconds of a
    reset is dropped (the cube teleports on reset, so that window is unusable).

    Per frame it stores `t` (s, relative to take start), `hand_joint_pos`
    (articulation order), `cmd_q22` (commanded retarget angles) and `cube_state`
    (13 = pos[3] + quat_wxyz[4] + lin_vel[3] + ang_vel[3]). `meta` holds the
    joint-name orders, fixed hand pose and run settings so a take is self-describing.

    `params_fn` (the Simulator panel) makes that self-description cover the PHYSICS too:
    every take saves `sim_params_json`, and `sim_param_changes` counts knobs moved while
    it was rolling -- a take with a nonzero count mixes dynamics and should not be trusted
    as single-config training data.
    """

    def __init__(self, out_dir, base_path, meta, trim=1.5, params_fn=None):
        self.out_dir = out_dir
        self.base_path = base_path
        self.meta = meta
        self.trim = trim
        self.params_fn = params_fn
        self.frames = []
        self.reset_times = []
        self.active = False
        self.take = 0
        self.param_changes = 0
        self.started_t = None

    def start(self):
        if self.active:
            print("[record] ['s'] already recording -- ignored")
            return
        self.frames = []
        self.reset_times = []
        self.param_changes = 0
        self.started_t = None          # stamped by the run loop on the take's first frame
        self.active = True
        print("[record] ['s'] take started")

    def note_param_change(self, _params=None):
        """A simulator knob moved -- harmless between takes, contaminating inside one."""
        if self.active:
            self.param_changes += 1
            print("[record] WARNING: simulator parameter changed mid-take -- this take now "
                  "mixes dynamics (recorded as sim_param_changes)")

    def mark_reset(self, t):
        if self.active:
            self.reset_times.append(float(t))
            print(f"[record] reset @ {t:.2f}s -> trimming +/-{self.trim}s around it")

    def capture(self, **arrays):
        if self.active:
            self.frames.append(arrays)

    def stop_and_save(self):
        if not self.active:
            print("[record] ['e'] not recording -- ignored")
            return
        self.active = False
        self._save()

    def finish(self):
        """Flush a still-running take on exit (Ctrl-C / window close)."""
        if self.active:
            self.active = False
            self._save()

    def _keep(self, t):
        return all(abs(t - rt) > self.trim for rt in self.reset_times)

    def _next_path(self):
        self.take += 1
        if self.base_path:
            stem, ext = os.path.splitext(self.base_path)
            return self.base_path if self.take == 1 else f"{stem}_take{self.take}{ext}"
        return os.path.join(
            self.out_dir, time.strftime(f"play_%Y%m%d_%H%M%S_take{self.take}.npz"))

    def _save(self):
        kept = [f for f in self.frames if self._keep(float(f["t"]))]
        dropped = len(self.frames) - len(kept)
        if not kept:
            print(f"[record] ['e'] nothing to save -- all {len(self.frames)} frames "
                  f"fell within reset windows")
            self.frames, self.reset_times = [], []
            return
        path = self._next_path()
        data = {k: np.asarray([f[k] for f in kept]) for k in kept[0]}
        data.update(self.meta)
        data["reset_times"] = np.asarray(self.reset_times, dtype=np.float32)
        data["trim_s"] = np.array(self.trim)
        # the rate the loop ACTUALLY ran at (the nominal --hz is in meta["hz"]); per-frame human
        # motion scales with it, so a take that drifted from CONTROL_HZ is visible here
        dts = np.diff(np.asarray(data["t"], np.float64))
        dts = dts[(dts > 0) & (dts < 0.5)]
        data["hz_measured"] = np.array(1.0 / float(np.median(dts)) if len(dts) else 0.0, dtype=np.float32)
        if self.params_fn is not None:          # the simulator this take was collected in
            p = self.params_fn()
            data["sim_params_json"] = np.array(params_json(p))
            data["sim_params_hash"] = np.array(params_hash(p))
            data["sim_param_changes"] = np.array(self.param_changes)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"[record] ['e'] saved {len(kept)} frames "
              f"(trimmed {dropped} around {len(self.reset_times)} reset(s)), "
              f"loop {float(data['hz_measured']):.1f} Hz measured -> {path}")
        self.frames, self.reset_times = [], []


# ── Action sources ────────────────────────────────────────────────────────────
def make_source():
    """Return step() -> (22,) Sharpa joint angles in get_urdf_joint_names order."""
    names = get_urdf_joint_names(args.side)
    if args.source == "demo":
        lo = np.zeros(22)
        t0 = time.time()

        def step():
            # slow open/close: flex MCP/PIP/DIP-ish joints together
            phase = 0.5 - 0.5 * math.cos(2 * math.pi * (time.time() - t0) / 4.0)
            q = np.zeros(22, dtype=np.float32)
            for i, n in enumerate(names):
                if any(k in n for k in ("MCP_FE", "PIP", "DIP", "thumb_MCP_FE")):
                    q[i] = phase * 1.2
            return q
        return step

    if _REPLAY_TAKE is not None:
        # a take already holds the retargeted commands: map its cmd_q22 columns by name
        rec = _REPLAY_TAKE["cmd_q22"].astype(np.float32)
        rec_names = [str(n) for n in _REPLAY_TAKE.get("cmd_q22_names", names)]
        cols = [rec_names.index(n) for n in names]
        state = {"i": 0}

        def step():
            q = rec[state["i"] % len(rec), cols]
            state["i"] += 1
            return q
        return step

    import sharpa_retarget as sr
    sr.set_abduction_fix(args.ab_correct_deg)
    # Reuse the retargeter built before Isaac booted (see the pre-build note above);
    # rebuilding here, inside the Kit process, would hit the pinocchio/boost conflict.
    retargeter = (_PREBUILT_RETARGETER if _PREBUILT_RETARGETER is not None
                  else sr.ShapeRetargeter(args.side))
    scale = sr.make_scale_fn(retargeter, master=args.scaling,
                             finger_scales=_CFG["finger_scales"])
    coupling = sr.build_distal_coupling(names)
    thumb_tracker = (sr.make_thumb_tracker(names, str(sr._SHARPA_URDF), _CFG)
                     if args.thumb_mode == "joint" else None)
    print(f"[params] from {_rp.PARAMS_PATH}: scaling={args.scaling} "
          f"finger_scales={_CFG['finger_scales']} dip_coupling={args.dip_coupling} "
          f"ab_correct_deg={args.ab_correct_deg} thumb_mode={args.thumb_mode}")

    if args.source == "live":
        glove = _GLOVE                           # connected + checked before Isaac booted

        def step():
            data = glove.get_data(args.side)
            if data is None or not data.raw_nodes:
                return retargeter._last_qpos.copy()
            sk = sr.normalize_hand_positions(data, side=args.side)
            q = retargeter.retarget(data, scale(sk))
            q = sr.apply_distal_coupling(q, coupling, args.dip_coupling)
            return sr.apply_thumb_joint(q, data, thumb_tracker,
                                        args.thumb_joint_weight)
        return step

    # replay
    data = np.load(args.npz, allow_pickle=True)["manus_node_pos"]
    T = len(data)
    state = {"i": 0}

    def step():
        gd = sr.glove_data_from_nodes(np.asarray(data[state["i"] % T]))
        sk = sr.normalize_hand_positions(gd, side=args.side)
        q = retargeter.retarget(gd, scale(sk))
        state["i"] += 1
        q = sr.apply_distal_coupling(q, coupling, args.dip_coupling)
        return sr.apply_thumb_joint(q, gd, thumb_tracker, args.thumb_joint_weight)
    return step


def _seed_from_take(robot, cube, art_names):
    """Start a take replay from the take's own first frame (cube pose + hand joints).

    Only the start is matched: the commands are then re-simulated open loop, so the cube
    drifts from the recording within ~a second (contact chaos). For a frame-exact viewer use
    scripts/replay.sh; to re-simulate with live physics sliders, scripts/replay.sh --sim."""
    cs = torch.as_tensor(_REPLAY_TAKE["cube_state"][0, :7], dtype=torch.float32,
                         device=cube.data.root_state_w.device)[None]
    cube.write_root_pose_to_sim(cs)
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=cs.device))
    cube.reset()
    if "hand_joint_pos" in _REPLAY_TAKE and "hand_joint_names" in _REPLAY_TAKE:
        rec_names = [str(n) for n in _REPLAY_TAKE["hand_joint_names"]]
        jp = robot.data.default_joint_pos.clone()
        for j, n in enumerate(art_names):
            if n in rec_names:
                jp[0, j] = float(_REPLAY_TAKE["hand_joint_pos"][0, rec_names.index(n)])
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.reset()
    print("[teleop] replay starts from the take's first frame (cube pose + hand joints)")


def main():
    sim, robot, cube = build_scene()
    sim.reset()
    _apply_rubik_texture("/World/Cube", _RUBIK_TEX)
    print(f"[teleop] articulation joints ({robot.num_joints}): {list(robot.joint_names)}")

    # map our 22-joint order -> articulation joint order
    our_names = get_urdf_joint_names(args.side)
    art_names = list(robot.joint_names)
    col = [art_names.index(n) for n in our_names if n in art_names]
    our_idx = [k for k, n in enumerate(our_names) if n in art_names]
    if len(col) != len(our_names):
        missing = [n for n in our_names if n not in art_names]
        print(f"[warn] joints not found in articulation: {missing}")

    recorder = Recorder(
        out_dir=args.out_dir,
        base_path=args.record_path,
        trim=args.reset_trim,
        meta={
            "source": np.array(args.source),
            "hz": np.array(args.hz),
            "randomize_orientation": np.array(bool(args.randomize_orientation)),
            "cube_size": np.array(args.cube_size),
            "hand_joint_names": np.array(art_names),
            "cmd_q22_names": np.array(our_names),
            "hand_root_pos": np.array(args.hand_pos, dtype=np.float32),
            "hand_root_quat": np.array(
                _quat_wxyz_from_rpy_deg(args.hand_rpy), dtype=np.float32),
        })
    if args.record:
        recorder.start()

    viser_state = None
    panel = None
    reset_flag = {"v": False}
    if args.viz == "viser":
        viser_state = setup_viser(args.viz_port, args.cube_size,
                                  args.hand_pos, _quat_wxyz_from_rpy_deg(args.hand_rpy))
        if args.sim_panel:
            # Simulator knobs on sliders (shared with isaac/viser_sim_replay.py). It seeds
            # itself from the running scene, so an untouched panel changes no dynamics; the
            # rate/gain knobs land on the next step, the cube's USD properties cost a ~0.3 s
            # timeline bounce that carries the current hand + cube state across.
            panel = SimParamPanel(viser_state[0], sim, robot, cube,
                                  overrides={"phys_hz": PHYS_BASE_HZ,
                                             "substeps": SUBSTEPS})
            panel.on_change(recorder.note_param_change)
        reset_btn = viser_state[0].gui.add_button("Reset (r)")

        @reset_btn.on_click
        def _on_reset(_):  # noqa: ANN001
            reset_flag["v"] = True

        start_btn = viser_state[0].gui.add_button("Start rec (s)")

        @start_btn.on_click
        def _on_start(_):  # noqa: ANN001
            recorder.start()

        end_btn = viser_state[0].gui.add_button("End rec (e)")

        @end_btn.on_click
        def _on_end(_):  # noqa: ANN001
            recorder.stop_and_save()

    # Every take stamps the physics it was collected under, read from the RUNNING sim -- so
    # it is right whether or not the panel exists (--no-sim-panel is exactly the collection
    # run that most needs the provenance) and whether or not a knob was moved since launch.
    recorder.params_fn = lambda: read_sim_params(
        sim, robot, cube, steps_per_frame=(panel.substeps if panel is not None else SUBSTEPS),
        control_hz=args.hz, extra={"source": args.source, "cube_size": args.cube_size})

    step_source = make_source()
    dt = 1.0 / max(args.hz, 1.0)
    default_targets = robot.data.default_joint_pos.clone()
    rand_rng = np.random.default_rng() if args.randomize_orientation else None
    if rand_rng is not None:
        # the first episode should be random too, not just the ones after a reset
        reset_sim(robot, cube, default_targets, rand_rng)
        for _ in range(SUBSTEPS * 30):          # let it settle into the hand before recording
            sim.step(render=False)
        robot.update(1.0 / max(args.hz, 1.0)); cube.update(1.0 / max(args.hz, 1.0))
        print("[teleop] cube spawns at a UNIFORM SO(3) orientation on every reset "
              "(--no-randomize-orientation to disable)")
    if _REPLAY_TAKE is not None and "cube_state" in _REPLAY_TAKE:
        _seed_from_take(robot, cube, art_names)
    keys = TerminalKeyPoller()
    print(f"[teleop] source={args.source}; running. Keys (terminal) / viser buttons: "
          "'s' start take, 'e' end+save, 'r' reset; Ctrl-C to stop.")
    if panel is not None:
        print("[teleop] simulator sliders live in the viser 'Simulator' folder "
              f"(physics {panel.params['phys_hz']:.0f}Hz x{panel.substeps}, "
              f"stiffness {panel.params['stiffness']:.3g}, damping {panel.params['damping']:.3g})")
    try:
        _run_loop(sim, robot, cube, step_source, dt, default_targets, viser_state,
                  reset_flag, keys, art_names, col, our_idx, recorder, panel, rand_rng)
    finally:
        keys.restore()
        recorder.finish()


def _run_loop(sim, robot, cube, step_source, dt, default_targets, viser_state,
              reset_flag, keys, art_names, col, our_idx, recorder, panel=None, rand_rng=None):
    nframe = 0
    drop_cooldown = max(1, int(0.5 / dt))   # frames to wait after an auto-reset before re-checking
    last_drop_frame = -10 ** 9
    t_start = time.time()
    work_ms, t_stat = [], time.time()      # per-frame work time -> printed every 120 frames
    while simulation_app.is_running():
        t0 = time.time()
        t_rel = t0 - t_start
        k = keys.poll()
        if k:
            kl = k.lower()
            if kl == "r":
                print("[key] 'r' pressed -> reset (cube + hand)")
                reset_flag["v"] = True
            elif kl == "s":
                print("[key] 's' pressed -> start recording take")
                recorder.start()
            elif kl == "e":
                print("[key] 'e' pressed -> end + save recording take")
                recorder.stop_and_save()
        if panel is not None and panel.poll():
            # the panel bounced the timeline to load new cube properties: the physics views
            # were rebuilt, so re-take the default pose the targets are built from
            default_targets = robot.data.default_joint_pos.clone()
        if reset_flag["v"]:
            reset_sim(robot, cube, default_targets, rand_rng)
            recorder.mark_reset(t_rel)
            reset_flag["v"] = False
        q22 = step_source()                      # (22,) radians
        targets = default_targets.clone()
        targets[0, col] = torch.as_tensor(
            q22[our_idx], dtype=targets.dtype, device=targets.device)
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        for _ in range(panel.substeps if panel is not None else SUBSTEPS):
            sim.step(render=False)   # physics only: same PhysX step, minus a full Kit update per call
        robot.update(dt)
        cube.update(dt)
        # Auto-reset on a dropped cube. Going through reset_flag means the next loop iteration
        # runs the same reset path as 'r' -- reset_sim + recorder.mark_reset -- so while recording
        # the +/-reset-trim window around the drop is trimmed on save.
        if (args.auto_reset and not reset_flag["v"]
                and nframe - last_drop_frame > drop_cooldown):
            cube_z = float(cube.data.root_pos_w[0, 2].item())
            if cube_z < args.drop_z:
                reset_flag["v"] = True
                last_drop_frame = nframe
                print(f"[teleop] cube dropped (z={cube_z:.3f} < {args.drop_z:.2f}) -> auto-reset"
                      + ("  (marking trimmed reset)" if recorder.active else ""))
        recorder.capture(
            t=np.float32(t_rel),
            hand_joint_pos=robot.data.joint_pos[0].cpu().numpy().astype(np.float32),
            cmd_q22=np.asarray(q22, dtype=np.float32),
            cube_state=cube.data.root_state_w[0].cpu().numpy().astype(np.float32))
        if viser_state is not None:
            update_viser(viser_state[1], viser_state[2], art_names,
                         robot.data.joint_pos[0].cpu().numpy(),
                         cube.data.root_pos_w[0].cpu().numpy(),
                         cube.data.root_quat_w[0].cpu().numpy())
        # fixed-length take: end it (and, if it was auto-started, the run) on the clock
        if recorder.active and recorder.started_t is None:
            recorder.started_t = t_rel
        if (args.record_secs > 0 and recorder.active
                and t_rel - recorder.started_t >= args.record_secs):
            print(f"[record] --record-secs {args.record_secs:g} reached -> ending take")
            recorder.stop_and_save()
            if args.record:
                break
        nframe += 1
        work_ms.append((time.time() - t0) * 1e3)          # work this frame, before the pacing sleep
        if nframe % 120 == 0:
            cp = (cube.data.root_pos_w[0] - robot.data.root_pos_w[0]).cpu().numpy()
            w = np.asarray(work_ms); loop_hz = len(w) / max(time.time() - t_stat, 1e-6)
            print(f"[teleop] frame {nframe}  cube_rel_hand={np.round(cp, 3)}  "
                  f"loop {loop_hz:.1f} Hz  work {np.mean(w):.1f} ms mean / {np.percentile(w, 90):.1f} p90 "
                  f"/ {w.max():.1f} max  (budget {dt * 1e3:.1f} ms)")
            work_ms.clear(); t_stat = time.time()
        rest = dt - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)


if __name__ == "__main__":
    code = 0
    try:
        main()                     # main() saves a still-running take on the way out
    except KeyboardInterrupt:
        pass
    except Exception:              # noqa: BLE001
        traceback.print_exc()
        code = 1
    # Kit can hang inside simulation_app.close() -- exit hard once the take is on disk
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
