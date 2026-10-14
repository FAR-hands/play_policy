#!/usr/bin/env python3
"""Live-physics replay: re-simulate a recorded rollout with the SIMULATOR on sliders.

``viser_replay_server.py`` plays the recorded poses back frame by frame -- nothing is
computed, so a physics knob would have nothing to act on. This tool instead boots the
Isaac Lab scene the data was collected in (Sharpa hand + cube, same spawn / gains as
teleop_isaac + the self-play collector) and RE-EXECUTES the recording's ``cmd_q22`` joint targets
one step at a time. The hand follows the recorded commands; the cube's motion is
simulated, so every physics parameter is visible in the browser and adjustable live:

  Physics rate (Hz) x Substeps  ->  physics dt = 1/(rate*substeps)        [live]
  Hand joint stiffness / damping / armature (implicit-actuator PD gains)  [live]
  Cube contact stiffness / damping (PhysX compliant contact),
  cube linear / angular damping                                          [sim restart]

Live knobs go straight into PhysX (``set_physics_dt`` / ``write_joint_*_to_sim``). The
cube's material and body properties live in USD, so those are applied by bouncing the
timeline (``sim.reset()``, ~0.3 s, no Isaac reboot) and re-seeding the sim from the
RECORDED state at the current frame -- same start state, same actions, new physics: an
interactive A/B of a parameter change against the data.

A wireframe ghost cube marks the RECORDED pose; the readout is the geodesic angle between
simulated and recorded, i.e. how far this parameter set drifts from the data it was
recorded at. Defaults reproduce the recording config (isaac/sim_params.py: 240 Hz x 3
substeps, stiffness 3.0, damping 0.1). Drift grows even there -- a recording can't restore PhysX's solver state and
carries no joint velocities (they are finite-differenced back) -- so that curve is the
simulator's chaos floor, not a parameter effect. Use "Re-seed every N frames" to keep the
comparison in the short-horizon regime where parameters, not chaos, dominate.

Needs Isaac Sim (unlike the pure replay server): Kit takes ~30-60 s to boot before the
viser page comes up, and the replay then runs at roughly 2x real time.

  OMNI_KIT_ACCEPT_EULA=YES python isaac/viser_sim_replay.py --rollout <take>.npz --port 8080
  OMNI_KIT_ACCEPT_EULA=YES python isaac/viser_sim_replay.py --selftest   # no browser

or just ``scripts/replay.sh --sim`` (same flags, sets the EULA variable for you).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)      # viser_replay_server (rollout discovery + Rubik mesh)
sys.path.insert(0, _ROOT)
_USD = os.path.join(_HERE, "assets", "sharpa_right", "sharpa_right.usd")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rollout", default="data",
                help="rollout .npz OR a directory of them (a dropdown then switches between all)")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed multiplier")
ap.add_argument("--start-frame", type=int, default=0, help="frame to seed the sim from")
ap.add_argument("--drive", choices=["cmd", "pos"], default="cmd",
                help="joint targets to replay: the recorded cmd_q22 (default, what the policy "
                     "actually sent) or the recorded achieved hand_joint_pos")
ap.add_argument("--seed-vel", choices=["fd", "zero"], default="fd",
                help="recordings store joint POSITIONS only; 'fd' reconstructs the joint "
                     "velocities a re-seed needs by finite-differencing them (zeroing them "
                     "is itself a large perturbation of a chaotic contact state)")
ap.add_argument("--record-phys-hz", type=float, default=240.0,
                help="physics rate the recording was made at -- only used to scale the "
                     "finite-differenced seed velocities")
ap.add_argument("--reseed-every", type=int, default=0,
                help="re-seed from the recording every N frames (0 = free-run). Small N turns "
                     "the drift readout into a short-horizon error, which is what actually "
                     "separates parameter sets -- long free runs are dominated by sim chaos")
ap.add_argument("--selftest", action="store_true",
                help="boot, exercise every knob (incl. the restart path), print a report, exit")
# ---- simulator defaults: the config the recordings were made at -------------------
ap.add_argument("--phys-hz", type=float, default=None, help="physics rate; ONE recorded frame "
                                                             "advances the sim by 1/phys_hz s")
ap.add_argument("--substeps", type=int, default=None, help="physics steps per recorded frame at "
                                                        "dt=1/(phys_hz*substeps): finer integration, same timing")
ap.add_argument("--stiffness", type=float, default=3.0, help="hand joint stiffness")
ap.add_argument("--damping", type=float, default=0.1, help="hand joint damping")
ap.add_argument("--armature", type=float, default=None,
                help="hand joint armature (rotor inertia); default = whatever the USD defines")
ap.add_argument("--cube-contact-stiffness", type=float, default=None,
                help="cube compliant-contact spring stiffness (0 = rigid contact)")
ap.add_argument("--cube-contact-damping", type=float, default=None,
                help="cube compliant-contact damping (only used when stiffness > 0)")
ap.add_argument("--cube-linear-damping", type=float, default=0.2)
ap.add_argument("--cube-angular-damping", type=float, default=0.2)
# ---- scene (only used when the recording doesn't carry them) ----------------------
ap.add_argument("--usd", default=_USD)
ap.add_argument("--cube-size", type=float, default=0.05)
ap.add_argument("--hand-pos", type=float, nargs=3, default=[0.0, 0.0, 0.30])
ap.add_argument("--hand-rpy", type=float, nargs=3, default=[-173.746, -59.607, 94.102])
ap.add_argument("--drop-z", type=float, default=0.15, help="cube z below this counts as dropped (m)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True                        # Isaac's RTX renderer is broken here; viser draws the scene
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils                                    # noqa: E402
import torch                                                        # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg                  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402

from sim_params import (CUBE_CONTACT_DAMPING, CUBE_CONTACT_STIFFNESS,  # noqa: E402
                        PHYS_BASE_HZ, SUBSTEPS, SimParamPanel, cube_material_prim)
from viser_replay_server import build_rubik_mesh, discover_rollouts  # noqa: E402

DEV = args.device
CUBE_PRIM = "/World/Cube"
HAND_PRIM = "/World/Sharpa"


def q_rpy(rpy):
    """xyz-euler degrees -> quaternion (w,x,y,z), as used for the hand mount."""
    q = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def geo_deg(qa, qb):
    """Geodesic angle (deg) between two wxyz quaternions."""
    d = float(np.clip(abs(float(np.dot(np.asarray(qa, float), np.asarray(qb, float)))), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(d)))


def cube_edges(size):
    """(12,2,3) local-frame line segments outlining a cube -- the 'recorded' ghost."""
    h = size / 2.0
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], np.float32)
    idx = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
           (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return np.stack([np.stack([v[a], v[b]]) for a, b in idx]).astype(np.float32)


def load_rollout(path):
    """Load the fields the sim replay needs (a superset of the pure replay server's)."""
    d = np.load(path, allow_pickle=True)
    files = set(d.files)
    hand = np.asarray(d["hand_joint_pos"], np.float32)              # (T,22) achieved
    hand_names = [str(n) for n in d["hand_joint_names"]]
    cube = np.asarray(d["cube_state"], np.float32)                  # (T,>=7) pos3+quat4(+vel6)
    T = len(hand)
    if "cmd_q22" in files:
        cmd = np.asarray(d["cmd_q22"], np.float32)                  # (T,22) commanded targets
        cmd_names = [str(n) for n in d["cmd_q22_names"]]
    else:                                                           # fall back to achieved poses
        cmd, cmd_names = hand, hand_names
    if "t" in files:
        tarr = np.asarray(d["t"], float).ravel()
    elif "hz" in files:
        tarr = np.arange(T) / max(float(d["hz"]), 1e-6)
    else:
        tarr = np.arange(T) / 60.0
    hz = float(d["hz"]) if "hz" in files else None
    if hz is None and T > 1:                                        # MPC rollouts: infer from t
        dts = np.diff(tarr)
        dts = dts[(dts > 0) & np.isfinite(dts)]
        hz = 1.0 / float(np.median(dts)) if len(dts) else 60.0
    return dict(
        hand=hand, hand_names=hand_names, cmd=cmd, cmd_names=cmd_names, cube=cube, T=T,
        tarr=tarr, hz=float(hz or 60.0), path=path,
        episode=np.asarray(d["episode"]).ravel().astype(int) if "episode" in files else None,
        cube_size=float(d["cube_size"]) if "cube_size" in files else args.cube_size,
        hroot_pos=(np.asarray(d["hand_root_pos"], float) if "hand_root_pos" in files
                   else np.asarray(args.hand_pos, float)),
        hroot_quat=(np.asarray(d["hand_root_quat"], float) if "hand_root_quat" in files
                    else np.asarray(q_rpy(args.hand_rpy), float)),
    )


def build_scene(P, hand_pos, hand_quat, cube_size):
    """Sharpa hand + cube, identical to teleop_isaac.build_scene apart from the knobs."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / (P["phys_hz"] * P["substeps"]), device=DEV,
        physx=sim_utils.PhysxCfg(enable_stabilization=True, bounce_threshold_velocity=0.2)))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))
    if not os.path.exists(args.usd):
        raise FileNotFoundError(f"Sharpa USD not found: {args.usd}\nRun convert_sharpa_usd.py first.")
    robot = Articulation(ArticulationCfg(
        prim_path=HAND_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                                                                      enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(float(x) for x in hand_pos),
                                                   rot=tuple(float(x) for x in hand_quat),
                                                   joint_pos={".*": 0.0}),
        actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=P["stiffness"],
                                                  damping=P["damping"], effort_limit_sim=2.0,
                                                  velocity_limit_sim=10.0)}))
    cube = RigidObject(RigidObjectCfg(
        prim_path=CUBE_PRIM,
        spawn=sim_utils.CuboidCfg(
            size=(cube_size,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4, linear_damping=P["cube_lin_damp"],
                angular_damping=P["cube_ang_damp"], sleep_threshold=0.005,
                stabilization_threshold=0.0025),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
                restitution_combine_mode="min",
                compliant_contact_stiffness=P["cube_stiff"],
                compliant_contact_damping=P["cube_damp"])),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5))))
    return sim, robot, cube


def main() -> int:
    rollouts = discover_rollouts(args.rollout)
    if not rollouts:
        raise SystemExit(f"[simreplay] no rollout .npz (with hand_joint_pos) under: {args.rollout}")
    labels = [lab for lab, _ in rollouts]
    by_label = dict(rollouts)
    data = load_rollout(rollouts[0][1])
    print(f"[simreplay] {len(rollouts)} rollout(s) from {args.rollout}; first "
          f"{os.path.basename(data['path'])}: {data['T']} frames @ {data['hz']:.0f}Hz, "
          f"driving with {'cmd_q22' if args.drive == 'cmd' else 'hand_joint_pos'}")

    # ---- simulator parameters: spawn values here, then owned by the shared panel ----
    # Unset flags fall back to the project defaults in isaac/sim_params.py, so a replay
    # reproduces the simulator the data is collected in without repeating the numbers.
    args.phys_hz = PHYS_BASE_HZ if args.phys_hz is None else args.phys_hz
    args.substeps = SUBSTEPS if args.substeps is None else args.substeps
    if args.cube_contact_stiffness is None:
        args.cube_contact_stiffness = CUBE_CONTACT_STIFFNESS
    if args.cube_contact_damping is None:
        args.cube_contact_damping = CUBE_CONTACT_DAMPING
    P = dict(phys_hz=float(args.phys_hz), substeps=int(args.substeps),
             stiffness=float(args.stiffness), damping=float(args.damping),
             cube_stiff=float(args.cube_contact_stiffness),
             cube_damp=float(args.cube_contact_damping),
             cube_lin_damp=float(args.cube_linear_damping),
             cube_ang_damp=float(args.cube_angular_damping))

    sim, robot, cube = build_scene(P, data["hroot_pos"], data["hroot_quat"], data["cube_size"])
    sim.reset()
    art_names = list(robot.joint_names)
    nj = len(art_names)
    print(f"[simreplay] articulation: {nj} joints, physics dt {sim.get_physics_dt() * 1e3:.3f} ms")

    st = {"frame": int(np.clip(args.start_frame, 0, data["T"] - 1)), "playing": True,
          "speed": float(args.speed), "loop": True, "max_speed": False, "data": data,
          "drive": args.drive, "reseed_on_episode": True,
          "reseed_every": int(max(args.reseed_every, 0)), "since_seed": 0,
          "cmd_t": None, "hand_t": None, "cube_t": None, "act_col": None, "hand_col": None,
          "rtf": 0.0, "steps_per_s": 0.0, "div": 0.0, "dpos": 0.0, "dropped": False}
    cmds = deque()                      # GUI thread -> main loop (deque ops are atomic)

    def bind_rollout(d):
        """Cache the recording on the GPU + the name maps into articulation columns."""
        st["data"] = d
        st["hand_col"] = torch.as_tensor(np.array([art_names.index(n) for n in d["hand_names"]]),
                                         device=DEV, dtype=torch.long)
        st["act_col"] = torch.as_tensor(np.array([art_names.index(n) for n in d["cmd_names"]]),
                                        device=DEV, dtype=torch.long)
        st["hand_t"] = torch.as_tensor(d["hand"], device=DEV)
        st["cmd_t"] = torch.as_tensor(d["cmd"], device=DEV)
        st["cube_t"] = torch.as_tensor(d["cube"], device=DEV)

    bind_rollout(data)
    default_q = robot.data.default_joint_pos.clone()

    def seed(frame):
        """Put the sim at the recorded state of `frame`: hand pose (+ reconstructed joint
        velocity), cube pose and cube velocity.

        The recordings carry cube velocities but only joint POSITIONS, so the joint velocity
        is finite-differenced from the previous frame (one physics step at the recording's
        rate). It is not a bit-exact state copy either way -- PhysX's solver/contact state
        can't be restored -- so expect the drift readout to grow even at the recorded
        parameters; that growth IS the simulator's chaos floor."""
        d = st["data"]
        f = int(np.clip(frame, 0, d["T"] - 1))
        q = default_q.clone()
        q[0, st["hand_col"]] = st["hand_t"][f]
        qd = torch.zeros_like(q)
        if args.seed_vel == "fd" and f > 0:
            qd[0, st["hand_col"]] = (st["hand_t"][f] - st["hand_t"][f - 1]) * args.record_phys_hz
        robot.write_joint_state_to_sim(q, qd)
        tgt = default_q.clone()
        if st["drive"] == "cmd":
            tgt[0, st["act_col"]] = st["cmd_t"][f]
        else:
            tgt[0, st["hand_col"]] = st["hand_t"][f]
        robot.set_joint_position_target(tgt)
        robot.reset()
        cs = st["cube_t"][f]
        cube.write_root_pose_to_sim(cs[None, :7])
        vel = cs[None, 7:13] if cs.shape[0] >= 13 else torch.zeros(1, 6, device=DEV)
        cube.write_root_velocity_to_sim(vel)
        cube.reset()
        st["frame"] = f
        st["div"] = st["dpos"] = 0.0
        st["since_seed"] = 0
        st["dropped"] = False

    def after_restart():
        """Called by the panel once PhysX has re-parsed the stage: this tool has a better
        anchor than the panel's generic state snapshot -- the recording itself."""
        nonlocal default_q
        default_q = robot.data.default_joint_pos.clone()   # views are rebuilt by the bounce
        seed(st["frame"])

    # The panel owns every simulator knob (isaac/sim_params.py, shared with teleop).
    # phys_hz/substeps are passed explicitly because the scene's dt alone can't tell them
    # apart; the rest it reads back from the live scene, so it starts as a no-op.
    panel = SimParamPanel(
        None, sim, robot, cube, cube_prim=CUBE_PRIM,
        overrides={"phys_hz": args.phys_hz, "substeps": args.substeps,
                   "armature": args.armature},
        after_restart=after_restart, reset_label="Reset to recording defaults",
        baseline_note="matches the recording")
    P = panel.params

    def advance():
        """Execute the next recorded action for one frame (substeps physics steps)."""
        d = st["data"]
        f = st["frame"] + 1
        if f >= d["T"]:
            return False
        ep = d["episode"]
        if st["reseed_on_episode"] and ep is not None and ep[f] != ep[st["frame"]]:
            seed(f)                                    # recorded reset -> re-seed, don't simulate it
            return True
        if st["reseed_every"] > 0 and st["since_seed"] >= st["reseed_every"]:
            seed(f)                                    # short-horizon protocol: re-anchor to the data
            return True
        src = st["cmd_t"] if st["drive"] == "cmd" else st["hand_t"]
        tgt = default_q.clone()
        tgt[0, st["act_col"] if st["drive"] == "cmd" else st["hand_col"]] = src[f]
        robot.set_joint_position_target(tgt)
        robot.write_data_to_sim()
        for _ in range(panel.substeps):
            sim.step(render=False)
        dt_ctrl = 1.0 / max(d["hz"], 1e-6)
        robot.update(dt_ctrl)
        cube.update(dt_ctrl)
        st["frame"] = f
        st["since_seed"] += 1
        # drift from the recording: the whole point of the knobs
        sq = cube.data.root_quat_w[0].cpu().numpy()
        sp = cube.data.root_pos_w[0].cpu().numpy()
        rec = d["cube"][f]
        st["div"] = geo_deg(sq, rec[3:7])
        st["dpos"] = float(np.linalg.norm(sp - rec[:3])) * 1000.0
        st["dropped"] = bool(sp[2] < args.drop_z)
        return True

    panel.apply_live()
    seed(st["frame"])

    if args.selftest:
        def freefall(steps=40):
            """Drop the cube in open space -- a direct read on whether PhysX took the cube's
            damping (position after a fixed number of steps from rest)."""
            pose = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=DEV)
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=DEV))
            cube.reset()
            for _ in range(steps):
                sim.step(render=False)
            cube.update(1.0 / 60.0)
            return float(cube.data.root_pos_w[0, 2])

        return selftest(sim, robot, cube, panel, advance, seed, freefall, st)

    # ------------------------------------------------------------------------ viser
    import viser
    from viser.extras import ViserUrdf

    srv = viser.ViserServer(host="0.0.0.0", port=args.port)
    srv.gui.configure_theme(control_width="large", brand_color=(0, 128, 128))
    srv.scene.add_frame("/hand", position=tuple(map(float, data["hroot_pos"])),
                        wxyz=tuple(map(float, data["hroot_quat"])), show_axes=False)
    vurdf = ViserUrdf(srv, Path(os.path.join(_ROOT, "assets", "urdf", "sharpa", "right_fixed.urdf")),
                      root_node_name="/hand")
    srv.scene.add_grid("/floor", width=1.0, height=1.0, cell_size=0.05)
    cube_node = srv.scene.add_mesh_trimesh("/cube", build_rubik_mesh(data["cube_size"]))
    ghost_frame = srv.scene.add_frame("/ghost", show_axes=False)
    srv.scene.add_line_segments("/ghost/edges", cube_edges(data["cube_size"]),
                                np.array([170, 170, 170], np.uint8), line_width=2.0)

    @srv.on_client_connect
    def _(client):                                      # same close framing as the replay server
        client.camera.position = (0.232, 0.364, 0.435)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.061, 0.068, 0.311)

    def push():
        """Mirror the live sim (and the recorded ghost) into the browser."""
        d = st["data"]
        f = st["frame"]
        jp = robot.data.joint_pos[0].cpu().numpy()
        vurdf.update_cfg({n: float(jp[i]) for i, n in enumerate(art_names)})
        cp = cube.data.root_pos_w[0].cpu().numpy()
        cq = cube.data.root_quat_w[0].cpu().numpy()
        cube_node.position = tuple(float(x) for x in cp)
        cube_node.wxyz = tuple(float(x) for x in cq)
        rec = d["cube"][f]
        ghost_frame.position = tuple(float(x) for x in rec[:3])
        ghost_frame.wxyz = tuple(float(x) for x in rec[3:7])
        ui["tl"].value = f                     # server-side set -> callback sees client=None
        info_md.content = (
            f"**{os.path.basename(d['path'])}**\n\n"
            f"frame&nbsp;**{f} / {d['T'] - 1}**&nbsp;·&nbsp;t&nbsp;{d['tarr'][f]:.2f}s"
            f"{'&nbsp;·&nbsp;**cube dropped**' if st['dropped'] else ''}\n\n"
            f"sim&nbsp;vs&nbsp;recorded&nbsp;**{st['div']:.1f}°**&nbsp;·&nbsp;{st['dpos']:.1f}&nbsp;mm"
            f"&nbsp;·&nbsp;{st['since_seed']}&nbsp;frames&nbsp;since&nbsp;re-seed\n\n"
            f"playback&nbsp;{st['rtf']:.2f}×&nbsp;·&nbsp;{st['steps_per_s']:.0f}&nbsp;physics&nbsp;steps/s")

    ui = {"tl": None}
    folder_pb = srv.gui.add_folder("Playback")
    with folder_pb:
        play_cb = srv.gui.add_checkbox("Play", True, order=1)
        speed_sl = srv.gui.add_slider("Speed", 0.1, 4.0, 0.1, float(args.speed), order=3)
        turbo_cb = srv.gui.add_checkbox("Max speed (no pacing)", False, order=4)
        loop_cb = srv.gui.add_checkbox("Loop", True, order=5)
        reseed_btn = srv.gui.add_button("⟲ Re-seed sim here", color="teal", order=6)
        restart_btn = srv.gui.add_button("↺ Restart from frame 0", order=7)
        every_sl = srv.gui.add_slider("Re-seed every [frames]", 0, 600, 10,
                                      int(st["reseed_every"]), order=8,
                                      hint="0 = free-run. Otherwise the sim is re-anchored to "
                                           "the recording this often, so the drift below is a "
                                           "SHORT-horizon error -- that is what separates "
                                           "parameter sets; long runs just measure chaos")
        reseed_cb = srv.gui.add_checkbox("Re-seed when a knob changes", True, order=9,
                                         hint="restart the frame from the recorded state on every "
                                              "parameter edit, so the comparison is a clean A/B")

    def build_timeline(T):
        """(Re)create the frame scrubber -- a slider's range is fixed at creation, so a
        rollout switch (different length) needs a fresh one."""
        if ui["tl"] is not None:
            ui["tl"].remove()
        with folder_pb:
            tl = srv.gui.add_slider("Frame", 0, max(T - 1, 1), 1, min(st["frame"], T - 1), order=2)

        @tl.on_update
        def _(e):
            if e.client is not None:
                cmds.append(("seek", int(tl.value)))
        ui["tl"] = tl

    build_timeline(data["T"])

    dd = None
    if len(rollouts) > 1:
        with srv.gui.add_folder("Rollout"):
            dd = srv.gui.add_dropdown("File", labels, initial_value=labels[0])

    # ---- the simulator knobs (shared panel; it owns apply/restart/debounce) --------
    panel.after_live = lambda: seed(st["frame"]) if reseed_cb.value else None
    panel.attach(srv, reset_label="Reset to recording defaults")

    with srv.gui.add_folder("Info"):
        info_md = srv.gui.add_markdown("")

    @play_cb.on_update
    def _(e):
        if e.client is not None:
            st["playing"] = bool(play_cb.value)

    @speed_sl.on_update
    def _(e):
        if e.client is not None:
            st["speed"] = float(speed_sl.value)

    @turbo_cb.on_update
    def _(e):
        if e.client is not None:
            st["max_speed"] = bool(turbo_cb.value)

    @loop_cb.on_update
    def _(e):
        if e.client is not None:
            st["loop"] = bool(loop_cb.value)

    @every_sl.on_update
    def _(e):
        if e.client is not None:
            st["reseed_every"] = int(every_sl.value)
            st["since_seed"] = 0

    reseed_btn.on_click(lambda e: cmds.append(("seek", int(st["frame"]))))
    restart_btn.on_click(lambda e: cmds.append(("seek", 0)))

    if dd is not None:
        @dd.on_update
        def _(e):
            if e.client is not None:
                cmds.append(("load", by_label[dd.value]))

    # viser silently takes the next free port when the requested one is busy (e.g. a replay
    # server is already up), so report the one it actually bound, not the one we asked for.
    bound = srv.get_port() if hasattr(srv, "get_port") else args.port
    if bound != args.port:
        print(f"[simreplay] port {args.port} was busy -- serving on {bound} instead")
    print(f"[simreplay] viser 3D view -> http://0.0.0.0:{bound}   (Ctrl-C to stop)")
    push()

    # ------------------------------------------------------------------- main loop
    next_t = time.time()
    last_ui = 0.0
    n_frames = n_steps = 0
    t_rate = time.time()
    try:
        while simulation_app.is_running():
            while cmds:
                kind, payload = cmds.popleft()
                if kind == "seek":
                    seed(payload)
                elif kind == "load":
                    nd = load_rollout(payload)
                    if nd["cube_size"] != st["data"]["cube_size"]:
                        print("[simreplay] note: rollout has a different cube size; the sim keeps "
                              "the one it was built with")
                    bind_rollout(nd)
                    seed(0)
                    build_timeline(nd["T"])
                next_t = time.time()
                push()

            if panel.poll():           # a cube property settled -> the sim was bounced+re-seeded
                next_t = time.time()
                push()

            now = time.time()
            if st["playing"] and (st["max_speed"] or now >= next_t):
                if not advance():                       # hit the end of the recording
                    if st["loop"]:
                        seed(0)
                    else:
                        st["playing"] = False
                        play_cb.value = False
                n_frames += 1
                n_steps += panel.substeps
                step_dt = 1.0 / (st["data"]["hz"] * max(st["speed"], 1e-3))
                next_t = max(next_t + step_dt, now - 0.25)   # never bank more than 0.25s of lag
                if now - last_ui > 0.033:                # ~30 Hz UI; the sim runs faster
                    push()
                    last_ui = now
                if now - t_rate > 0.5:
                    st["rtf"] = n_frames / (now - t_rate) / max(st["data"]["hz"], 1e-6)
                    st["steps_per_s"] = n_steps / (now - t_rate)
                    n_frames = n_steps = 0
                    t_rate = now
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n[simreplay] bye")
    return 0


def selftest(sim, robot, cube, panel, advance, seed, freefall, st):
    """Exercise every knob path in isaac/sim_params.py headlessly (the same code the
    teleop panel runs) and report what the simulator actually did."""
    ok = True
    P, DEFAULTS = panel.params, panel.launch

    def phys_dt_per_step():
        t0 = sim.current_time
        sim.step(render=False)
        return sim.current_time - t0

    print("\n=== selftest ===")
    n_sub = panel.substeps                   # one physics step = 1/(phys_hz * substeps) s
    d0 = phys_dt_per_step()
    e0 = 1.0 / (P["phys_hz"] * n_sub)
    print(f"[1] physics dt at {P['phys_hz']:.0f}Hz x{n_sub}: {d0 * 1e3:.4f} ms (expect {e0 * 1e3:.4f})")
    ok &= abs(d0 - e0) < 1e-6

    P["phys_hz"] = 480.0
    panel.apply_live()
    d1 = phys_dt_per_step()
    e1 = 1.0 / (480.0 * n_sub)
    print(f"[2] after slider -> 480Hz: {d1 * 1e3:.4f} ms (expect {e1 * 1e3:.4f}) "
          f"{'OK' if abs(d1 - e1) < 1e-6 else 'FAIL — dt is not live'}")
    ok &= abs(d1 - e1) < 1e-6
    P["phys_hz"] = DEFAULTS["phys_hz"]

    P["stiffness"], P["damping"], P["armature"] = 12.0, 1.5, 0.01
    panel.apply_live()
    ks = float(robot.data.joint_stiffness[0].mean())
    kd = float(robot.data.joint_damping[0].mean())
    print(f"[3] gains -> stiffness {ks:.2f} damping {kd:.2f} "
          f"{'OK' if abs(ks - 12) < 1e-3 and abs(kd - 1.5) < 1e-3 else 'FAIL'}")
    ok &= abs(ks - 12) < 1e-3 and abs(kd - 1.5) < 1e-3
    P.update(stiffness=DEFAULTS["stiffness"], damping=DEFAULTS["damping"], armature=DEFAULTS["armature"])

    # cube knobs go through USD -> need the restart; check the values stick AND that PhysX
    # actually re-read them (a free fall is damped visibly less far)
    fall_steps = 40 * n_sub                  # 40 frames of sim time (1/6 s at the default config)
    z_rigid = freefall(fall_steps)
    P.update(cube_stiff=5000.0, cube_damp=250.0, cube_lin_damp=12.0, cube_ang_damp=2.5)
    t0 = time.time()
    panel.restart()
    stage = sim.stage
    prim = cube_material_prim(stage, CUBE_PRIM)
    rb = stage.GetPrimAtPath(CUBE_PRIM)
    got_lin = float(rb.GetAttribute("physxRigidBody:linearDamping").Get())
    got_ang = float(rb.GetAttribute("physxRigidBody:angularDamping").Get())
    got_ks = float(prim.GetAttribute("physxMaterial:compliantContactStiffness").Get())
    got_kd = float(prim.GetAttribute("physxMaterial:compliantContactDamping").Get())
    good = (abs(got_lin - 12.0) < 1e-6 and abs(got_ang - 2.5) < 1e-6
            and abs(got_ks - 5000) < 1e-3 and abs(got_kd - 250) < 1e-3)
    print(f"[4] restart ({time.time() - t0:.1f}s) -> USD now holds lin/ang damping "
          f"{got_lin}/{got_ang}, contact stiffness/damping {got_ks}/{got_kd} "
          f"{'OK' if good else 'FAIL'}")
    ok &= good
    z_damped = freefall(fall_steps)
    took = (z_damped - z_rigid) > 0.005                # damped cube stays measurably higher
    print(f"[5] PhysX re-read it? free fall from z=1.0: damping 0.2 -> z {z_rigid:.4f}, "
          f"damping 12 -> z {z_damped:.4f} ({'OK' if took else 'FAIL — USD change was ignored'})")
    ok &= took
    ok &= abs(float(robot.data.joint_stiffness[0].mean()) - P["stiffness"]) < 1e-3

    # the panel's DEFAULT restore (what teleop uses -- no recording to re-anchor to):
    # the live hand + cube state must survive the timeline bounce
    seed(min(400, st["data"]["T"] - 1))
    for _ in range(20):
        advance()
    before = cube.data.root_state_w[0].clone()
    keep_hook, panel.after_restart = panel.after_restart, None
    P["cube_lin_damp"] = 0.35
    panel.restart()
    panel.after_restart = keep_hook
    after = cube.data.root_state_w[0]
    dp = float((after[:7] - before[:7]).abs().max())
    dv = float((after[7:13] - before[7:13]).abs().max())
    print(f"[5b] generic restore across the bounce (teleop path): cube pose delta {dp:.2e}, "
          f"velocity delta {dv:.2e} {'OK' if dp < 1e-5 and dv < 1e-5 else 'FAIL — state was lost'}")
    ok &= dp < 1e-5 and dv < 1e-5

    P.update({k: DEFAULTS[k] for k in DEFAULTS})
    st["frame"] = 0
    panel.restart()
    # Drift vs horizon at the recorded config: the chaos floor every parameter comparison
    # sits on top of. Run it from a MID-rollout frame (the hand is moving there, so the
    # reconstructed seed velocity matters) once per seed-vel mode.
    marks = [10, 25, 50, 100, 200, 300]
    start = min(500, max(1, st["data"]["T"] // 3))
    n = min(marks[-1], st["data"]["T"] - 1 - start)
    keep_mode = args.seed_vel
    print(f"[6] replay at the recorded config from frame {start}; drift from the recording "
          f"vs horizon (deg / mm):")
    print("      frames    " + "".join(f"{h:>13d}" for h in marks[:len([m for m in marks if m <= n])]))
    rate = 0.0
    for mode in ("fd", "zero"):
        args.seed_vel = mode
        seed(start)
        t0 = time.time()
        curve = {}
        for i in range(n):
            advance()
            if (i + 1) in marks:
                curve[i + 1] = (st["div"], st["dpos"])
        rate = n / max(time.time() - t0, 1e-6)
        print(f"      seed-vel {mode:5s}" + "".join(f"{curve[h][0]:7.1f}/{curve[h][1]:5.1f}"
                                                    for h in curve))
    args.seed_vel = keep_mode
    print(f"    {rate:.0f} frames/s = {rate / st['data']['hz']:.2f}x real time")
    print(f"=== selftest {'PASSED' if ok else 'FAILED'} ===\n")
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)
