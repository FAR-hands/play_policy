#!/usr/bin/env python3
"""BATCHED world-model MPC: many Isaac envs in ONE process, one fused inference batch.

Goal-reorientation benchmark: random initial cube orientation, random goal >= --goal_min_deg
away, receding-horizon replanning, the true geodesic-to-goal logged every control step.
``--num_envs`` trials run in LOCKSTEP inside a single simulation, and every replan fuses all
envs' proposals into one GPU batch (the planning itself lives in ``mpc_core.py``):

  every --n_exec control steps, for the envs due to replan (usually all of them):
    1. sample --n_samples action chunks PER ENV from the play policy   (one (k*N) batch)
    2. roll every chunk through the world model                        (same (k*N) batch)
    3. score = min over the horizon of the predicted geodesic to that env's goal
    4. execute the first --n_exec actions of each env's best chunk

  --mode mpc     : play policy PROPOSES, world model SCORES, Isaac EXECUTES (needs --wm_ckpt)
  --mode policy  : play policy only, no world model, no goal awareness. One chunk is sampled
                   per replan and executed blind (the goal-agnostic baseline), with identical
                   trials/goals/logging. With --record this is also the SELF-PLAY COLLECTOR:
                   every trial is written as roll_XXXX.npz in the training-data schema.

A dropped cube (z < --drop-z) is re-spawned in the palm with a random orientation; the hand
holds its default pose for 30 steps while it settles, and the trial continues.

  # MPC, the paper recipe (elite archive: 512 fresh + 512 re-used proposals per replan)
  OMNI_KIT_ACCEPT_EULA=YES python isaac/wm_mpc_eval_batch.py \
      --policy_ckpt checkpoints/dsp_sim_play_policy.pt --wm_ckpt checkpoints/dsp_sim_world_model.pt \
      --num_envs 32 --trials 128 --sim_steps 2500 --n_samples 1024 --plan_batch 256 --n_exec 25 \
      --amp bf16 --compile --sample_eta 1.0 --init_noise_scale 1.1 --goal_min_deg 90 \
      --reuse_elites --reuse_frac 0.5 --reuse_hist 3 --reuse_keep 512 --reuse_joint_tol 0.5 \
      --out runs/mpc
  # self-play collection: 128 trials x 5250 frames = 672k frames
  OMNI_KIT_ACCEPT_EULA=YES python isaac/wm_mpc_eval_batch.py --mode policy --record \
      --policy_ckpt checkpoints/dsp_sim_play_policy.pt --num_envs 64 --trials 128 \
      --sim_steps 5250 --n_exec 50 --sample_eta 1.0 --init_noise_scale 1.1 --amp bf16 --compile \
      --out data/play_new

Outputs in --out: traj_XXXX.npz (per-step geodesic, plan scores, init/goal) and one
shard_<trial_start>.jsonl line per trial; ``isaac/wm_mpc_report.py <out>`` aggregates them.
Seeding: trial i draws its init/goal from numpy seed (--seed + i); torch is seeded per ROUND of
--num_envs trials (--seed + 100003 + round), so re-running with the same --num_envs and
--trial_start reproduces every first replan exactly (later steps diverge: contact is chaotic).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from sim_params import CONTROL_HZ  # noqa: E402  (isaac/ is the script dir; Isaac-free at module level)
_USD = os.path.join(_HERE, "assets", "sharpa_right", "sharpa_right.usd")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--policy_ckpt", required=True)
ap.add_argument("--wm_ckpt", default=None, help="required for --mode mpc; ignored for --mode policy")
ap.add_argument("--mode", choices=["mpc", "policy"], default="mpc",
                help="'mpc' = policy proposes + WM scores; 'policy' = play policy only (baseline / collector)")
ap.add_argument("--num_envs", type=int, default=8, help="trials simulated in lockstep per round")
ap.add_argument("--env_spacing", type=float, default=2.0)
ap.add_argument("--trials", type=int, default=128)
ap.add_argument("--trial_start", type=int, default=0, help="global index of the first trial")
ap.add_argument("--sim_steps", type=int, default=2500, help="control steps per trial")
ap.add_argument("--n_samples", type=int, default=1024, help="policy action chunks proposed per replan per env")
ap.add_argument("--n_exec", type=int, default=50, help="executed steps of the chosen chunk before replanning")
ap.add_argument("--goal_min_deg", type=float, default=90.0, help="min goal distance from init orientation")
ap.add_argument("--success_deg", type=float, default=15.0, help="min-distance below this = success")
ap.add_argument("--init_quats_npz", default=None,
                help="npz with 'quat_wxyz' (N,4): trial i starts from row i %% N instead of a random "
                     "orientation (pins a trial, e.g. to redo it)")
ap.add_argument("--goal_quats_npz", default=None,
                help="npz with 'quat_wxyz' (N,4): trial i uses goal row i %% N instead of a random goal; "
                     "an unpinned init is then re-drawn until it is >= --goal_min_deg from it")
ap.add_argument("--time_weight", type=float, default=0.0,
                help="time-efficiency penalty (deg per horizon step) added to the score")
ap.add_argument("--sample_eta", type=float, default=0.0,
                help="stochastic reverse-sampling strength for the proposals (0 = DDIM, 1 = DDPM)")
ap.add_argument("--init_noise_scale", type=float, default=1.0, help="scale on the initial action latent")
ap.add_argument("--noop-gate", dest="noop_gate", action=argparse.BooleanOptionalAction, default=True,
                help="hold when no proposal is predicted to beat the CURRENT object pose")
ap.add_argument("--noop-margin", dest="noop_margin", type=float, default=0.0)
ap.add_argument("--reuse_elites", action="store_true",
                help="elite-archive MPC: sample only (1-reuse_frac)*n_samples FRESH chunks per replan "
                     "and fill the rest with the best-scoring chunks from the last --reuse_hist "
                     "replans, time-aligned to the current step. Total scored per replan is "
                     "unchanged (n_samples), so compute is comparable to the standard arms.")
ap.add_argument("--reuse_frac", type=float, default=0.5,
                help="fraction of each replan's proposals drawn from the elite archive")
ap.add_argument("--reuse_hist", type=int, default=3,
                help="how many past replans the archive spans")
ap.add_argument("--reuse_keep", type=int, default=0,
                help="elites stored per replan (0 = reuse_frac*n_samples). The stored sets from "
                     "all --reuse_hist replans are POOLED and ranked together each replan; the "
                     "best reuse_frac*n_samples survive.")
ap.add_argument("--reuse_joint_tol", type=float, default=0.5,
                help="action-continuity gate (rad): an archived chunk is reusable only if its first "
                     "aligned command is within this of the LAST ISSUED COMMAND. Fresh "
                     "proposals already jump ~0.35 rad median, so 0.5 is not stricter.")
ap.add_argument("--plan_batch", type=int, default=256, help="chunks per forward (memory/throughput cap)")
ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off",
                help="autocast dtype for the diffusion forwards ('off' = fp32). bf16 leaves the "
                     "chosen proposal unchanged in practice (score error ~0.1 deg vs a ~6 deg spread)")
ap.add_argument("--compile", action="store_true",
                help="torch.compile the play policy's denoiser (the paper configuration)")
ap.add_argument("--compile_wm", action="store_true",
                help="also torch.compile the world model's denoiser: faster, but not bit-identical "
                     "to the eager world model the paper's numbers were produced with")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True, help="output dir for results")
ap.add_argument("--record", action="store_true",
                help="also write full per-step trajectories (roll_XXXX.npz, training-data schema)")
ap.add_argument("--skip_existing", action="store_true",
                help="skip any round whose trials all already have a traj_XXXX.npz in --out "
                     "(resume an interrupted run)")
ap.add_argument("--hz", type=float, default=CONTROL_HZ,
                help="control-rate label stamped into the data (sim_params.CONTROL_HZ = the demo pool's rate)")
ap.add_argument("--cube-size", type=float, default=0.05)
ap.add_argument("--hand-pos", type=float, nargs=3, default=[0.0, 0.0, 0.30])
ap.add_argument("--hand-rpy", type=float, nargs=3, default=[-173.746, -59.607, 94.102])
ap.add_argument("--cube-offset", type=float, nargs=3, default=[0.0, 0.07, 0.06])
ap.add_argument("--stiffness", type=float, default=3.0)
ap.add_argument("--damping", type=float, default=0.1)
ap.add_argument("--drop-z", type=float, default=0.15)
ap.add_argument("--usd", default=_USD)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
assert args.mode == "policy" or args.wm_ckpt, "--mode mpc needs --wm_ckpt"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- sim app live below ----
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sim_params import (PHYSICS_DT, SUBSTEPS, cube_physics_material,  # noqa: E402
                        stamp as stamp_sim_params)
from mpc_core import Planner, geo_deg, geo_deg_t, rand_quat_wxyz  # noqa: E402

DEV = args.device
torch.backends.cudnn.benchmark = True


def quat_wxyz_from_rpy_deg(rpy):
    q = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()  # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


# ---------------------------------------------------------------------------
# Scene: num_envs clones of (hand + cube), identical physics to the teleop scene
# ---------------------------------------------------------------------------
@configclass
class MpcSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Sharpa",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                                                                       enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(args.hand_pos),
                                                   rot=quat_wxyz_from_rpy_deg(args.hand_rpy),
                                                   joint_pos={".*": 0.0}),
        actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=args.stiffness,
                                                  damping=args.damping, effort_limit_sim=2.0,
                                                  velocity_limit_sim=10.0)})
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(args.cube_size, args.cube_size, args.cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5,
                                                         solver_position_iteration_count=16,
                                                         solver_velocity_iteration_count=4,
                                                         linear_damping=0.2, angular_damping=0.2,
                                                         sleep_threshold=0.005, stabilization_threshold=0.0025),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=cube_physics_material()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(args.hand_pos[0] + args.cube_offset[0],
                                                       args.hand_pos[1] + args.cube_offset[1],
                                                       args.hand_pos[2] + args.cube_offset[2])))


def _pinned(path, label):
    if not path:
        return None
    q = np.asarray(np.load(path)["quat_wxyz"], np.float64).reshape(-1, 4)
    print(f"[{label}] {len(q)} pinned orientations from {path}")
    return q


def main():
    os.makedirs(args.out, exist_ok=True)
    E = int(args.num_envs)
    init_quats = _pinned(args.init_quats_npz, "init")
    goal_quats = _pinned(args.goal_quats_npz, "goal")

    # ---- scene ----
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=DEV,
                                      physx=sim_utils.PhysxCfg(enable_stabilization=True,
                                                               bounce_threshold_velocity=0.2))
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(MpcSceneCfg(num_envs=E, env_spacing=args.env_spacing))
    sim.reset()
    robot, cube = scene["robot"], scene["cube"]
    origins = scene.env_origins                                       # (E,3) world offsets
    art_names = list(robot.joint_names)

    # ---- models (the planner remaps every legend to the articulation's joint order) ----
    planner = Planner(args.policy_ckpt, args.wm_ckpt if args.mode == "mpc" else None, art_names, DEV,
                      num_envs=E, n_samples=args.n_samples, plan_batch=args.plan_batch,
                      sample_eta=args.sample_eta, init_noise_scale=args.init_noise_scale,
                      amp=args.amp, compile=args.compile, compile_wm=args.compile_wm,
                      time_weight=args.time_weight,
                      noop_gate=args.noop_gate, noop_margin=args.noop_margin,
                      reuse_elites=args.reuse_elites, reuse_frac=args.reuse_frac,
                      reuse_hist=args.reuse_hist, reuse_keep=args.reuse_keep,
                      reuse_joint_tol=args.reuse_joint_tol)
    print(f"{planner.describe()}  n_exec={args.n_exec} envs={E} amp={args.amp} "
          f"compile={args.compile} compile_wm={args.compile_wm}", flush=True)
    H, HBUF, frame_dim = planner.H, planner.HBUF, planner.frame_dim
    use_wm = planner.wm is not None
    act_col_t = planner.act_col_t

    default_targets = robot.data.default_joint_pos.clone()             # (E, 22)
    dt = 1.0 / max(args.hz, 1.0)

    def step_sim():
        scene.write_data_to_sim()
        for _ in range(SUBSTEPS):
            sim.step(render=False)   # physics only: same PhysX step, no Kit update
        scene.update(dt)

    def cube_local():
        """Cube root state with the env origin removed -> the single-env frame the models saw."""
        st = cube.data.root_state_w.clone()
        st[:, :3] -= origins
        return st

    last_cmd_t = torch.zeros(E, 22, device=DEV)   # last q22 actually commanded, per env

    # ---- trial loop (rounds of E trials in lockstep) ----
    n_rounds = (args.trials + E - 1) // E
    t_start = time.time()
    done = 0
    for rd in range(n_rounds):
        gis = [args.trial_start + rd * E + e for e in range(E)]
        live = [e for e in range(E) if rd * E + e < args.trials]           # envs with a real trial
        if args.skip_existing and all(os.path.exists(os.path.join(args.out, f"traj_{gis[e]:04d}.npz"))
                                      for e in live):
            done += len(live)
            print(f"[round {rd + 1}/{n_rounds}] already done, skipping (trials "
                  f"{gis[live[0]]}..{gis[live[-1]]})", flush=True)
            continue
        rngs = [np.random.default_rng(args.seed + gi) for gi in gis]
        # per-round torch seed so proposal noise is reproducible across re-runs
        torch.manual_seed(args.seed + 100003 + rd)
        # -- reset: random init orientation per env, then settle --
        root = cube.data.default_root_state.clone()
        pose = root[:, :7].clone()
        pose[:, :3] += origins
        for e in range(E):
            q0 = rand_quat_wxyz(rngs[e])
            if init_quats is not None:
                q0 = init_quats[gis[e] % len(init_quats)]
            elif goal_quats is not None:                  # keep the trial non-trivial: init far from goal
                for _ in range(1000):
                    if geo_deg(q0, goal_quats[gis[e] % len(goal_quats)]) >= args.goal_min_deg:
                        break
                    q0 = rand_quat_wxyz(rngs[e])
            pose[e, 3:7] = torch.as_tensor(q0, dtype=pose.dtype, device=DEV)
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
        cube.reset()
        robot.write_joint_state_to_sim(default_targets.clone(), torch.zeros_like(default_targets))
        robot.reset(); robot.set_joint_position_target(default_targets)
        for _ in range(60):
            step_sim()
        if rd == 0:   # sanity: the env-origin-removed cube pose must match the single-env frame
            cl = cube_local()[:, :3].cpu().numpy()
            print(f"[scene] envs={E} origins[0]={origins[0].tolist()} origins[-1]={origins[-1].tolist()}\n"
                  f"[scene] cube local pos: min={cl.min(0).round(4).tolist()} "
                  f"max={cl.max(0).round(4).tolist()}  (expect ~[0, 0.07, 0.33-0.36])", flush=True)
        init_q = cube.data.root_quat_w.cpu().numpy().astype(np.float64)     # (E,4)
        goal_q = np.zeros((E, 4), np.float64)
        for e in range(E):
            if goal_quats is not None:
                goal_q[e] = goal_quats[gis[e] % len(goal_quats)]
                continue
            for _ in range(1000):
                g = rand_quat_wxyz(rngs[e])
                if geo_deg(g, init_q[e]) >= args.goal_min_deg:
                    break
            goal_q[e] = g
        goal_init_deg = geo_deg(goal_q, init_q)                            # (E,)
        goal_t_all = torch.as_tensor(goal_q, device=DEV, dtype=torch.float32)

        # history buffer: prefill with the settled state (oldest -> newest along dim 1)
        hist = torch.zeros(E, HBUF, frame_dim, device=DEV)
        if use_wm:
            hist[:] = planner.wm_frame(robot.data.joint_pos, cube_local())[:, None, :]

        geo_hist = np.empty((E, args.sim_steps), np.float32)
        best_chunk = torch.zeros(E, H, 22, device=DEV)
        idx = np.full(E, 10**9, np.int64)                                  # force replan on step 0
        drops = np.zeros(E, np.int64); holds = np.zeros(E, np.int64)
        plan_scores = [[] for _ in range(E)]; plan_at = [[] for _ in range(E)]
        # per-env post-drop settle: the hand is commanded to its default pose for SETTLE steps while
        # the re-spawned cube falls into it. These steps stay INSIDE the main loop and are logged,
        # so a drop in one env never advances -- or freezes -- another env's trial.
        SETTLE = 30
        settle_left = np.zeros(E, np.int64)
        default_q22 = default_targets[:, act_col_t].clone()
        live_mask = np.zeros(E, bool); live_mask[live] = True
        planner.reset_archive()                                            # plans do not survive a trial
        frames = [] if args.record else None
        t_round = time.time()
        for t in range(args.sim_steps):
            if not simulation_app.is_running():
                break
            need = np.where((idx >= args.n_exec) & (settle_left <= 0) & live_mask)[0]
            if len(need):
                hand_art = robot.data.joint_pos
                cst = cube_local()
                ch, pscore, _best_step, execute, _geo_now = planner.plan(
                    need, hand_art, cst, goal_q, hist, t, last_cmd_t)
                ex = execute.cpu().numpy(); ps_np = pscore.cpu().numpy()
                for j, e in enumerate(need):
                    plan_scores[e].append(float(ps_np[j])); plan_at[e].append(t)
                    if not ex[j]:                                          # nothing beats the current pose
                        holds[e] += 1
                        best_chunk[e] = hand_art[e][act_col_t][None, :].expand(H, 22)
                    else:
                        best_chunk[e] = ch[j]
                    idx[e] = 0
            q22 = best_chunk[torch.arange(E, device=DEV), torch.as_tensor(
                np.minimum(idx, H - 1), device=DEV)]                        # (E,22)
            if (settle_left > 0).any():                                     # post-drop: default pose
                q22 = torch.where(torch.as_tensor(settle_left > 0, device=DEV)[:, None],
                                  default_q22, q22)
            idx = np.where(settle_left > 0, idx, idx + 1)
            settle_left = np.maximum(settle_left - 1, 0)
            last_cmd_t.copy_(q22)
            targets = default_targets.clone()
            targets[:, act_col_t] = q22
            robot.set_joint_position_target(targets)
            step_sim()
            cst = cube_local()
            cube_q = torch.nn.functional.normalize(cst[:, 3:7].float(), dim=-1)
            geo_hist[:, t] = geo_deg_t(cube_q, goal_t_all).cpu().numpy()
            if use_wm:                                                      # roll the history buffer
                hist = torch.roll(hist, -1, dims=1)
                hist[:, -1] = planner.wm_frame(robot.data.joint_pos, cst)
            if frames is not None:
                frames.append({"hand_joint_pos": robot.data.joint_pos.cpu().numpy().astype(np.float32),
                               "cmd_q22": q22.cpu().numpy().astype(np.float32),
                               "cube_state": cst.cpu().numpy().astype(np.float32)})
            lowz = (cst[:, 2] < args.drop_z).cpu().numpy() & (settle_left <= 0) & live_mask
            if lowz.any():
                ee = np.where(lowz)[0]
                pose = cube.data.default_root_state[:, :7].clone()
                pose[:, :3] += origins
                for e in ee:
                    pose[e, 3:7] = torch.as_tensor(rand_quat_wxyz(rngs[e]), dtype=pose.dtype, device=DEV)
                eidx = torch.as_tensor(ee, device=DEV)
                cube.write_root_pose_to_sim(pose[eidx], env_ids=eidx)
                cube.write_root_velocity_to_sim(
                    torch.zeros(len(ee), 6, device=DEV, dtype=pose.dtype), env_ids=eidx)
                jp = default_targets.clone()
                robot.write_joint_state_to_sim(jp[eidx], torch.zeros_like(jp[eidx]), env_ids=eidx)
                drops[ee] += 1
                settle_left[ee] = SETTLE                                    # settle inside the main loop
                idx[ee] = 10**9
                planner.reset_archive(ee)                                   # teleport invalidates plans
                if use_wm:                                                  # history is stale after a teleport
                    f = planner.wm_frame(robot.data.joint_pos, cube_local())
                    hist[eidx] = f[eidx][:, None, :].expand(len(ee), HBUF, frame_dim)

        # ---- per-trial records ----
        def _longest_below(gg, th):
            below = gg < th; best = cur = 0
            for bb in below:
                cur = cur + 1 if bb else 0
                if cur > best:
                    best = cur
            return best / args.hz
        for e in live:
            gi = gis[e]
            gh = geo_hist[e]
            rec = {"trial": gi, "init_quat": init_q[e].tolist(), "goal_quat": goal_q[e].tolist(),
                   "goal_init_deg": float(goal_init_deg[e]), "min_geo_deg": float(gh.min()),
                   "final_geo_deg": float(gh[-1]), "min_at_step": int(gh.argmin()),
                   "success": bool(gh.min() < args.success_deg), "drops": int(drops[e]),
                   "mean_plan_score": (float(np.mean(plan_scores[e])) if (plan_scores[e] and use_wm)
                                       else None),
                   "n_replans": len(plan_scores[e]), "holds": int(holds[e]),
                   "time_weight": float(args.time_weight),
                   "hold_secs_15": float(_longest_below(gh, args.success_deg)),
                   "hold_secs_30": float(_longest_below(gh, 30.0))}
            rec["held_3s_15"] = bool(rec["hold_secs_15"] >= 3.0)
            rec["held_3s_30"] = bool(rec["hold_secs_30"] >= 3.0)
            np.savez_compressed(os.path.join(args.out, f"traj_{gi:04d}.npz"),
                                geo=gh, plan_scores=np.array(plan_scores[e], np.float32),
                                plan_at=np.array(plan_at[e], np.int64),
                                init_quat=init_q[e], goal_quat=goal_q[e])
            if frames is not None:
                fd = {"t": np.arange(len(frames), dtype=np.float32) / args.hz,
                      "hand_joint_pos": np.asarray([f["hand_joint_pos"][e] for f in frames]),
                      "cmd_q22": np.asarray([f["cmd_q22"][e] for f in frames]),
                      "cube_state": np.asarray([f["cube_state"][e] for f in frames]),
                      "hand_joint_names": np.array(art_names),
                      "cmd_q22_names": np.array(list(planner.meta_p["cmd_q22_names"])),
                      "cube_size": np.array(args.cube_size),
                      "hand_root_pos": np.array(args.hand_pos, np.float32),
                      "hand_root_quat": np.array(quat_wxyz_from_rpy_deg(args.hand_rpy), np.float32),
                      "goal_quat": goal_q[e].astype(np.float32), "geo": gh}
                stamp_sim_params(fd, sim, robot, cube, steps_per_frame=SUBSTEPS, control_hz=args.hz)
                np.savez_compressed(os.path.join(args.out, f"roll_{gi:04d}.npz"), **fd)
            with open(os.path.join(args.out, f"shard_{args.trial_start:04d}.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            done += 1
        el = time.time() - t_round
        eta = (time.time() - t_start) / max(done, 1) * (args.trials - done)
        rs, aw = planner.reuse_stat, planner.arch_won
        print(f"[round {rd + 1}/{n_rounds}] {len(live)} trials in {el / 60:.1f} min "
              f"({el / max(len(live), 1):.0f}s/trial)  min_geo={geo_hist[live].min(axis=1).mean():.1f}deg mean  "
              + (f"reuse={100.0 * rs[0] / max(rs[1], 1):.0f}% ARCH-WON={100.0 * aw[0] / max(aw[1], 1):.1f}%  "
                 if args.reuse_elites else "")
              + f"done={done}/{args.trials}  ETA {eta / 60:.0f} min", flush=True)

    print(f"[batch-mpc] done {done} trials -> {args.out}  "
          f"total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    _ok = False
    try:
        main(); _ok = True
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if _ok else 1)
