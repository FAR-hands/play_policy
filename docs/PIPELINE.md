# The simulation pipeline

```
 (0) human play            (1) play policy         (2) autonomous play       (3) world model         (4) MPC
 Manus glove -> Isaac  ->  DiT diffusion BC   ->  policy alone, N envs  ->  action-conditioned  ->  policy proposes,
 teleop takes (.npz)       on goal-free takes      in lockstep, hours        DiT, H=50 frames         WM scores, best
                                                   of (o, a, o') data                                 chunk is executed
```

`scripts/pipeline.sh` runs stages 1–4 with the paper's settings, resumably (see the header of
the script for every knob). This page shows what each stage does and the equivalent manual
commands, run from the repository root inside the `dsp` env with `OMNI_KIT_ACCEPT_EULA=YES`.

| stage | default budget | time on one RTX 5090 |
|---|---|---|
| 1 play policy | 40k steps, batch 64 | ~6 min |
| 2 self-play | 672,000 frames (3.1 h @60 Hz label), 64 envs | ~10 min (5 h: ~15–25 min) |
| 3 world model | 150k steps, batch 128 | ~38 min |
| 4 MPC eval | 128 trials x 2,500 frames, 32 envs | ~45 min |

**Reference run.** `bash scripts/pipeline.sh` with all defaults on the released teleop takes
(policy retrained from scratch -> 672,000 new self-play frames -> new world model -> 128-trial
MPC; 1 h 33 min on one RTX 5090) gave SR@0.4/0.6/0.8 rad **71.9 / 90.6 / 96.1 %**, reach@30
86.7 %, mean min error 16.9° — the paper's checkpoints give 72.7 / 91.4 / 96.1 %, 87.5 %, 17.3°.
Its self-play covered 49.5 % of the 46,656 SO(3) bins, the same as the paper's corpus (49.5 %).

## Starting from released artifacts

Every stage can start from an existing artifact by pre-setting its variable, so the released
checkpoints and dataset slot in anywhere:

```bash
# world model on the released 5 h of play + MPC with the paper's play policy (stages 3-4)
POLICY_CKPT=checkpoints/dsp_sim_play_policy.pt PLAY_DATA=data/dsp_sim_dataset/play \
    RUN=runs/wm_5h bash scripts/pipeline.sh

# the paper's world-model corpus exactly (672,000 frames)
POLICY_CKPT=checkpoints/dsp_sim_play_policy.pt PLAY_DATA=data/dsp_sim_dataset/play/paper \
    RUN=runs/wm_paper bash scripts/pipeline.sh

# MPC only, with any pair of checkpoints
POLICY_CKPT=<policy.pt> PLAY_DATA=- WM_CKPT=<world_model.pt> RUN=runs/mpc_only bash scripts/pipeline.sh
```

With the default flags, `play_policy.train` on `teleop/` and `play_policy.train_worldmodel` on
`play/paper/` are the recipes of the released checkpoints: on our stack they reproduce the
original training runs exactly over the first 300 steps (losses, weights, optimizer state).
Over a full run, nondeterministic GPU kernels make runs drift apart; a full re-run of the
policy recipe ended at validation action MSE 0.226 vs 0.224 for the released checkpoint.

## 0. Human play (optional)

Record goal-free play with the glove — move the cube around in the hand, re-grasp, roll it over
every face; no task, no goals:

```bash
bash scripts/teleop.sh --no-sim-panel        # glove server + Isaac teleop; takes -> data/teleop_takes/
```

See [TELEOP.md](TELEOP.md) for hardware, the MANUS SDK, calibration and the recording keys
(`s` start, `e` end + save, `r` reset the cube). The paper's play policy was trained on the two takes in
`data/dsp_sim_dataset/teleop` (29,474 frames). Use `isaac/inspect_npz.py` to check a take:
its `sim_params` must not have been changed mid-take, and its physics fields must match the
released data.

## 1. Play policy

```bash
python -m play_policy.train --data_dir data/dsp_sim_dataset/teleop      # or data/teleop_takes
```

DiT diffusion policy (7.5M parameters): observation = 22 hand joint positions + cube pose
(position + wxyz quaternion), output = a 200-frame chunk of absolute joint targets of which 50
are executed before re-planning. Defaults are the paper's; the run directory
(`runs/play_policy/<timestamp>`, printed as `run dir: ...`) holds `ckpt_best.pt` (lowest validation action MSE) and TensorBoard logs.

## 2. Autonomous play

```bash
python isaac/wm_mpc_eval_batch.py --mode policy --record \
    --policy_ckpt runs/play_policy/<run>/ckpt_best.pt \
    --num_envs 64 --trials 128 --sim_steps 5250 --n_exec 50 \
    --sample_eta 1.0 --init_noise_scale 1.1 --amp bf16 --compile \
    --out runs/play_data
```

The policy runs alone (no goal) in `--num_envs` environments stepped in lockstep in one
process, so a policy forward is one batched call. A dropped cube is respawned in the hand at a
random orientation. Each episode becomes one `roll_XXXX.npz` in the training schema with the
physics stamped in (`sim_params_hash`, `21c4c9c015e1` for the released settings). 128 x 5,250
frames = 672,000 frames, the paper's budget; `--trials 206` gives the 5 h of the released
dataset. Larger `--num_envs` is faster as long as GPU memory allows (512 envs collect ~40 h
of label time per wall-clock hour).

Coverage of the collected orientations (46,656-bin SO(3) grid):

```bash
python isaac/coverage_so3.py "runs/play_data/roll_*.npz" --outdir runs/play_data
```

## 3. World model

```bash
python -m play_policy.train_worldmodel --data_dir runs/play_data      # or data/dsp_sim_dataset/play
```

DiT denoiser (39M parameters) that predicts the next 50 frames of cube pose given the current
state, two history states (3 and 6 frames back) and a 50-frame chunk of joint targets.
Trained only on autonomous play — no teleop, no goals, no rewards.

## 4. Play-guided MPC

```bash
python isaac/wm_mpc_eval_batch.py \
    --policy_ckpt runs/play_policy/<run>/ckpt_best.pt --wm_ckpt runs/world_model/<run>/ckpt_best.pt \
    --num_envs 32 --trials 128 --sim_steps 2500 --n_samples 1024 --plan_batch 256 --n_exec 25 \
    --amp bf16 --compile --skip_existing --sample_eta 1.0 --init_noise_scale 1.1 --goal_min_deg 90 \
    --reuse_elites --reuse_frac 0.5 --reuse_hist 3 --reuse_keep 512 --reuse_joint_tol 0.5 \
    --out runs/mpc
python isaac/wm_mpc_report.py runs/mpc        # stats.json + mpc_report.png
```

At every replan the play policy proposes 512 fresh chunks and up to 512 elite chunks from the
last three replans are re-used (time-shifted, kept only if their first command is within
0.5 rad of the last issued command on every joint). The world model rolls all 1024 out 50
frames, the score is the closest predicted approach to the goal orientation, and the best
chunk is executed for 25 frames. Each trial starts from a random orientation with a random
goal at least 90° away and lasts 2,500 frames; dropped cubes respawn. `--mode policy` with the
same seed runs the goal-blind play policy on the identical trials as a baseline.

## Practical notes

- Isaac Lab ignores SIGTERM; stop a stuck run with `kill -9 <pid>`. All Isaac scripts here
  end with `os._exit` because Kit can hang inside `simulation_app.close()`.
- Run long stages detached (`setsid nohup ... &`) and with `python -u` (or
  `PYTHONUNBUFFERED=1`, which `pipeline.sh` sets) so logs stream.
- Checkpoints store model + optimizer every `--ckpt_every` steps (~630 MB each for the world
  model); delete intermediate ones you do not need.
- MPC results are noisy: per-trial outcomes diverge chaotically, so two runs of the same
  models differ by several points at n=128. Compare arms on the same trials and seeds.
