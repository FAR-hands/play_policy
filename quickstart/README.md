# Quick start: play policy + world model + MPC for any-to-any reorientation

This folder checks the released simulation checkpoints, the ones behind the paper's simulated
reorientation result, at three levels. Each level takes longer than the one before:

| level | what runs | needs | time (RTX 5090) |
|---|---|---|---|
| 1. offline check | one MPC replan on 4 recorded states, compared with a stored reference | torch (GPU optional) | ~10 s GPU, 1-5 min CPU |
| 2. small closed loop | 8 reorientation trials in Isaac Lab | Isaac Sim 5.1 + Isaac Lab 2.3.2 | ~4 min |
| 3. paper reproduction | the 128-trial benchmark | Isaac Sim 5.1 + Isaac Lab 2.3.2 | ~45 min |

The two checkpoints:

| file | model | trained on |
|---|---|---|
| `dsp_sim_play_policy.pt` (30 MB) | DiT diffusion play policy, 7.5M params; predicts 200-step chunks of 22 joint targets from hand joints + cube pose | 29,474 frames of human glove teleop |
| `dsp_sim_world_model.pt` (157 MB) | DiT diffusion world model, 39M params; predicts the next 50 cube poses from a 3-frame state history (stride 3) and a 50-step action chunk | 672,000 frames of autonomous play collected by that policy |

## 0. Get the checkpoints

```bash
quickstart/download_checkpoints.sh             # -> checkpoints/, sha256-verified
# or, with the files already on disk:
quickstart/download_checkpoints.sh --from /path/to/dir [--link]
```

## 1. Offline check (no simulator)

```bash
python quickstart/check_inference.py           # add --amp bf16 to match the harness, --device cpu without a GPU
```

`sample_states.npz` holds 4 real states from the released dataset: two from teleop takes and two
from self-play rollouts. Each state is 7 consecutive frames (the world model's history) plus a goal
orientation drawn at least 90 deg away. For each state the script runs one MPC replan with
`isaac/mpc_core.py`, the same code the simulator harness runs, using the paper's settings:

1. Sample 1024 play-policy proposals (DDPM sampling, eta 1.0, init noise 1.1).
2. Roll each proposal 50 steps through the world model.
3. Score each proposal by its closest predicted approach to the goal and keep the best.

Expected output (RTX 5090, fp32, seed 0):

```
[replan] 4 states x 1024 proposals = 4096 rollouts in 2396 ms (599 ms per state)
state                                        now    best     p10     p50     p90  step exec
teleop/teleop_0001.npz                     100.5    27.5    56.9    88.4    98.2    49 True
teleop/teleop_0000.npz                     117.7    83.5   109.4   117.5   118.4    49 True
play/paper/play_0003.npz                   154.0   138.5   150.4   153.4   154.2    49 True
play/paper/play_0077.npz                   171.7    99.7   128.1   144.0   159.5    49 True
[check] PASS -- policy, world model and MPC scoring behave as released
```

Column key:
- `now`: the current distance to the goal.
- `best`: the chosen plan's closest predicted approach.
- `p10` / `p50` / `p90`: percentiles of the 1024 proposals' closest predicted approach.

The figure `quickstart_out/check_inference.png` has two rows:
- Top: predicted distance-to-goal along the 50 world-model steps, for random proposals, the top 16, and the chosen one.
- Bottom: the score histogram.

**What PASS means.** The proposals are diffusion samples. A different GPU, driver, torch build or
the CPU generator draws a different random stream, which is effectively a different seed. So the
check compares robust statistics against `reference.json` and accepts any result within
max(1 deg, 4 x the seed-to-seed std) of the 8-seed reference mean:
- the best predicted approach
- the 10th-percentile proposal score
- the median proposal score

The percentiles are tight (std < 1 deg). A wrong checkpoint, normalizer or joint ordering moves
them by tens of degrees. On the reference stack (RTX 5090, torch 2.7.0+cu128) seed 0 reproduces
the reference exactly.

## 2. Small closed-loop run (Isaac Lab)

```bash
conda activate <your Isaac Lab env>     # Isaac Sim 5.1 + Isaac Lab 2.3.2
quickstart/run_sim_mpc.sh               # 8 trials, MPC                                 (~4 min)
quickstart/run_sim_mpc.sh --baseline    # same 8 trials and goals, play policy alone   (~3 min)
```

Protocol:
- Each trial starts from a random cube orientation in the palm and gets a random goal at least 90 deg away.
- Each trial runs 2500 control steps.
- Every 25 steps the planner proposes 1024 chunks: 512 freshly sampled, plus 512 "elites" re-used from the last 3 replans, time-aligned and gated on action continuity. The world model scores all of them and the best chunk is executed.
- Dropped cubes are re-spawned in the palm.

Outputs in `quickstart_out/sim_mpc_small/`:
- `stats.json`
- `mpc_report.png`, a 6-panel summary that includes the average best-so-far distance to goal over time
- per-trial `traj_*.npz` and `shard_*.jsonl`
- `roll_*.npz` full trajectories. Watch them in a browser:

```bash
python isaac/viser_replay_server.py --rollout quickstart_out/sim_mpc_small   # http://localhost:8080
```

A second, goal-oriented cube is drawn beside the hand. When the stickers match, the goal is reached.

## 3. Paper reproduction

```bash
quickstart/run_sim_mpc.sh --full              # 128 trials, MPC (~45 min)
quickstart/run_sim_mpc.sh --full --baseline   # 128 trials, play policy alone (~3 min)
```

`--full` runs the 128 trials the way the published run was launched: two passes of 64, with 32
envs per round. The seeding, the physics and the planner numerics then all match, so on the
reference stack the whole closed loop reproduces: all 128 trials gave a per-step geodesic
bit-identical to the published run, and the report reads SR 72.7 / 91.4 / 96.1 %, reach@30
87.5 %, min-geo 17.31 deg, 3 stuck trials (RTX 5090, torch 2.7.0+cu128, Isaac Sim 5.1, ~45 min).
`--full --baseline` (128 envs in one round, n_exec 50) likewise reproduced all 128 published
baseline trials exactly: 32.8 / 53.9 / 72.7 %.
Contact dynamics are chaotic, so on a different GPU, driver or PhysX build the trajectories
diverge after the first replans. Expect the headline numbers to scatter by a few points there
(the noise floor at n=128 is about ±6 deg of min-geo).

Paper reference, 128 trials, goals at least 90 deg away, 2500 steps. e(t) is the true geodesic
to the goal; SR@theta is the fraction of trials with min_t e(t) <= theta:

| arm | SR@0.4 rad | SR@0.6 rad | SR@0.8 rad | reach@30 deg | mean min error |
|---|---|---|---|---|---|
| play + world model + MPC | 72.7 % | 91.4 % | 96.1 % | 87.5 % | 17.3 deg |
| play policy alone | 32.8 % | 53.9 % | 72.7 % | 46.9 % | 34.7 deg |

**Stuck trials.** Sometimes the cube comes to rest on the hand above the drop line, where no finger
reaches it. It is then never re-spawned. `wm_mpc_report.py` lists such trials under
`stuck_trials`: the geodesic is frozen for at least 300 steps while more than 30 deg from the goal.
The published run had 3. To redo one, pin its start and goal from `shard_*.jsonl`, write
`np.savez(f, quat_wxyz=[q])` files, and pass them as
`--init_quats_npz` / `--goal_quats_npz` with a new `--seed`.

## Speed notes

- `--amp bf16` and `--compile` are what the paper runs used. `--compile` compiles the **policy only**,
  and the paper's world model ran eager.
- One replan of 32 envs x 1024 proposals takes 5.75 s in bf16 on an RTX 5090, which is about 100
  replans per 32-trial round, ~10 min.
- `isaac/wm_mpc_eval_batch.py --compile_wm` also compiles the world model. It cuts the replan only to
  5.24 s (-9 %) and is not bit-identical to the eager model. On the 64 reproduced first replans the
  chosen plan's score moved by 1.2 deg (median; mean 1.9, max 13). That is inside the
  seed-to-seed spread, but the run no longer matches the published one, so it is off by default.
- The offline check's replan (4 x 1024 proposals) takes 2.4 s in fp32 and 0.84 s in bf16.

## Files

| file | what |
|---|---|
| `download_checkpoints.sh` | fetch / copy the checkpoints into `checkpoints/`, verify `SHA256SUMS` |
| `check_inference.py` | level 1, offline replan + reference check + figure |
| `sample_states.npz` | 4 recorded 7-frame states + goals; `source` / `frame` name their dataset file and frame |
| `reference.json` | 8-seed reference statistics for `check_inference.py` (RTX 5090, torch 2.7.0) |
| `run_sim_mpc.sh` | levels 2 and 3 (wraps `isaac/wm_mpc_eval_batch.py` + `isaac/wm_mpc_report.py`) |
