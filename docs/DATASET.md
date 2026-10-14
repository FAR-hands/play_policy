# DSP simulation dataset (v1)

Get it with `scripts/download_dataset.sh` (or `scripts/download_dataset.sh --from <tar|dir>`):
it verifies checksums and extracts to **`data/dsp_sim_dataset/`**. Paths in this document are
relative to that directory unless they start with `data/`. The same text ships inside the
dataset as `README.md`.

Data for in-hand cube reorientation with a simulated 22-DoF Sharpa right hand in Isaac Lab, as used
in *Dexterity from Supervised Play* (DSP):

- **`teleop/`** — the human play demonstrations: goal-free Manus-glove teleoperation. The paper's
  play policy was trained on exactly these two takes.
- **`play/`** — 5.0 hours (60 Hz label time) of autonomous play, collected by rolling out that play
  policy in parallel simulated environments. `play/paper/` is the exact corpus the paper's world
  model was trained on; `play/extension/` is 1.9 h more from the same policy and collector.

No rewards, goals or task labels are involved anywhere: the play policy imitates goal-free human
play, and the world model is trained on the policy's own rollouts.

## Contents

| split | files | frames | label time (60 Hz) | simulated physics time | cube drops | used in the paper for |
|---|---:|---:|---:|---:|---:|---|
| `teleop/` | 2 | 29,474 | 8.2 min | 2.0 min | 0 (resets trimmed) | play policy (behaviour cloning) |
| `play/paper/` | 128 | 672,000 | 3.11 h | 0.78 h | 794 | world model |
| `play/extension/` | 78 | 409,500 | 1.90 h | 0.47 h | 476 | — (released extra data) |
| **`play/` total** | 206 | 1,081,500 | **5.01 h** | 1.25 h | 1,270 | |

The teleop takes were recorded live: the operator spent 18.1 min of wall-clock time in the frames
that were kept (19.9 min including the trimmed reset windows, see below).

```
dsp_sim_dataset/
  README.md            this datasheet (= docs/DATASET.md)
  teleop/              teleop_0000.npz, teleop_0001.npz
  play/paper/          play_0000.npz ... play_0127.npz   (trials 0-127)
  play/extension/      play_0128.npz ... play_0205.npz   (trials 128-205)
  splits/
    manifest.csv       file, split, trial index, original file name, frames, sha256
    teleop.txt  play_paper.txt  play_extension.txt     file lists per split
    play_trials.jsonl  one record per play trial written by the collector
                       (init/goal quaternions, cube drops, min/final distance to the logged goal, ...)
  stats.json           the numbers in this datasheet, plus SO(3) coverage
  SHA256SUMS           sha256 of every .npz (check with: sha256sum -c SHA256SUMS)
```

Each `.npz` is one recording, written with `numpy.savez_compressed` (so the tarball is not
compressed again). All arrays load with plain `numpy.load(path)`; string arrays are numpy unicode,
no pickles.

## Three clocks: label time, physics time, wall time

Read this before quoting any duration or speed.

- **Physics.** Every recorded frame advances the simulation by **1/240 s**: 3 PhysX substeps of
  1/720 s each.
- **Labels.** Frames are *labelled* 60 Hz (`sim_params_json.control_hz = 60`), and all of the data
  hours above — and in the paper — are frames / 60. The play files' `t` array is this label clock
  (`frame_index / 60`).
- **So the data is in 4x slow motion relative to its labels.** 5.0 h of label time is 1.25 h of
  simulated physics; a velocity computed as Δ/(1/60 s) is 4x lower than the physical one. The
  `cube_state` velocity channels come from PhysX and *are* in physical units (per simulated second).
  This setting was chosen for teleoperation: it gives the operator time to react through the glove.
- **Wall time.** The teleop loop ran at ~26–27 frames per wall-clock second, and the teleop files'
  `t` is wall-clock seconds since the teleop session started (not the label clock). Autonomous
  collection runs far faster than real time (all 206 trials took about 23 min on one RTX 5090 with
  64 environments in lockstep).

## How it was produced

**Scene.** A fixed-base Sharpa right hand mounted palm-up (root at `(0, 0, 0.30)` m, orientation
`hand_root_quat`), and a 5 cm, 80 g Rubik's-textured cube spawned at `(0, 0.07, 0.06)` m from the
hand root. Joints are position-controlled through an implicit PD actuator (stiffness 3.0, damping
0.1). A cube whose centre falls below **z = 0.15 m** counts as dropped and is respawned. The full
physics configuration is stamped into every file (see "Simulator parameters").

**Teleop (`teleop/`).** A human operator wearing a Manus glove played freely with the cube; glove
keypoints were retargeted to the 22 hand joints and sent as position targets. When the cube dropped
or the operator pressed reset, the cube was respawned and the recorder trimmed ±1.5 s
(`trim_s`) around that moment, so each take contains gaps in `t` at the `reset_times` (37 reset
events over the two takes) and no frame with a dropped cube.

**Autonomous play (`play/`).** The paper's DiT diffusion play policy (trained on `teleop/`, 7.5 M
parameters, action chunks of 200 joint targets, 50 executed before resampling) was rolled out with no
goal and no world model: 64 environments in lockstep, one sampled chunk per environment per replan,
DDIM with eta 1.0 and initial-noise scale 1.1, bf16. Each trial starts from a uniformly random cube
orientation (after 60 unlogged settling frames) and lasts 5,250 frames. On a drop the cube is
respawned at the spawn point with a new random orientation and the hand is commanded to its default
pose for 30 frames while the cube settles; those frames are recorded and `t` keeps running, so
consumers must cut segments at the teleport (the released loader does this from the jump in cube
pose). Every trial dropped the cube at least once, 6.2 times on average.

`play/paper/` is byte-identical to the corpus behind the paper's world model. `play/extension/`
continues the same seeded stream (seed 0, trials 128–205) with the same policy weights and collector
flags, exactly as a single 206-trial run would have drawn them. Collection is deterministic on our
hardware: re-collecting trials 0–63 reproduced the first 300 frames of every archived paper file
bit-for-bit.

## File schema

All play and teleop files share the training keys; per-frame arrays have length `T`.

| key | shape / dtype | meaning |
|---|---|---|
| `t` | `(T,)` float32 | time stamp, s — play: label clock `frame/60`, continuous through respawns; teleop: wall-clock s since the session started, with gaps where reset windows were trimmed |
| `hand_joint_pos` | `(T, 22)` float32 | measured hand joint positions, rad, in **articulation order** (`hand_joint_names`) |
| `cmd_q22` | `(T, 22)` float32 | commanded joint position targets applied at this frame, rad, in **command order** (`cmd_q22_names`) — the action |
| `cube_state` | `(T, 13)` float32 | cube root state after the frame's physics step: position xyz (m), orientation quaternion **wxyz** (unit norm, sign not canonicalised), linear velocity xyz (m/s), angular velocity xyz (rad/s); world frame of a single environment (gravity along −z) |
| `hand_joint_names` | `(22,)` str | joint order of `hand_joint_pos` |
| `cmd_q22_names` | `(22,)` str | joint order of `cmd_q22` |
| `cube_size` | `()` float64 | cube edge length, m (0.05) |
| `hand_root_pos` | `(3,)` float32 | hand root position, m: `(0, 0, 0.30)` |
| `hand_root_quat` | `(4,)` float32 | hand root orientation, wxyz (rpy −173.746°, −59.607°, 94.102°) |
| `sim_params_json` | `()` str | the physics configuration read from the running simulator (JSON) |
| `sim_params_hash` | `()` str | 12-hex-digit hash of `sim_params_json` |
| *play only:* `goal_quat` | `(4,)` float32 | a random goal orientation (≥ 90° from the start) logged by the benchmark harness; the play policy never sees it |
| *play only:* `geo` | `(T,)` float32 | geodesic angle between the cube and `goal_quat`, deg (diagnostic only) |
| *teleop only:* `source` | `()` str | `"live"` (Manus glove) |
| *teleop only:* `hz` | `()` float64 | nominal label rate (60) |
| *teleop only:* `reset_times` | `(K,)` float32 | `t` of each cube reset during the take |
| *teleop only:* `trim_s` | `()` float64 | half-width of the window removed around each reset, s (1.5) |
| *teleop only:* `sim_param_changes` | `()` int64 | number of simulator knobs moved while recording (0 for both takes) |

The training state is `[hand_joint_pos (22), cube position (3), cube quaternion (4)] = 29` dims and
the action is `cmd_q22`; the velocities in `cube_state[:, 7:13]` are not used by the released
models.

**Joint orders.** Command order (`cmd_q22_names`) is thumb, index, middle, ring, pinky:

```
thumb_CMC_FE thumb_CMC_AA thumb_MCP_FE thumb_MCP_AA thumb_IP
index_MCP_FE index_MCP_AA index_PIP index_DIP      middle_MCP_FE middle_MCP_AA middle_PIP middle_DIP
ring_MCP_FE ring_MCP_AA ring_PIP ring_DIP          pinky_CMC pinky_MCP_FE pinky_MCP_AA pinky_PIP pinky_DIP
```

(each prefixed `right_`). Articulation order (`hand_joint_names`) is the simulator's
breadth-first order and differs; map between them by name, never by index.

## Simulator parameters

Isaac Sim 5.1 / Isaac Lab 2.3.2, PhysX on GPU. From `sim_params_json` (identical in every file):

| | |
|---|---|
| physics | dt 1/720 s, 3 substeps per frame → 1/240 s per frame; control label 60 Hz |
| hand | implicit PD, stiffness 3.0, damping 0.1, armature 0; fixed root, gravity disabled, no self-collision |
| cube | 5 cm, 0.08 kg; static / dynamic friction 1.2 / 1.0; restitution 0; compliant contact k = 3000 N/m, d = 15 N·s/m; linear / angular damping 0.2 |
| solver | 16 position / 4 velocity iterations; max depenetration velocity 0.5 m/s; sleep 0.005; stabilisation threshold 0.0025 |
| PhysX | stabilisation on, bounce threshold 0.2 m/s, gravity (0, 0, −9.81) m/s² |

`sim_params_hash` is **`21c4c9c015e1`** in every play file and **`90d13aa1b16d`** in both teleop
takes. The physics fields are identical; the hashes differ only because the hashed JSON also holds
non-physics bookkeeping: the cube's USD prim path (`/World/Cube` in the single-environment teleop
scene vs `/World/envs/env_0/Cube` in the multi-environment collector), and the teleop recorder's
extra `source` and `cube_size` fields. If you collect new data with the released code, compare
its hash against these values before pooling it with this dataset.

## Coverage

Fraction of cube orientations visited, from `stats.json` (10° Euler-angle grid; because the
middle Euler angle only spans 180°, at most ~23,300 of the 46,656 grid cells can ever be occupied,
so we report a same-size sample of uniformly random rotations as the reference):

| | occupied cells | effective cells (exp entropy) | uniform rotations within 10° of a sample |
|---|---:|---:|---:|
| `teleop/` | 7,044 | 4,750 | 63.0% |
| `play/paper/` | 22,961 | 15,411 | 100% |
| `play/` (5 h) | 23,219 | 15,742 | 100% |
| uniform reference (1.08 M rotations) | 23,324 | 20,053 | 100% |

Autonomous play visits essentially every orientation within 10°, whereas the two teleop takes cover
about 30% of the occupied cells; visitation is less even than uniform (78% of the uniform effective
cell count).

## Using the data

From the repository root (the trainers' defaults are the paper's recipes):

```bash
# play policy on the human demos (DiT, H=200, 50 executed steps, 40k steps)
python -m play_policy.train --data_dir data/dsp_sim_dataset/teleop

# world model on the paper's 3.11 h corpus (DiT, H=50, 3-frame history at stride 3, 150k steps)
python -m play_policy.train_worldmodel --data_dir data/dsp_sim_dataset/play/paper

# ... or on all 5 h of play
python -m play_policy.train_worldmodel --data_dir data/dsp_sim_dataset/play
```

The loader cuts every recording into contiguous segments at respawn teleports (cube position jump
> 3 cm or orientation jump > 45° between frames) and, for teleop, at the trimmed reset gaps in `t`,
then holds out whole segments for validation. With the default split seed this reproduces the
paper's splits exactly: `teleop/` → 31 segments (26 train / 5 val, 20,801 / 1,845 windows of 200
frames), `play/paper/` → 910 segments (819 / 91, 559,331 / 62,251 windows of 50 frames). All of
`play/` gives 1,458 segments (1,312 / 146, 907,373 / 93,397 windows).

Minimal loading without the code release:

```python
import numpy as np
d = np.load("data/dsp_sim_dataset/play/paper/play_0000.npz")
state = np.concatenate([d["hand_joint_pos"], d["cube_state"][:, :7]], axis=1)   # (T, 29)
action = d["cmd_q22"]                                                            # (T, 22)
pos = d["cube_state"][:, :3]
quat = d["cube_state"][:, 3:7]                                                   # wxyz
q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
dang = np.degrees(2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(1)), 0, 1)))
cut = np.where((np.linalg.norm(np.diff(pos, axis=0), axis=1) > 0.03) | (dang > 45))[0] + 1
segments = np.split(np.arange(len(state)), cut)                                  # respawns removed
```

## Integrity

`(cd data/dsp_sim_dataset && sha256sum -c SHA256SUMS)` checks every recording;
`scripts/download_dataset.sh` does this automatically.

## License and citation

TODO(authors): dataset license and BibTeX entry.
