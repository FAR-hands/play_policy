# play_policy — DiT play policy + DiT world model

Torch-only library (no simulator needed): data loading, the goal-free diffusion **play policy**
and the action-conditioned diffusion **world model** used by the MPC controller, plus their
training scripts.

```
data.py               recording loader + segmentation, Normalizer, window datasets, rotation helpers
model.py              PlayDiTPolicy, PlayDPWorldModel (shared DDPM/DDIM core, DiT1D denoiser),
                      build_policy / build_world_model / load_world_model_ckpt
train.py              behavior cloning of the play policy on teleop takes
train_worldmodel.py   world-model training on autonomous play
viz.py                TensorBoard figures (pred vs GT, samples, error vs horizon)
export_ckpt.py        strip optimizer / raw training weights for inference
../isaac/deploy_core.py   load_policy + observation / chunk helpers (Isaac-free)
```

## Data

Every `*.npz` under a data directory is one recording (searched recursively; files without the
keys below, such as the collector's `traj_*.npz` planner sidecars, are skipped):

| key | shape | meaning |
|---|---|---|
| `t` | (T,) | timestamps (s); label clock of 60 frames/s |
| `hand_joint_pos` | (T, 22) | observed hand joints, articulation order (`hand_joint_names`) |
| `cmd_q22` | (T, 22) | commanded joint targets — **the action** (`cmd_q22_names` order) |
| `cube_state` | (T, 13) | cube pos (m, hand-env frame) + quaternion **wxyz** + lin vel + ang vel |
| `hand_joint_names`, `cmd_q22_names` | (22,) | joint legends, copied into every checkpoint |

State = `[hand_joint_pos(22), cube_state[:, :7]]` (29-D), action = `cmd_q22` (22-D). Each file is
split into contiguous segments at every reset — a cube teleport (> 3 cm or > 45° in one frame,
which is how the collector respawns a dropped cube), `t` going backwards, or a `t` gap (teleop
takes trim the frames around a reset) — and no training window crosses a segment boundary.
States and actions are min-max normalized to [-1, 1] with statistics fit on the training split;
the normalizers are stored in the checkpoint.

## Training (defaults = the paper recipe)

```bash
# play policy on human teleop takes: H=200 / exec 50, DiT d256 depth 6 patch 4, 50 diffusion
# steps (10 DDIM), Adam 1e-4, batch 64, 40k steps, 15% of segments held out
python -m play_policy.train --data_dir <teleop dir>

# world model on autonomous play: H=50, cond = [s_t, s_t-3, s_t-6], DiT d512 depth 8 patch 5,
# 100 diffusion steps (10 DDIM), AdamW 1e-4 fixed LR + EMA, batch 128, 150k steps, 10% held out
python -m play_policy.train_worldmodel --data_dir <play dir> [<more dirs> ...]
```

On one RTX 5090 the policy runs at ~120 steps/s (**~6 min** for 40k steps) and the world model at
~65 steps/s (**~40 min** for 150k steps). Each run writes `<ckpt_dir>/<timestamp>/` with
`ckpt_best.pt` (lowest held-out error — the one to use), `ckpt_final.pt`, periodic
`ckpt_XXXXXXXX.pt` and TensorBoard logs under `tb/`; the log prints `run dir: <path>` once.
Training is seeded (`--seed`): on the same GPU / software stack two runs agree exactly over the
first few hundred steps, then drift apart slowly (nondeterministic GPU kernels); a full re-run of
the policy recipe ended at validation action MSE 0.226 vs 0.224 for the released checkpoint. Resume an
interrupted run with `--resume <ckpt> --max_steps <final step>`.

`python -m play_policy.export_ckpt <ckpt_best.pt> <out.pt>` drops the optimizer state and the
world model's raw (non-EMA) weights: 90 → 30 MB for the policy, 626 → 157 MB for the world model.

Trainer flags (`--help` for the full text):

- `train.py`: `--data_dir` (required) `--H 200 --n_action_steps 50 --patch_size 4 --d_model 256
  --depth 6 --n_heads 8 --mlp_ratio 4 --dsed 256 --schedule cosine --n_timesteps 50
  --n_sample_steps 10 --lr 1e-4 --batch_size 64 --max_steps 40000 --seed 0 --split_seed 0
  --val_frac 0.15 --resume --ckpt_dir runs/play_policy --ckpt_every 5000 --log_every 100
  --val_every 1000 --val_batches 8 --val_mc_samples 4 --viz_every 5000 --device`
- `train_worldmodel.py`: `--data_dir` (required) `--val_data_dir --H 50 --n_hist 3
  --hist_stride 3 --rot_weight (off) --patch_size 5 --d_model 512 --depth 8 --n_heads 8
  --mlp_ratio 4 --dsed 256 --schedule cosine --n_timesteps 100 --n_sample_steps 10 --lr 1e-4
  --weight_decay 1e-6 --batch_size 128 --max_steps 150000 --ema/--no-ema --seed 0
  --split_seed 0 --val_frac 0.1 --resume --ckpt_dir runs/world_model --ckpt_every 25000
  --log_every 100 --val_every 1000 --val_batches 6 --train_eval_every 5000 --val_mc_samples 2
  --viz_every 5000 --device`

## Inference API

```python
import sys; sys.path[:0] = [".", "isaac"]          # repo root + isaac/
import torch
from deploy_core import load_policy
from play_policy.model import load_world_model_ckpt
from play_policy.data import Normalizer

policy, norm_s, norm_a, meta, cfg = load_policy("checkpoints/dsp_sim_play_policy.pt", "cuda")
wm, ck = load_world_model_ckpt("checkpoints/dsp_sim_world_model.pt", device="cuda")
wm_norm_s = Normalizer.from_state_dict(ck["norm_state"])
wm_norm_a = Normalizer.from_state_dict(ck["norm_action"])

# proposals: (N, 29) normalized states -> (N, 200, 22) normalized joint-command chunks
chunks = policy(state_n.repeat(1024, 1), eta=1.0, init_noise_scale=1.1)
joint_targets = chunks.cpu().numpy() * norm_a.scale + norm_a.center          # radians

# prediction: (N, 87) history stack [s_t, s_t-3, s_t-6] in the WM's normalization +
# (N, 50, 22) actions in the WM's cmd order/normalization -> (N, 50, 7) normalized cube pose
cube_n = wm(cond_n, act_n)
cube = cube_n.cpu().numpy() * wm_norm_s.scale[22:29] + wm_norm_s.center[22:29]  # pos (m) + quat wxyz
```

Both models sample with DDIM in normalized space and return values clamped to [-1, 1]. Pass a
`torch.Generator` (or `x_init`) for repeatable samples; otherwise the global torch RNG is used.
Remap joints by name (`meta["hand_joint_names"]`, `meta["cmd_q22_names"]`, and the same keys in
`ck["meta"]`), never by position. `data.stack_history(frames, 3, 3)` builds the world model's
conditioning from a buffer of past frames exactly as the training windows do. Both models run
under `torch.autocast("cuda", dtype=torch.bfloat16)` (the MPC harness default) and
`torch.compile` of their `.dit` attribute.

## Checkpoint schema

Play policy (`policy_type: "dit"`): `policy` (state_dict), `norm_state` / `norm_action`
(`{center, scale}`), `meta` (`hand_joint_names`, `cmd_q22_names`, `cube_dims`, `hz`, ...),
`state_dim` 29, `action_dim` 22, `horizon`, `n_action_steps`, `patch_size`, `d_model`, `depth`,
`n_heads`, `mlp_ratio`, `dsed`, `schedule`, `n_timesteps`, `n_sample_steps`, `cube_dims` 7,
`step`, `best_val_action_mse_norm`; training checkpoints add `opt`.

World model (`world_model: True`): `policy` (EMA state_dict, the deploy weights), `norm_state` /
`norm_action`, `meta`, `state_dim` (= `n_hist` x `frame_dim` = 87), `frame_dim` 29, `n_hist`,
`hist_stride`, `act_dim` 22, `obj_dim` / `cube_dim` 7, `out_cols` (state columns of the target,
22..28), `rot_repr` "quat", `horizon`, `backbone` "dit", `objective` "ddpm", `patch_size`,
`d_model`, `depth`, `n_heads`, `mlp_ratio`, `dsed`, `schedule`, `n_timesteps`,
`n_sample_steps`, `out_weight`, `step`, `best_val_obj_mse_norm`; training checkpoints add `opt`
and `policy_train` (raw weights). The released checkpoints carry extra legacy keys from the
original training code (e.g. `policy_type: "unet_worldmodel"`, `down_dims`); the loaders ignore
them and refuse checkpoints that use features outside this release.
