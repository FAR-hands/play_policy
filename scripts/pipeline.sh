#!/usr/bin/env bash
# DSP simulation pipeline, end to end and resumable:
#
#   human teleop takes --(1)--> play policy --(2)--> autonomous self-play --(3)--> world model
#                                                                            --(4)--> MPC eval
#
# Defaults reproduce the paper's simulation recipe (any-to-any cube reorientation, Sharpa hand):
#   1  DiT diffusion play policy, H=200 chunks / 50 executed, 40k steps        ~6 min
#   2  batched self-play, 64 envs in lockstep, 672,000 frames (3.1 h @60 Hz)   ~10 min
#   3  DiT world model, H=50, 3-frame history at stride 3, 150k steps          ~40 min
#   4  play-guided MPC, 128 trials, 1024 proposals/replan (512 fresh + 512     ~50 min
#      re-used elites), 25 executed steps per replan, goals >= 90 deg away
# (times: one RTX 5090). Paper result for this recipe: reach@30 deg 87.5%, mean min error 17.3 deg,
# SR@0.4/0.6/0.8 rad = 72.7/91.4/96.1%.
#
# Usage (from the repository root, inside the `dsp` conda env):
#   bash scripts/pipeline.sh                                   # released teleop data
#   HUMAN_DATA=data/my_takes RUN=runs/mine bash scripts/pipeline.sh   # your own glove takes
#   setsid nohup bash scripts/pipeline.sh > pipeline.out 2>&1 &   # detached; tail -f $RUN/pipeline.log
#
# Resumable: each stage records its artifact in $RUN/state.env and is skipped on a re-run.
# Delete a line from state.env to redo that stage, or pre-set POLICY_CKPT / PLAY_DATA / WM_CKPT
# to start from an existing artifact (e.g. PLAY_DATA=data/dsp_sim_dataset/play skips stages 1-2).
#
# Knobs (env vars):
#   HUMAN_DATA    teleop takes for stage 1              [data/dsp_sim_dataset/teleop]
#   RUN           output dir (logs, state.env)          [runs/pipeline_<date>]
#   POLICY_STEPS  play-policy training steps            [40000]
#   FRAMES        self-play frames to collect           [672000]  (5 h = 1080000)
#   COLLECT_TRIALS self-play episodes (FRAMES/COLLECT_TRIALS frames each)   [128]
#   NUM_ENVS      parallel envs for self-play           [64]
#   WM_STEPS      world-model training steps            [150000]
#   MPC_TRIALS    MPC evaluation trials                 [128]
#   MPC_ENVS      parallel envs for MPC                 [32]
#   MPC_STEPS     control frames per MPC trial          [2500]
#   POLICY_ARGS / WM_ARGS   extra flags for the two trainers, e.g. WM_ARGS="--batch_size 64"
#   PYTHON        interpreter                           [python]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-runs/pipeline_$(date +%Y%m%d)}"; mkdir -p "$RUN"
LOG="$RUN/pipeline.log"
STATE="$RUN/state.env"; touch "$STATE"; . "$STATE"
PY="${PYTHON:-python}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1

HUMAN_DATA="${HUMAN_DATA:-data/dsp_sim_dataset/teleop}"
FRAMES="${FRAMES:-672000}"; COLLECT_TRIALS="${COLLECT_TRIALS:-128}"; NUM_ENVS="${NUM_ENVS:-64}"
FRAMES_PER_TRIAL=$(( FRAMES / COLLECT_TRIALS ))
MPC_TRIALS="${MPC_TRIALS:-128}"; MPC_ENVS="${MPC_ENVS:-32}"; MPC_STEPS="${MPC_STEPS:-2500}"

log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }
save(){ grep -v "^$1=" "$STATE" > "$STATE.tmp" 2>/dev/null || true; mv "$STATE.tmp" "$STATE"
        echo "$1=$2" >> "$STATE"; export "$1=$2"; }
die(){ log "FAILED at $* -- see the stage log in $RUN/"; exit 1; }
run_dir(){ grep -oE "run dir: .*" "$1" | tail -1 | cut -d' ' -f3-; }

log "=== DSP pipeline -> $RUN ==="

# ---- 1. play policy: DiT behaviour cloning on goal-free human teleop ------------------------
if [ -z "${POLICY_CKPT:-}" ]; then
  [ -d "$HUMAN_DATA" ] || die "stage 1: HUMAN_DATA '$HUMAN_DATA' not found (bash scripts/download_dataset.sh, or record takes with scripts/teleop.sh)"
  log "[1/4] play policy on $HUMAN_DATA (${POLICY_STEPS:-40000} steps) ..."
  TL="$RUN/1_policy.log"
  "$PY" -u -m play_policy.train --data_dir "$HUMAN_DATA" \
      --max_steps "${POLICY_STEPS:-40000}" --ckpt_dir runs/play_policy ${POLICY_ARGS:-} > "$TL" 2>&1 \
      || die "stage 1 (policy)"
  PDIR=$(run_dir "$TL")
  [ -f "$PDIR/ckpt_best.pt" ] || die "stage 1: no ckpt_best.pt in '$PDIR'"
  save POLICY_CKPT "$PDIR/ckpt_best.pt"
  log "[1/4] -> $POLICY_CKPT"
else
  log "[1/4] skip (POLICY_CKPT=$POLICY_CKPT)"
fi

# ---- 2. autonomous self-play: the policy alone, batched across NUM_ENVS envs ----------------
# Dropped cubes respawn at a random orientation; each episode is one roll_XXXX.npz in the
# training schema with the physics stamped in (sim_params_hash).
COLLECT=(--mode policy --record --policy_ckpt "$POLICY_CKPT" --n_exec 50
         --sample_eta 1.0 --init_noise_scale 1.1 --amp bf16 --compile)
if [ -z "${PLAY_DATA:-}" ]; then
  OUT="$RUN/play_data"
  # flag-identical smoke test: a crash after the ~1 min Isaac boot should cost 1 min, not an hour
  log "[2/4] smoke test (2 envs x 300 frames, same flags) ..."
  "$PY" -u isaac/wm_mpc_eval_batch.py "${COLLECT[@]}" --num_envs 2 --trials 2 --sim_steps 300 \
      --out "$RUN/smoke" > "$RUN/2_smoke.log" 2>&1 || die "stage 2 smoke test"
  log "[2/4] collecting $FRAMES frames: $COLLECT_TRIALS episodes x $FRAMES_PER_TRIAL, $NUM_ENVS envs ..."
  "$PY" -u isaac/wm_mpc_eval_batch.py "${COLLECT[@]}" --num_envs "$NUM_ENVS" \
      --trials "$COLLECT_TRIALS" --sim_steps "$FRAMES_PER_TRIAL" --skip_existing \
      --out "$OUT" > "$RUN/2_collect.log" 2>&1 || die "stage 2 (self-play)"
  FR=$("$PY" -c 'import glob,sys,numpy as np
print(sum(len(np.load(f)["cmd_q22"]) for f in glob.glob(sys.argv[1]+"/roll_*.npz")))' "$OUT")
  [ "${FR:-0}" -gt $(( FRAMES * 3 / 4 )) ] || die "stage 2: only ${FR:-0} frames in $OUT"
  save PLAY_DATA "$OUT"
  log "[2/4] -> $OUT ($FR frames)"
  "$PY" isaac/coverage_so3.py "$OUT/roll_*.npz" --outdir "$RUN" 2>&1 | grep -E "^(source|POOLED)" | tee -a "$LOG"
else
  log "[2/4] skip (PLAY_DATA=$PLAY_DATA)"
fi

# ---- 3. action-conditioned world model on the self-play data --------------------------------
if [ -z "${WM_CKPT:-}" ]; then
  log "[3/4] world model on $PLAY_DATA (${WM_STEPS:-150000} steps) ..."
  WL="$RUN/3_world_model.log"
  "$PY" -u -m play_policy.train_worldmodel --data_dir "$PLAY_DATA" \
      --max_steps "${WM_STEPS:-150000}" --ckpt_dir runs/world_model ${WM_ARGS:-} > "$WL" 2>&1 \
      || die "stage 3 (world model)"
  WDIR=$(run_dir "$WL")
  [ -f "$WDIR/ckpt_best.pt" ] || die "stage 3: no ckpt_best.pt in '$WDIR'"
  save WM_CKPT "$WDIR/ckpt_best.pt"
  log "[3/4] -> $WM_CKPT"
else
  log "[3/4] skip (WM_CKPT=$WM_CKPT)"
fi

# ---- 4. play-guided MPC on any-to-any reorientation ----------------------------------------
MOUT="$RUN/mpc"
if [ -z "${MPC_DONE:-}" ]; then
  log "[4/4] MPC: $MPC_TRIALS trials, $MPC_ENVS envs, 1024 proposals, elite re-use ..."
  "$PY" -u isaac/wm_mpc_eval_batch.py \
      --policy_ckpt "$POLICY_CKPT" --wm_ckpt "$WM_CKPT" \
      --num_envs "$MPC_ENVS" --trials "$MPC_TRIALS" --sim_steps "$MPC_STEPS" \
      --n_samples 1024 --plan_batch 256 --n_exec 25 \
      --amp bf16 --compile --skip_existing \
      --sample_eta 1.0 --init_noise_scale 1.1 --goal_min_deg 90 \
      --reuse_elites --reuse_frac 0.5 --reuse_hist 3 --reuse_keep 512 --reuse_joint_tol 0.5 \
      --out "$MOUT" > "$RUN/4_mpc.log" 2>&1 || die "stage 4 (MPC)"
  save MPC_DONE "$MOUT"
fi
"$PY" isaac/wm_mpc_report.py "$MOUT" 2>&1 | tail -20 | tee -a "$LOG"
log "=== DONE -> $MOUT/stats.json (paper: reach@30 87.5%, SR@0.4/0.6/0.8 rad 72.7/91.4/96.1%) ==="
