#!/usr/bin/env bash
# Closed-loop any-to-any reorientation in Isaac Lab with the released checkpoints:
# play policy proposes, world model scores, MPC executes -- the paper's recipe.
#
#   quickstart/run_sim_mpc.sh                 # 8 trials, one round (~5 min on an RTX 5090)
#   quickstart/run_sim_mpc.sh --full          # the paper's 128 trials (~50 min on an RTX 5090)
#   quickstart/run_sim_mpc.sh --baseline      # play policy alone on the same trials and goals
#   quickstart/run_sim_mpc.sh --full --baseline
#   quickstart/run_sim_mpc.sh --trials 32 --num_envs 32 --out quickstart_out/my_run
#
# Small runs also write roll_XXXX.npz (full per-step hand/cube/goal trajectories; --record adds
# them to --full runs, --no-record drops them) for the browser replay:
#   python isaac/viser_replay_server.py --rollout quickstart_out/sim_mpc_small   # -> http://localhost:8080
#
# Protocol: random initial cube orientation, random goal >= 90 deg away, 2500 control steps,
# dropped cubes are re-spawned in the palm. --full runs the 128 trials as two passes of 64
# (trial_start 0 and 64, 32 envs per round), exactly how the published run was launched, so every
# trial's first replan is reproduced bit-for-bit (later steps diverge: contact is chaotic).
# Resumable: re-running the same command skips finished rounds.
#
# Env: PYTHON (default: python of the active env, which needs Isaac Sim 5.1 + Isaac Lab 2.3.2),
#      CKPT_DIR (default: checkpoints/, filled by quickstart/download_checkpoints.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
POL="$CKPT_DIR/dsp_sim_play_policy.pt"
WM="$CKPT_DIR/dsp_sim_world_model.pt"

FULL=0; BASE=0; TRIALS=""; ENVS=""; OUT=""; RECORD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --baseline) BASE=1 ;;
    --record) RECORD=1 ;;
    --no-record) RECORD=0 ;;
    --trials) TRIALS="$2"; shift ;;
    --num_envs) ENVS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    -h|--help) sed -n 2,26p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
for f in "$POL" "$WM"; do
  [ -f "$f" ] || { echo "missing $f -- run quickstart/download_checkpoints.sh first" >&2; exit 1; }
done

TAG=$([ $FULL = 1 ] && echo full || echo small)$([ $BASE = 1 ] && echo _baseline || echo "")
OUT="${OUT:-quickstart_out/sim_mpc_$TAG}"
mkdir -p "$OUT"
export OMNI_KIT_ACCEPT_EULA=YES

COMMON=(--policy_ckpt "$POL" --sim_steps 2500 --sample_eta 1.0 --init_noise_scale 1.1
        --goal_min_deg 90 --seed 0 --amp bf16 --skip_existing --out "$OUT")
if [ $BASE = 1 ]; then
  # play policy alone: one chunk per replan, executed blind (n_exec 50, as in the paper baseline)
  ARM=(--mode policy --n_exec 50)
  [ $FULL = 1 ] && PASSES=("0 128") || PASSES=("0 ${TRIALS:-8}")
  ENVS_DEF=$([ $FULL = 1 ] && echo 128 || echo "${TRIALS:-8}")
else
  ARM=(--wm_ckpt "$WM" --n_samples 1024 --plan_batch 256 --n_exec 25 --compile
       --reuse_elites --reuse_frac 0.5 --reuse_hist 3 --reuse_keep 512 --reuse_joint_tol 0.5)
  [ $FULL = 1 ] && PASSES=("0 64" "64 64") || PASSES=("0 ${TRIALS:-8}")
  ENVS_DEF=$([ $FULL = 1 ] && echo 32 || echo "${TRIALS:-8}")
fi
ENVS="${ENVS:-$ENVS_DEF}"
RECORD="${RECORD:-$([ $FULL = 1 ] && echo 0 || echo 1)}"
[ "$RECORD" = 1 ] && COMMON+=(--record)

echo "[run_sim_mpc] $TAG -> $OUT  (envs/round $ENVS; log $OUT/run.log; Isaac boot + compile ~1-2 min)"
for p in "${PASSES[@]}"; do
  read -r TS NT <<< "$p"
  set +e
  "$PY" -u isaac/wm_mpc_eval_batch.py "${COMMON[@]}" "${ARM[@]}" \
    --num_envs "$ENVS" --trial_start "$TS" --trials "$NT" 2>&1 | tee -a "$OUT/run.log" \
    | grep --line-buffered -E "^\[(mpc|policy-only|round [0-9/]+|batch-mpc)\]|Error|error:"
  st=${PIPESTATUS[0]}
  set -e
  [ "$st" = 0 ] || { echo "[run_sim_mpc] harness failed (exit $st) -- see $OUT/run.log" >&2; exit 1; }
done

"$PY" isaac/wm_mpc_report.py "$OUT" | tail -1
echo "[run_sim_mpc] figure: $OUT/mpc_report.png   per-trial records: $OUT/shard_*.jsonl"
[ "$RECORD" = 1 ] && echo "[run_sim_mpc] replay in a browser: $PY isaac/viser_replay_server.py --rollout $OUT"
cat <<'EOF'
[run_sim_mpc] paper reference, 128 trials (small runs scatter widely around these):
    play + WM + MPC     SR@0.4/0.6/0.8 rad = 72.7 / 91.4 / 96.1 %   reach@30 87.5 %   min-geo 17.31 deg
    play policy alone   SR@0.4/0.6/0.8 rad = 32.8 / 53.9 / 72.7 %   reach@30 46.9 %   min-geo 34.7 deg
EOF
