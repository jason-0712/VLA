#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/eval_competition.sh {clean|randomized|all}

Required environment variables:
  LINGBOT_VLA_ROOT   Pinned LingBot-VLA 2.0 checkout
  ROBOTWIN_ROOT      Pinned RoboTwin 2.0 checkout
  MODEL_PATH         Exported checkpoint's hf_ckpt directory
  QWEN3VL_PATH       Qwen3-VL-4B-Instruct checkpoint
  OUTPUT_BASE        Parent directory for timestamped evaluation runs
  CONDA_SH           Path to conda.sh

Optional:
  INFERENCE_ENV      Default: lingbotvla
  SIM_ENV            Default: RoboTwin
  NUM_GPUS           Default: 4
  NUM_PER_GPU        Default: 1
  NUM_TASKS          Default: 50
  USE_LENGTH         Default: 50
EOF
}

setting="${1:-}"
if [[ "$setting" == "-h" || "$setting" == "--help" || -z "$setting" ]]; then
  usage
  [[ -n "$setting" ]] && exit 0 || exit 1
fi

if [[ "$setting" != "clean" && "$setting" != "randomized" && "$setting" != "all" ]]; then
  echo "Setting must be clean, randomized, or all." >&2
  exit 1
fi

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT}"
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to the exported hf_ckpt directory}"
: "${QWEN3VL_PATH:?Set QWEN3VL_PATH}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE}"
: "${CONDA_SH:?Set CONDA_SH}"

[[ -d "$LINGBOT_VLA_ROOT" ]] || { echo "Missing LingBot checkout: $LINGBOT_VLA_ROOT" >&2; exit 2; }
[[ -d "$ROBOTWIN_ROOT" ]] || { echo "Missing RoboTwin checkout: $ROBOTWIN_ROOT" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing checkpoint directory: $MODEL_PATH" >&2; exit 2; }
[[ "$(basename "$MODEL_PATH")" == "hf_ckpt" ]] || { echo "MODEL_PATH must point to the exported hf_ckpt directory." >&2; exit 2; }

inference_env="${INFERENCE_ENV:-lingbotvla}"
sim_env="${SIM_ENV:-RoboTwin}"
num_gpus="${NUM_GPUS:-4}"
num_per_gpu="${NUM_PER_GPU:-1}"
num_tasks="${NUM_TASKS:-50}"
use_length="${USE_LENGTH:-50}"

run_setting() {
  local task_config="$1"
  (
    cd "$LINGBOT_VLA_ROOT"
    bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
      --model_path "$MODEL_PATH" \
      --eval_workdir "$ROBOTWIN_ROOT" \
      --output_base "$OUTPUT_BASE" \
      --conda_sh "$CONDA_SH" \
      --inference_env "$inference_env" \
      --sim_env "$sim_env" \
      --task_config "$task_config" \
      --num_tasks "$num_tasks" \
      --num_gpus "$num_gpus" \
      --num_per_gpu "$num_per_gpu" \
      --use_length "$use_length" \
      --use_bf16 False \
      --use_fp32 True
  )
}

case "$setting" in
  clean) run_setting demo_clean ;;
  randomized) run_setting demo_randomized ;;
  all)
    run_setting demo_clean
    run_setting demo_randomized
    ;;
esac
