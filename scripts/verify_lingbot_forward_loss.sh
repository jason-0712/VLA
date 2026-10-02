#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot source checkout}"
: "${LEROBOT_V2_ROOT:?Set LEROBOT_V2_ROOT to the pinned LeRobot v2 source checkout}"
: "${LEROBOT_V2_ENV:?Set LEROBOT_V2_ENV to the isolated LeRobot v2 environment}"
: "${TRAIN_DATASET_PATH:?Set TRAIN_DATASET_PATH to one audited clean LeRobot dataset}"
: "${NORM_STATS_PATH:?Set NORM_STATS_PATH to clean-only normalization statistics}"
: "${MODEL_PATH:?Set MODEL_PATH to the LingBot-VLA 2.0 base-model snapshot}"
: "${QWEN3VL_PATH:?Set QWEN3VL_PATH to the Qwen3-VL-4B-Instruct snapshot}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to an output directory outside Git}"

lerobot_python="$LEROBOT_V2_ENV/bin/python"
if [[ ! -x "$lerobot_python" ]]; then
  echo "Missing LeRobot v2 Python interpreter: $lerobot_python" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$LEROBOT_V2_ROOT:$LINGBOT_VLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log_dir="$OUTPUT_BASE/forward_loss_smoke/logs"
mkdir -p "$log_dir"
timestamp="$(date +%Y%m%dT%H%M%S)"
log_path="$log_dir/core_fp32_${timestamp}.log"
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite forward log: $log_path" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$lerobot_python" "$script_dir/verify_lingbot_forward_loss.py" 2>&1 | tee "$log_path"
