#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot checkout}"
: "${LEROBOT_V2_ROOT:?Set LEROBOT_V2_ROOT to the pinned LeRobot v2 checkout}"
: "${LEROBOT_V2_ENV:?Set LEROBOT_V2_ENV to the lingbotvla environment}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the audited one-task clean manifest}"
: "${NORM_STATS_PATH:?Set NORM_STATS_PATH to its clean-only normalization statistics}"
: "${MODEL_PATH:?Set MODEL_PATH to the base LingBot-VLA 2.0 checkpoint}"
: "${QWEN3VL_PATH:?Set QWEN3VL_PATH to Qwen3-VL-4B-Instruct}"
: "${MOGE_PATH:?Set MOGE_PATH to the pinned MoGe model.pt}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to a directory outside Git}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "E000 requires exactly four visible GPUs; got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ')" -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES contains duplicate GPU indices" >&2
  exit 2
fi

gpu_table="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
for gpu_id in "${gpu_ids[@]}"; do
  used_mib="$(awk -F',' -v wanted="$gpu_id" '$1 + 0 == wanted {gsub(/ /, "", $2); print $2}' <<< "$gpu_table")"
  if [[ -z "$used_mib" ]]; then
    echo "GPU index $gpu_id was not reported by nvidia-smi" >&2
    exit 2
  fi
  if (( used_mib > 1024 )); then
    echo "GPU $gpu_id is busy (${used_mib} MiB used); refusing to interfere" >&2
    exit 5
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
overrides="$repo_root/configs/competition/E000_training_step_smoke_overrides.yaml"
timestamp="$(date +%Y%m%dT%H%M%S)"
run_dir="$OUTPUT_BASE/training_step_smoke/E000_training_step_smoke_${timestamp}"

"$LEROBOT_V2_ENV/bin/python" "$script_dir/prepare_lingbot_training_step_smoke.py" \
  --output-dir "$run_dir" \
  --overrides "$overrides"

export COMPETITION_CONFIG="$run_dir/config.yaml"
export TRAIN_MANIFEST="$run_dir/train_manifest.txt"
export OUTPUT_DIR="$run_dir"
export ALLOW_PREPARED_OUTPUT=1
export EXPECTED_GPU_COUNT=4

"$script_dir/train_competition.sh"

hf_path="$run_dir/checkpoints/global_step_1/hf_ckpt"
reload_output="$run_dir/reload"
mkdir -p "$reload_output"
reload_gpu="${gpu_ids[0]}"
CUDA_VISIBLE_DEVICES="$reload_gpu" \
MODEL_PATH="$hf_path" \
OUTPUT_BASE="$reload_output" \
"$script_dir/verify_lingbot_model_load.sh"

"$LEROBOT_V2_ENV/bin/python" "$script_dir/finalize_lingbot_training_step_smoke.py" \
  --run-dir "$run_dir"

echo "run_dir=$run_dir"
