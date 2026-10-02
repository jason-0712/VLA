#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot source checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the LingBot-VLA 2.0 base-model snapshot}"
: "${QWEN3VL_PATH:?Set QWEN3VL_PATH to the Qwen3-VL-4B-Instruct snapshot}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to an output directory outside Git}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$script_dir/verify_lingbot_model_load.py"
