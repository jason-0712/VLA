#!/usr/bin/env bash
set -euo pipefail

# Keep CUDA indices aligned with nvidia-smi PCI bus ordering.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

usage() {
  cat <<'EOF'
Usage: scripts/train_competition.sh

Required environment variables:
  LINGBOT_VLA_ROOT   Pinned LingBot-VLA 2.0 checkout
  LEROBOT_V2_ROOT    Pinned LeRobot v2 checkout
  LEROBOT_V2_ENV     Python environment containing torchrun
  COMPETITION_CONFIG Absolute path to an immutable experiment YAML
  TRAIN_MANIFEST     Clean-only training manifest
  OUTPUT_DIR         New output directory, or an existing run with ALLOW_RESUME=1

Optional:
  ALLOW_RESUME=1     Permit an existing OUTPUT_DIR for an intentional resume
  ALLOW_PREPARED_OUTPUT=1
                     Permit an existing prepared directory with no train.log
  EXPECTED_GPU_COUNT Abort unless exactly this many GPUs are visible
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot-VLA checkout}"
: "${LEROBOT_V2_ROOT:?Set LEROBOT_V2_ROOT to the pinned LeRobot v2 checkout}"
: "${LEROBOT_V2_ENV:?Set LEROBOT_V2_ENV to the training environment}"
: "${COMPETITION_CONFIG:?Set COMPETITION_CONFIG to an immutable YAML snapshot}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the clean-only data manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique run directory}"

[[ -d "$LINGBOT_VLA_ROOT" ]] || { echo "Missing LingBot checkout: $LINGBOT_VLA_ROOT" >&2; exit 2; }
[[ -d "$LEROBOT_V2_ROOT" ]] || { echo "Missing LeRobot v2 checkout: $LEROBOT_V2_ROOT" >&2; exit 2; }
[[ -x "$LEROBOT_V2_ENV/bin/torchrun" ]] || { echo "Missing torchrun: $LEROBOT_V2_ENV/bin/torchrun" >&2; exit 2; }
[[ -f "$COMPETITION_CONFIG" ]] || { echo "Missing config: $COMPETITION_CONFIG" >&2; exit 2; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "Missing training manifest: $TRAIN_MANIFEST" >&2; exit 2; }

if [[ "$TRAIN_MANIFEST" =~ [Rr][Aa][Nn][Dd][Oo][Mm][Ii][Zz][Ee][Dd] ]]; then
  echo "Refusing training manifest whose path contains 'randomized': $TRAIN_MANIFEST" >&2
  exit 3
fi

if grep -Eiq 'randomized' "$TRAIN_MANIFEST"; then
  echo "Refusing training manifest containing randomized data entries: $TRAIN_MANIFEST" >&2
  exit 3
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  if [[ "${ALLOW_RESUME:-0}" != "1" && "${ALLOW_PREPARED_OUTPUT:-0}" != "1" ]]; then
    echo "OUTPUT_DIR already exists. Use a new path, ALLOW_PREPARED_OUTPUT=1, or an intentional ALLOW_RESUME=1." >&2
    exit 4
  fi
  if [[ "${ALLOW_PREPARED_OUTPUT:-0}" == "1" && -e "$OUTPUT_DIR/train.log" ]]; then
    echo "Prepared OUTPUT_DIR already has train.log; refusing to overwrite it." >&2
    exit 4
  fi
else
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="$LEROBOT_V2_ROOT:$LINGBOT_VLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  nproc_per_node="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  nproc_per_node="$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | tr -d ' ')"
fi
if [[ -n "${EXPECTED_GPU_COUNT:-}" && "$nproc_per_node" -ne "$EXPECTED_GPU_COUNT" ]]; then
  echo "Expected $EXPECTED_GPU_COUNT visible GPUs, got $nproc_per_node" >&2
  exit 5
fi

cd "$LINGBOT_VLA_ROOT"
tee_args=("$OUTPUT_DIR/train.log")
if [[ "${ALLOW_RESUME:-0}" == "1" ]]; then
  tee_args=(-a "${tee_args[@]}")
fi
"$LEROBOT_V2_ENV/bin/torchrun" \
  --nnodes="${NNODES:-1}" \
  --nproc-per-node="$nproc_per_node" \
  --node-rank="${NODE_RANK:-0}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-62500}" \
  "$LINGBOT_VLA_ROOT/tasks/vla/train_lingbotvla.py" \
  "$COMPETITION_CONFIG" \
  --data.train_path "$TRAIN_MANIFEST" \
  --train.output_dir "$OUTPUT_DIR" \
  2>&1 | tee "${tee_args[@]}"
