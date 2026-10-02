#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot source checkout}"
: "${LEROBOT_V2_ROOT:?Set LEROBOT_V2_ROOT to the pinned LeRobot v2 source checkout}"
: "${LEROBOT_V2_ENV:?Set LEROBOT_V2_ENV to the isolated LeRobot v2 environment}"
: "${TRAIN_DATASET_PATH:?Set TRAIN_DATASET_PATH to one audited clean LeRobot dataset}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to an output directory outside Git}"

lerobot_python="$LEROBOT_V2_ENV/bin/python"
if [[ ! -x "$lerobot_python" ]]; then
  echo "Missing LeRobot v2 Python interpreter: $lerobot_python" >&2
  exit 2
fi

case "$TRAIN_DATASET_PATH" in
  *randomized*|*Randomized*|*RANDOMIZED*)
    echo "Refusing forbidden randomized dataset path: $TRAIN_DATASET_PATH" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$LEROBOT_V2_ROOT:$LINGBOT_VLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$lerobot_python" "$script_dir/verify_lingbot_data_loader.py" "$@"
