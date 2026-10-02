#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot source checkout}"
: "${LEROBOT_V2_ROOT:?Set LEROBOT_V2_ROOT to the pinned LeRobot v2 source checkout}"
: "${LEROBOT_V2_ENV:?Set LEROBOT_V2_ENV to the isolated LeRobot v2 environment}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to an audited clean-only manifest}"
: "${NORM_STATS_PATH:?Set NORM_STATS_PATH to a new JSON path outside Git}"
: "${EXPECTED_DATASET_COUNT:?Set EXPECTED_DATASET_COUNT to the manifest row count}"

expected_lingbot_commit="be969b8fd117fb70550c5d4bf4bc328211b5b1b6"
expected_lerobot_commit="a445d9c9da6bea99a8972daa4fe1fdd053d711d2"

actual_lingbot_commit="$(git -C "$LINGBOT_VLA_ROOT" rev-parse HEAD)"
actual_lerobot_commit="$(git -C "$LEROBOT_V2_ROOT" rev-parse HEAD)"
if [[ "$actual_lingbot_commit" != "$expected_lingbot_commit" ]]; then
  echo "LingBot source pin mismatch: $actual_lingbot_commit" >&2
  exit 2
fi
if [[ "$actual_lerobot_commit" != "$expected_lerobot_commit" ]]; then
  echo "LeRobot v2 source pin mismatch: $actual_lerobot_commit" >&2
  exit 2
fi

lerobot_python="$LEROBOT_V2_ENV/bin/python"
if [[ ! -x "$lerobot_python" ]]; then
  echo "Missing LeRobot v2 Python interpreter: $lerobot_python" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "Training manifest does not exist: $TRAIN_MANIFEST" >&2
  exit 2
fi
if grep -qi 'randomized' "$TRAIN_MANIFEST"; then
  echo "Training manifest contains forbidden randomized data: $TRAIN_MANIFEST" >&2
  exit 2
fi
if [[ ! "$EXPECTED_DATASET_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_DATASET_COUNT must be a positive integer" >&2
  exit 2
fi

row_count=0
while read -r data_name dataset_path extra; do
  [[ -z "$data_name" ]] && continue
  if [[ "$data_name" != "robotwin" || -n "${extra:-}" ]]; then
    echo "Invalid manifest row: $data_name $dataset_path ${extra:-}" >&2
    exit 2
  fi
  if [[ "$dataset_path" != /* || ! -d "$dataset_path" ]]; then
    echo "Manifest dataset must be an existing absolute directory: $dataset_path" >&2
    exit 2
  fi
  row_count=$((row_count + 1))
done < "$TRAIN_MANIFEST"
if [[ "$row_count" -ne "$EXPECTED_DATASET_COUNT" ]]; then
  echo "Manifest row count mismatch: $row_count != $EXPECTED_DATASET_COUNT" >&2
  exit 2
fi
unique_path_count="$(awk 'NF {print $2}' "$TRAIN_MANIFEST" | sort -u | wc -l | tr -d ' ')"
if [[ "$unique_path_count" -ne "$row_count" ]]; then
  echo "Training manifest contains duplicate dataset paths" >&2
  exit 2
fi

if [[ -e "$NORM_STATS_PATH" ]]; then
  echo "Refusing to overwrite normalization stats: $NORM_STATS_PATH" >&2
  exit 2
fi
norm_parent="$(dirname -- "$NORM_STATS_PATH")"
mkdir -p "$norm_parent"
work_dir="${NORM_STATS_PATH%.json}_work"
log_path="${NORM_STATS_PATH%.json}.log"
if [[ -e "$work_dir" || -e "$log_path" ]]; then
  echo "Refusing to overwrite normalization work artifacts" >&2
  exit 2
fi
mkdir "$work_dir"

export PYTHONPATH="$LEROBOT_V2_ROOT:$LINGBOT_VLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

cd "$work_dir"
"$lerobot_python" "$LINGBOT_VLA_ROOT/scripts/compute_norm_stats.py" \
  --data.train_path "$TRAIN_MANIFEST" \
  --data.datasets_type vla \
  --data.data_name multi \
  --data.robot_name robotwin \
  --data.robot_config_root "$LINGBOT_VLA_ROOT/configs/robot_configs" \
  --data.norm_path "$NORM_STATS_PATH" \
  --data.num_workers 8 \
  --data.data_ratio_for_norm_compute 1.0 \
  --data.norm_merge_chunk_dim true \
  --train.output_dir "$work_dir" \
  --train.chunk_size 50 \
  --train.micro_batch_size 48 \
  --train.max_steps 1 2>&1 | tee "$log_path"

if [[ ! -s "$NORM_STATS_PATH" ]]; then
  echo "Normalization output is missing or empty: $NORM_STATS_PATH" >&2
  exit 2
fi

sha256sum "$TRAIN_MANIFEST" "$NORM_STATS_PATH"
echo "COMPETITION_NORM_STATS_OK"
