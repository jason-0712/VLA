#!/usr/bin/env bash
set -euo pipefail

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the pinned RoboTwin checkout}"

readonly task_name="${TASK_NAME:-beat_block_hammer}"
readonly gpu_id="${GPU_ID:-0}"
readonly config_name="robotwin_smoke_clean"
readonly expected_robotwin_commit="13c3c47ff4312dd62484bcd51be034af55c062d1"
readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly source_config="${project_root}/configs/competition/${config_name}.yml"
readonly deployed_config="${ROBOTWIN_ROOT}/task_config/${config_name}.yml"
readonly output_dir="${ROBOTWIN_ROOT}/data/${task_name}/${config_name}"

actual_robotwin_commit="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
[[ "$actual_robotwin_commit" == "$expected_robotwin_commit" ]] || {
  echo "Unexpected RoboTwin commit: $actual_robotwin_commit" >&2
  exit 2
}

[[ -f "$source_config" ]] || {
  echo "Missing smoke config: $source_config" >&2
  exit 2
}

[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite existing smoke output: $output_dir" >&2
  exit 3
}

cp "$source_config" "$deployed_config"
cmp --silent "$source_config" "$deployed_config"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$gpu_id"

gpu_capability="$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))))')"
if [[ "$gpu_capability" == "9.0" ]]; then
  export ROBOTWIN_DISABLE_CUROBO_FUSED_LBFGS=1
  export PYTHONPATH="${project_root}/compat${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Using the Curobo fused-LBFGS fallback for compute capability ${gpu_capability}."
fi

(
  cd "$ROBOTWIN_ROOT"
  PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py "$task_name" "$config_name"
)

readonly episode_file="${output_dir}/data/episode0.hdf5"
readonly seed_file="${output_dir}/seed.txt"
[[ -s "$episode_file" ]] || {
  echo "Missing or empty smoke episode: $episode_file" >&2
  exit 4
}
[[ "$(wc -w < "$seed_file")" -eq 1 ]] || {
  echo "Expected exactly one successful seed in: $seed_file" >&2
  exit 4
}

python - "$episode_file" <<'PY'
import sys

import h5py


with h5py.File(sys.argv[1], "r") as episode:
    assert len(episode.keys()) > 0, "HDF5 episode has no top-level groups"
    print("hdf5_top_level=" + ",".join(sorted(episode.keys())))
PY

echo "ROBOTWIN_CLEAN_SMOKE_OK task=${task_name} episode=${episode_file}"
