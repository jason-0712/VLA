#!/usr/bin/env bash
set -euo pipefail

: "${LEROBOT_ROOT:?Set LEROBOT_ROOT to the pinned LeRobot source checkout path}"
: "${LEROBOT_VENV:?Set LEROBOT_VENV to the isolated conversion venv path}"
: "${LINGBOT_PYTHON:?Set LINGBOT_PYTHON to the validated lingbotvla Python executable}"
export LEROBOT_ROOT

repo_url="https://github.com/huggingface/lerobot.git"
revision="a445d9c9da6bea99a8972daa4fe1fdd053d711d2"

if [[ ! -e "$LEROBOT_ROOT" ]]; then
  git clone --filter=blob:none "$repo_url" "$LEROBOT_ROOT"
  git -C "$LEROBOT_ROOT" checkout "$revision"
fi

actual_revision="$(git -C "$LEROBOT_ROOT" rev-parse HEAD)"
[[ "$actual_revision" == "$revision" ]] || {
  echo "LeRobot revision mismatch: $actual_revision != $revision" >&2
  exit 1
}

if [[ ! -x "$LEROBOT_VENV/bin/python" ]]; then
  "$LINGBOT_PYTHON" -m venv --system-site-packages "$LEROBOT_VENV"
fi

"$LEROBOT_VENV/bin/python" -m pip install --disable-pip-version-check \
  draccus==0.10.0 \
  deepdiff==8.1.1 \
  tyro==0.9.5 \
  termcolor==2.5.0

PYTHONPATH="$LEROBOT_ROOT" "$LEROBOT_VENV/bin/python" - <<'PY'
from pathlib import Path

import lerobot
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset


expected_root = Path(__import__("os").environ["LEROBOT_ROOT"]).resolve()
actual_file = Path(lerobot.__file__).resolve()
if not actual_file.is_relative_to(expected_root):
    raise RuntimeError(f"Imported the wrong LeRobot package: {actual_file}")
if CODEBASE_VERSION != "v2.1":
    raise RuntimeError(f"Expected LeRobot codebase v2.1, got {CODEBASE_VERSION}")
print(f"lerobot_file={actual_file}")
print(f"lerobot_codebase_version={CODEBASE_VERSION}")
print(f"lerobot_dataset_class={LeRobotDataset.__name__}")
print("LEROBOT_CONVERSION_ENV_OK")
PY
