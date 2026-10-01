#!/usr/bin/env bash
set -euo pipefail

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the pinned RoboTwin checkout}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:?Activate the RoboTwin conda environment}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"

expected_robotwin_commit="13c3c47ff4312dd62484bcd51be034af55c062d1"
expected_curobo_commit="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

actual_robotwin_commit="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
actual_curobo_commit="$(git -C "$ROBOTWIN_ROOT/envs/curobo" rev-parse HEAD)"

[[ "$actual_robotwin_commit" == "$expected_robotwin_commit" ]] || {
  echo "Unexpected RoboTwin commit: $actual_robotwin_commit" >&2
  exit 2
}
[[ "$actual_curobo_commit" == "$expected_curobo_commit" ]] || {
  echo "Unexpected Curobo commit: $actual_curobo_commit" >&2
  exit 2
}

python - <<'PY'
import importlib
import os
import sys

import mplib
import open3d
import pytorch3d
import sapien
import torch
import warp


expected_gpu_count = int(os.environ["EXPECTED_GPU_COUNT"])
versions = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "sapien": sapien.__version__,
    "mplib": mplib.__version__,
    "open3d": open3d.__version__,
    "pytorch3d": pytorch3d.__version__,
    "warp": warp.__version__,
}
for name, version in versions.items():
    print(f"{name}={version}")

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+", 1)[0] == "2.4.1", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert sapien.__version__ == "3.0.0b1", sapien.__version__
assert mplib.__version__ == "0.2.1", mplib.__version__
assert open3d.__version__ == "0.18.0", open3d.__version__
assert pytorch3d.__version__ == "0.7.8", pytorch3d.__version__
assert warp.__version__ == "1.12.0", warp.__version__
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == expected_gpu_count, torch.cuda.device_count()

compiled_extensions = (
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
)
for module_name in compiled_extensions:
    importlib.import_module(module_name)
    print(f"cuda_extension_ok={module_name}")

value = torch.ones((128, 128), device="cuda:0", dtype=torch.float32)
result = value @ value
torch.cuda.synchronize(0)
assert float(result[0, 0]) == 128.0
print("robotwin_gpu_smoke=ok")
PY

render_output="$(cd "$ROBOTWIN_ROOT" && python script/test_render.py 2>&1)"
printf '%s\n' "$render_output"
grep -Fq "Render Well" <<<"$render_output" || {
  echo "RoboTwin render smoke test did not report success." >&2
  exit 3
}

echo "ROBOTWIN_ENV_VALIDATION_OK"
