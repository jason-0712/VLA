#!/usr/bin/env bash
set -euo pipefail

# Keep CUDA indices aligned with nvidia-smi PCI bus ordering.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
export EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-H100}"

python - <<'PY'
import os
import sys

import accelerate
import flash_attn
import torch
import transformers
from flash_attn import flash_attn_func


expected_gpu_count = int(os.environ["EXPECTED_GPU_COUNT"])
expected_gpu_substring = os.environ["EXPECTED_GPU_SUBSTRING"]

versions = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
}
for name, version in versions.items():
    print(f"{name}={version}")

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == expected_gpu_count, torch.cuda.device_count()

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    assert expected_gpu_substring in properties.name, properties.name
    device = f"cuda:{index}"
    value = torch.ones((128, 128), device=device, dtype=torch.bfloat16)
    result = value @ value
    torch.cuda.synchronize(index)
    memory_mib = properties.total_memory // (1024**2)
    print(
        f"gpu[{index}]={properties.name} "
        f"memory_mib={memory_mib} capability={properties.major}.{properties.minor} "
        f"bf16_smoke={float(result[0, 0])}"
    )

query = torch.randn((1, 16, 4, 64), device="cuda:0", dtype=torch.bfloat16)
output = flash_attn_func(query, query, query)
torch.cuda.synchronize(0)
assert torch.isfinite(output).all(), "FlashAttention produced non-finite output"
print(f"flash_attn_smoke=ok shape={tuple(output.shape)} dtype={output.dtype}")
print("LINGBOT_ENV_VALIDATION_OK")
PY
