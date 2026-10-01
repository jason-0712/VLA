# Environment Record

## Reference training environment

Validated on 2026-10-01 with the pinned LingBot-VLA 2.0 checkout.

- OS: Ubuntu 24.04 LTS
- GPUs: 4 x NVIDIA H100 (two approximately 96 GiB and two approximately 80 GiB)
- NVIDIA driver: 595.84
- Host-reported CUDA compatibility: 13.2
- Conda environment: `lingbotvla`
- Python: 3.12.14
- PyTorch: 2.8.0+cu128
- PyTorch CUDA runtime: 12.8
- FlashAttention: 2.8.3
- Transformers: 4.57.3
- Accelerate: 1.7.0

The host CUDA compatibility shown by `nvidia-smi` is not the CUDA runtime bundled
with PyTorch. Record both values when diagnosing environment problems.

## Installation inputs

- LingBot-VLA commit: `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`
- Environment installer: `tools/create_train_env.sh` from that commit
- FlashAttention wheel:
  `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- FlashAttention wheel SHA256:
  `f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb`

## Validation

After activating the environment, run:

```bash
bash scripts/verify_lingbot_env.sh
```

The check asserts the pinned Python, PyTorch, CUDA, and FlashAttention versions,
then runs BF16 matrix multiplication on every visible GPU and a real
FlashAttention kernel on GPU 0. Success ends with
`LINGBOT_ENV_VALIDATION_OK`.

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in launchers so CUDA indices match
`nvidia-smi` PCI ordering. This matters on the reference host because the 80 GiB
and 96 GiB H100 variants are mixed.

## Known dependency metadata warnings

The official installer intentionally preserves LingBot's pinned core stack and
installs LeRobot and local depth packages with `--no-deps`. Consequently,
`pip check` reports metadata conflicts for LeRobot, MLflow, protobuf, PyArrow,
and related pins. The installer labels these as warnings. Treat runtime import
or kernel failures as blockers; do not independently upgrade one of these
packages without recording a new environment decision and rerunning validation.
