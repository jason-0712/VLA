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

## Pinned model snapshots

The reference host validated the base snapshot on 2026-10-02:

- Repository: `robbyant/lingbot-vla-v2-6b`
- Revision: `11c703bf6a5c1f45b3b69168482da11fdbba53d7`
- Location: `/home/hanyu/VLA/work/models/lingbot-vla-v2-6b`
- Files: 23, totaling 28,239,981,618 bytes
- LFS content verified by SHA256: 28,236,983,562 bytes
- Weight structure: 6 readable safetensors shards and 1,708 indexed tensors
- Bundled teachers: `depth/model.pt` and
  `dino_video/teacher_step_10000.pth` with its config

Use `scripts/download_lingbot_model_assets.sh` for pinned, resumable downloads.
The existing Qwen3-VL snapshot from the StarVLA workspace was reused rather
than duplicated:

- Repository: `Qwen/Qwen3-VL-4B-Instruct`
- Revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Location:
  `/home/hanyu/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct`
- Files: 14, totaling 8,887,292,732 bytes
- Weight structure: 2 readable safetensors shards and 713 indexed tensors

All Qwen file sizes and LFS SHA256 hashes were revalidated on 2026-10-02.

The MoGe snapshot was downloaded and validated separately:

- Repository: `Ruicheng/moge-2-vitb-normal`
- Revision: `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`
- Location: `/home/hanyu/VLA/work/models/moge-2-vitb-normal`
- Files: 3, totaling 419,111,765 bytes
- LFS content verified by SHA256: 419,110,160 bytes

The base, Qwen3-VL, MoGe, LingBot-Depth, and DINO-Video model assets are now
present and content-verified. This does not replace the model-load and
one-batch forward/loss verification gates.

## Reference RoboTwin simulation environment

Validated on 2026-10-01 against RoboTwin commit
`13c3c47ff4312dd62484bcd51be034af55c062d1`.

- Conda environment: `RoboTwin`
- Python: 3.10.21
- PyTorch: 2.4.1+cu121
- CUDA toolkit and runtime: 12.1
- Conda GCC/G++: 11.4
- SAPIEN: 3.0.0b1
- MPLib: 0.2.1
- Open3D: 0.18.0
- PyTorch3D: 0.7.8
- Curobo tag: v0.7.8
- Curobo commit: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- Warp: 1.12.0

The Curobo CUDA extensions were compiled for Hopper (`TORCH_CUDA_ARCH_LIST=9.0`).
All five compiled extensions imported successfully, a CUDA tensor operation ran,
and the pinned RoboTwin `script/test_render.py` reported `Render Well` on the
headless reference host.

After activating the simulation environment, run:

```bash
ROBOTWIN_ROOT=/path/to/pinned/RoboTwin bash scripts/verify_robotwin_env.sh
```

Success ends with `ROBOTWIN_ENV_VALIDATION_OK`.

## RoboTwin asset snapshot

The first baseline series pins the `TianxingChen/RoboTwin2.0` dataset repository
to revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`. The required simulator
archives are:

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| `background_texture.zip` | 10,970,687,027 | `54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070` |
| `embodiments.zip` | 219,847,741 | `85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789` |
| `objects.zip` | 3,737,778,549 | `6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27` |

Use `scripts/download_robotwin_assets.sh --download-only` first. Extract only
after all three byte counts, SHA256 hashes, and ZIP integrity checks pass.

The reference host completed extraction on 2026-10-01 under
`/home/hanyu/VLA/work/RoboTwin/assets`. The extracted snapshot contains 11,000
background-texture files, 229 embodiment files, and 9,368 object files. The
three verified archives are retained alongside the extracted directories for
recovery. The generated Aloha-AgileX `curobo_left.yml` and `curobo_right.yml`
contain resolved absolute paths and no `${ASSETS_PATH}` placeholders. Both the
competition repository and pinned RoboTwin checkout remained clean after
extraction.

## Task-level RoboTwin smoke test

The one-episode `beat_block_hammer` clean smoke test passed on 2026-10-01 with
config `configs/competition/robotwin_smoke_clean.yml` at project commit
`c7143e0`. Seed 0 was a normal planning failure and seed 1 succeeded. The
successful trajectory contains 1,543 frames, four readable RGB observation
streams, 14-D joint actions, and left/right end-effector poses. Its HDF5 file is
83,217,903 bytes with SHA256
`f769fd69889effc704d7298b931f9815a7bb1180917b380226b83904d93cdd46`.
The generated 320 x 240 diagnostic video also contains 1,543 frames.

The sm90 fused-LBFGS compatibility hook described in D-010 was enabled. The
smoke output is infrastructure-only and must not enter the competition training
manifest.
