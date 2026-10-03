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
present and content-verified.

### FP32 model-load smoke

The strict single-GPU load smoke passed on 2026-10-02 using
`scripts/verify_lingbot_model_load.sh` and GPU 0 (96 GiB H100). The verifier
constructed the full RoboTwin model from the pinned upstream config, loaded all
six base-model shards through LingBot's post-training weight mapper, and found:

- model class: `LingbotVlaV2Policy`
- parameters: 6,375,906,359 total and trainable
- parameter tensors: 1,672, all CUDA FP32 with none left on `meta`
- action/state dimensions: 55/55
- action-expert MoE: 36 layers, 32 experts, top-4 routing
- load time: 7.423 seconds
- resident allocation: 25,671,997,440 bytes (23.91 GiB)
- peak allocation: 27,228,084,224 bytes (25.36 GiB)

The immutable machine-readable record is outside Git at
`/home/hanyu/lingbot-vla-results/model_load_smoke/fp32_gpu0_20261002T110440.json`.
The load-only scope did not execute a forward pass or training.

### Core VLA FP32 forward/loss smoke

The first GPU loss sub-gate passed on 2026-10-02 using
`scripts/verify_lingbot_forward_loss.sh` on one 80 GiB H100. The verifier loaded
the full official model topology and weights, then disabled auxiliary
depth/video loss dispatch after construction so the audited clean batch could
exercise the primary flow-matching action objective in isolation. It used
explicit seed-42 noise, flow time 0.5, evaluation mode, and FP32 throughout.

Official fused-MoE result:

- batch: state `[1,55]`, actions `[1,50,55]`, images `[1,3,256,1536]`
- parameters: 6,375,906,359 across 1,672 CUDA FP32 tensors
- total loss: `0.2718366683`
- VLA loss: `0.2707217336`
- sequence-wise/router losses: `0.0010838973` / `0.0000310292`
- model load: 5.210 seconds
- cold/warm/warm forwards: 1.437 / 0.266 / 0.262 seconds
- allocation after load: 25,676,222,464 bytes (23.91 GiB)
- peak forward allocation: 25,852,809,728 bytes (24.08 GiB)
- gradients, optimizer, backward, checkpoint writes: none

The immutable audit is:

```text
/home/hanyu/lingbot-vla-results/forward_loss_smoke/core_fp32_20261002T141224/audit.json
SHA256 f26a4f449babc9cc74c11e4b26b1d5423208b9fdecfff453b57b8b74c7fec879
```

The official `robby_moe_forward` fast inference kernel is not bitwise
deterministic for fixed inputs: the two warm total losses differed by
`0.0011662543`. A diagnostic kept the same fused checkpoint layout but disabled
only that kernel, causing the cold and both warm losses to match exactly at
`0.2706921995`. Its audit is:

```text
/home/hanyu/lingbot-vla-results/forward_loss_smoke/core_fp32_20261002T141250/audit.json
SHA256 5324a41a1ebcc10a3a67697c327af00a3cfc51d6ede2cdcbe39dffbf9b975384
```

This fallback is diagnostic, not the reference configuration. The core smoke
did not validate the auxiliary teacher losses, gradients, optimizer state, or
checkpoint export. The following sub-gate covers the auxiliary forward only.

### Complete auxiliary-teacher FP32 forward/loss smoke

The complete auxiliary forward sub-gate passed on 2026-10-03 using
`scripts/verify_lingbot_auxiliary_forward_loss.sh` on GPU 2, an 80 GiB H100.
Only teacher asset paths, the visualization directory, `use_compile=false`, and
`image_augment=false` differed from the pinned training configuration. The
policy remained in train mode and FP32; frozen teacher target generation used
the official BF16 autocast path.

Validated targets and predictions:

- current and future LingBot-Depth features: `[1,256,1024]`
- current and future DINO-Video patch features: `[1,256,1024]`
- policy parameters: 6,375,906,359 FP32 parameters
- frozen teacher parameters: 732,208,426 total across MoGe, MoRGBD, and DINO-Video
- all targets/predictions finite and exactly shape-matched

Loss decomposition:

- VLA: `0.2695782185`
- current depth: `0.0125776147`
- future depth: `0.0130076967`
- current-plus-future video: `0.0031229425`
- sequence-wise: `0.0010836312`
- router z-loss: `0.0000303420`
- total: `0.2994004786`

Memory and timing from the final repeat:

- allocated after policy and all teachers: 28,003,341,824 bytes (26.08 GiB)
- peak allocation: 28,614,363,648 bytes (26.65 GiB)
- target generation: 1.712 seconds
- complete loss forward: 0.419 seconds

The timing is a functional-smoke measurement after host and CUDA kernel caches
had already been warmed by an earlier successful run; it is not a training
throughput benchmark. The immutable final audit is:

```text
/home/hanyu/lingbot-vla-results/forward_loss_smoke/auxiliary_fp32_20261003T133407/audit.json
SHA256 1242a1513ba657f67e3d9d9e63a0fd811bd3a5e471b93ff58adf82e93d372e33
```

The verifier also records hashes for the clean manifest, clean-only statistics,
all three teacher checkpoints, both upstream revisions, the source/runtime
alignment configs, and its own source. This smoke used `torch.no_grad()` and no
optimizer, so gradients, optimizer state, FSDP, export, and reload remain
unvalidated.

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

## LeRobot conversion environment

RoboTwin's `policy/pi0/uv.lock` pins LeRobot 0.1.0 at commit
`a445d9c9da6bea99a8972daa4fe1fdd053d711d2`, whose dataset code reports v2.1.
The RoboTwin simulation environment has no LeRobot installation, while the
LingBot environment contains the newer LeRobot 0.4.2 API. To avoid modifying
either core environment, data conversion uses an isolated venv layered on the
validated LingBot Python and places the pinned source first on `PYTHONPATH`.

The setup is reproducible with `scripts/setup_lerobot_conversion_env.sh`; its
import check asserts that `lerobot.__file__` belongs to the pinned checkout and
that `CODEBASE_VERSION == "v2.1"`. The small additional packages are pinned to
the RoboTwin lock where applicable: Draccus 0.10.0, DeepDiff 8.1.1, Tyro 0.9.5,
and Termcolor 2.5.0.

The same interpreter/source overlay is required for LingBot data loading. The
installed LeRobot 0.4.2 v3 reader rejects the official v2.1 dataset, while the
pinned LingBot code deliberately supports the v2 API as a fallback. Loader
wrappers therefore require `LEROBOT_V2_ROOT` and `LEROBOT_V2_ENV`, prepend the
pinned source on `PYTHONPATH`, and record the imported source file and commit.
This does not modify the LingBot or RoboTwin environments. Its use for an actual
training optimizer path remains gated on the backward/export check.
