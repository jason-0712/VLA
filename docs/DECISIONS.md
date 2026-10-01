# Decision Log

Decisions are append-only. A superseding decision references the previous ID instead of editing history.

## D-001 - Pin upstream revisions

- Date: 2026-10-01
- Status: accepted
- Decision: pin LingBot-VLA 2.0 to `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` and RoboTwin 2.0 to `13c3c47ff4312dd62484bcd51be034af55c062d1` for the first baseline series.
- Reason: the LingBot setup guide explicitly names the RoboTwin revision, and the LingBot repository changed shortly before project start.
- Revisit when: an organizer announcement requires another revision or an upstream blocker has a verified fix.

## D-002 - Strict clean-only training provenance

- Date: 2026-10-01
- Status: accepted
- Decision: reject any training manifest whose path or contents mention `randomized`; compute normalization statistics from clean data only.
- Reason: randomized demonstrations are prohibited for training, and data provenance must be auditable.

## D-003 - Released RoboTwin checkpoint is infrastructure-only

- Date: 2026-10-01
- Status: accepted
- Decision: use `lingbot-vla-v2-6b-robotwin` only to validate inference and evaluation infrastructure. Do not initialize a scored model from it unless the organizer explicitly approves it.
- Reason: the official repository describes that checkpoint's post-training recipe as using clean and randomized data together.

## D-004 - Baseline before research variants

- Date: 2026-10-01
- Status: accepted
- Decision: validate environment, data, evaluation, one-task overfit, and a short 50-task run before launching the full clean-only baseline; evaluate the baseline before implementing representation-preservation methods.
- Reason: closed-loop baseline behavior and compute throughput are currently unknown.

## D-005 - FP32 is the reporting reference

- Date: 2026-10-01
- Status: accepted
- Decision: use FP32 policy inference for reported checkpoint comparisons and final results. BF16 is smoke-test-only and must be labeled.
- Reason: the official LingBot RoboTwin guide warns that BF16 can materially change success rate.

## D-006 - Research method order

- Date: 2026-10-01
- Status: provisional
- Decision: test full fine-tuning, expert-only, and expert-first/full fine-tuning before adding explicit regularization or architectural changes.
- Reason: the upstream code already exposes `train_expert_only`, making these the cheapest causal probes of representation degradation.
- Revisit when: the first full clean-only baseline and representation-drift measurements are available.

## D-007 - Reference environment and CUDA device ordering

- Date: 2026-10-01
- Status: accepted
- Decision: use the official `lingbotvla` environment with Python 3.12, PyTorch 2.8.0+cu128, and FlashAttention 2.8.3; set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in competition launchers.
- Reason: all four H100s and an actual FlashAttention kernel passed validation. The host mixes approximately 80 GiB and 96 GiB H100 variants, and PyTorch's default enumeration did not match `nvidia-smi` ordering.
- Revisit when: the pinned upstream environment changes or the execution host changes.

## D-008 - Isolate and pin the RoboTwin simulation stack

- Date: 2026-10-01
- Status: accepted
- Decision: use a separate `RoboTwin` Conda environment with Python 3.10, PyTorch 2.4.1+cu121, a Conda-local CUDA 12.1 toolkit, GCC/G++ 11.4, and Curobo v0.7.8 at `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`.
- Reason: the pinned simulator stack is incompatible with the LingBot training environment, and Curobo requires `nvcc` to compile its CUDA extensions. Keeping the compiler in Conda avoids modifying the shared host installation.
- Validation: all required imports, five Curobo CUDA extensions, a GPU operation, and a headless SAPIEN render smoke test passed.
- Revisit when: RoboTwin or LingBot requires a different simulator revision.

## D-009 - Pin and verify RoboTwin simulator assets

- Date: 2026-10-01
- Status: accepted
- Decision: download the three required simulator asset archives from `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`, verify their exact sizes and SHA256 hashes, and separate download from extraction.
- Reason: the upstream helper follows a mutable dataset `main`, immediately deletes archives after extraction, and does not record content hashes. Keeping verified archives until the first simulator validation makes provenance and recovery auditable.
- Storage: keep simulator assets with the ignored RoboTwin checkout on the root filesystem; reserve `/mnt/data1` for raw and converted competition demonstrations.

## D-010 - Disable the fused Curobo LBFGS kernel on Hopper

- Date: 2026-10-01
- Status: accepted
- Decision: on compute capability 9.0, keep the Curobo LBFGS optimizer but disable its fused CUDA implementation through an opt-in Python startup hook. Do not modify the pinned RoboTwin or Curobo checkout.
- Reason: the first task-level smoke test failed deterministically in `lbfgs_step_cu.forward` with CUDA error 715. A synchronous single-seed trace isolated the failure to `MotionGen.warmup()`. [RoboTwin issue #452](https://github.com/RoboTwin-Platform/RoboTwin/issues/452) documents the same failure for the same PyTorch/CUDA/Curobo stack on an H800 and the same fallback; the local minimal warmup passed with the fused implementation disabled.
- Scope: infrastructure compatibility only. The fallback is enabled by `scripts/run_robotwin_clean_smoke.sh` after detecting capability 9.0 and must also be applied consistently to any expert-planning path on this host.
- Revisit when: Curobo ships a validated sm90 fused-kernel fix, or the execution GPU is not Hopper.

## Open decisions

- Organizer ruling on image augmentation and synthetic clean-frame perturbations.
- Whether base-teacher feature distillation on clean frames is explicitly permitted.
- Whether randomized development seeds may be used for checkpoint selection and how final seeds are separated.
- Four-H100 micro-batch, effective batch, learning rate, and maximum-step budget after the 100-step throughput test.
- Final experiment-selection score and acceptable clean-SR trade-off.
