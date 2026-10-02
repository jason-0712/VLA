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

## D-011 - Pin model snapshots and separate model/data storage

- Date: 2026-10-02
- Status: accepted
- Decision: pin the base model to `robbyant/lingbot-vla-v2-6b@11c703bf6a5c1f45b3b69168482da11fdbba53d7`, Qwen3-VL to `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`, and MoGe to `Ruicheng/moge-2-vitb-normal@ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`.
- Reason: upstream download helpers follow mutable repository heads. Exact revisions, repository byte totals, per-file sizes, and LFS SHA256 hashes must be checked before a model enters a run.
- Storage: store the 37.55 GB of model snapshots on the root filesystem under an ignored model root. Reserve `/mnt/data1` for the 23.78 GB of clean source archives plus extracted and converted datasets.
- Constraint: `lingbot-vla-v2-6b-robotwin` remains infrastructure-only and is not a valid initialization for scored clean-only training.

## D-012 - Build the clean manifest from pinned source archives

- Date: 2026-10-02
- Status: accepted
- Decision: acquire only the 50 files matching `dataset/<task>/aloha-agilex_clean_50.zip` from `TianxingChen/RoboTwin2.0@3dc3b798668feb99ac61cc9086d84cbcc3d79186`; preserve the verified ZIPs unchanged and extract/convert into separate directories.
- Reason: this selection yields exactly 50 tasks and 23,780,715,316 bytes. The upstream LingBot `assets/training_data/robotwin.txt` is invalid for this competition because its 99 entries include 50 randomized datasets and one clean Piper dataset.
- Validation order: validate one `beat_block_hammer` archive and its episode schema first, then acquire and audit all 50 tasks before conversion.

## D-013 - Make the official LeRobot conversion deterministic

- Date: 2026-10-02
- Status: accepted
- Decision: use RoboTwin's pinned LeRobot `a445d9c9da6bea99a8972daa4fe1fdd053d711d2` and its v2.1 image-mode conversion functions, while externally enforcing numeric episode order and NumPy seed 0 for `seen` instruction selection.
- Reason: the upstream converter collects HDF5 files with unsorted `os.walk` and calls `np.random.choice` without a seed. Both affect episode/prompt provenance without changing the intended data recipe.
- Validation: the one-task result has 50 episodes, 5,682 frames, exact state/action equality, 17,046/17,046 exact image matches, deterministic prompt reproduction, and a successful pinned-reader load.
- Constraint: do not change image mode, FPS metadata, instruction split, action shift, camera set, or resolution during the baseline series without a new decision.

## D-014 - Measure storage before full clean-data acquisition

- Date: 2026-10-02
- Status: provisional
- Decision: pause the remaining 49-task acquisition until a storage layout is chosen from measured conversion sizes; next validate the LingBot training-side loader using the completed one-task dataset.
- Reason: the measured image-mode LeRobot/source ratio is 5.47x. A proportional 50-task projection is about 130.1 GB for LeRobot alone and 239.5 GB while source, raw, processed, and LeRobot stages coexist, exceeding the current `/mnt/data1` free space.
- Supersedes: the bulk-acquisition ordering implied by D-012; one-task conversion is now validated before acquiring the remaining tasks.
- Revisit when: the one-task LingBot loader passes and root/data-volume allocation plus intermediate-retention policy are fixed.

## D-015 - Keep source archives on the Mac and stage conversion per task

- Date: 2026-10-02
- Status: accepted
- Decision: keep the immutable 23.78 GB clean source-archive set on the Mac under an external `CLEAN_DATA_ROOT`; on the server, stage one task at a time on `/mnt/data1` and keep only the audited LeRobot output on the root filesystem. Remove a task's server-side source/raw/processed staging copies only after the Mac ZIP hash and final LeRobot tree hash are recorded. Derived LeRobot data remains reproducible from the pinned ZIPs and conversion code.
- Reason: the Mac has 234 GiB available, while `/mnt/data1` has only 138 GiB available. The projected all-task LeRobot output is about 130.1 GB and all conversion stages together are about 239.5 GB, so retaining every stage on `/mnt/data1` is not viable. The server root has 218 GiB available for the final training dataset.
- Validation: `beat_block_hammer` was downloaded independently on the Mac, passed ZIP CRC, and produced SHA256 `a135ad233bdcffff65fb636780f95ec34abc36ccf2bd0ff26ab07c9c464cc6af`, exactly matching the server copy.
- Supersedes: the storage allocation in D-011 and resolves the layout decision requested by D-014. The one-task LingBot loader remains the next functional gate before bulk acquisition.

## D-016 - Read official v2.1 data through LingBot's pinned v2 compatibility branch

- Date: 2026-10-02
- Status: accepted
- Decision: keep the audited RoboTwin output in LeRobot v2.1 format and run LingBot data-loading commands with pinned LeRobot source `a445d9c9da6bea99a8972daa4fe1fdd053d711d2` first on `PYTHONPATH`, using the isolated v2 environment layered on the validated LingBot Python packages. Do not convert the baseline data to LeRobot v3.
- Reason: the LingBot environment's installed LeRobot 0.4.2 reader rejects v2.1 datasets with `BackwardCompatibilityError`. The pinned LingBot source explicitly implements a v2 import fallback, and that branch loaded all 5,682 one-task frames without changing the already audited conversion output.
- Provenance note: distribution metadata still reports installed LeRobot 0.4.2 because the isolated environment uses LingBot's system packages. Audits must record the imported `lerobot.__file__`, pinned source commit, selected LingBot API branch, and interpreter; package metadata alone is not sufficient.
- Revisit when: the organizer or pinned upstream recipe requires v3, or a v3 conversion is proven byte/semantic-equivalent and adopted as a separately audited data version.

## D-017 - Recompute normalization from clean-only data and inject it without patching upstream

- Date: 2026-10-02
- Status: accepted
- Decision: compute `bounds_99_woclip` statistics with LingBot's official `scripts/compute_norm_stats.py` over the exact clean-only manifest. Generate an immutable runtime copy of `robotwin.yaml` whose only change is the `norm_stats` path; do not use upstream `assets/norm_stats/robotwin.json` and do not edit the pinned checkout.
- Reason: the released statistics belong to the clean-plus-randomized training recipe and violate scored-model provenance. Although `MyDataArguments` exposes `norm_stats_file`, the pinned dataset builder does not pass it to `FeatureTransform`, so a runtime robot-config snapshot is currently the smallest auditable override.
- Validation: one-task clean statistics contain exactly the four expected 12-D arm and 2-D effector state/action entries with state count 5,682. A full processed batch passed with state `[1,55]`, actions `[1,50,55]`, three cameras, 72 language tokens, finite values, and active indices `[0..11,28,29]`.
- Constraint: the one-task statistics are valid only for loader smoke and one-task overfit. Recompute and hash a single all-50-task clean-only statistics file before any short or full 50-task training run.

## D-018 - Validate the core VLA loss before enabling auxiliary teachers

- Date: 2026-10-02
- Status: accepted
- Decision: split the first GPU loss check into two sub-gates. First load the complete official model topology and weights, then disable only auxiliary alignment-loss dispatch after construction and validate the core flow-matching VLA loss in FP32. Validate the depth/video teacher target and loss path separately before any optimizer step.
- Reason: this isolates the already audited clean batch, model input contract, action masks, MoE routing losses, and primary action objective from three additional teacher pipelines. The first sub-gate passed with no optimizer, backward pass, or checkpoint write. It is not evidence that the complete official training loss works.
- Numerical note: fixed-input repeats with the official `robby_moe_forward` inference kernel are not bitwise deterministic. Keeping the same fused checkpoint layout but disabling only that fast kernel made all three forward losses bitwise identical, localizing the observed jitter to the fast MoE inference path or its interaction with the model. The default official kernel remains the reference path; the fallback is diagnostic only and must be recorded if used.
- Next gate: run the complete auxiliary teacher/target forward in FP32, then one minimal backward/optimizer/checkpoint-export-and-reload check before evaluation or overfit training.

## Open decisions

- Organizer ruling on image augmentation and synthetic clean-frame perturbations.
- Whether base-teacher feature distillation on clean frames is explicitly permitted.
- Whether randomized development seeds may be used for checkpoint selection and how final seeds are separated.
- Four-H100 micro-batch, effective batch, learning rate, and maximum-step budget after the 100-step throughput test.
- Final experiment-selection score and acceptable clean-SR trade-off.
