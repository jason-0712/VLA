# AGENTS.md

## Objective

Maintain a reproducible competition repository for clean-only post-training of LingBot-VLA 2.0 on the 50 RoboTwin 2.0 Aloha-AgileX tasks. Work in small verified increments: environment and evaluation first, then baseline, then one research variable at a time.

## Non-negotiable competition constraints

- Use LingBot-VLA 2.0 as the base model.
- Use the RoboTwin 2.0 Aloha-AgileX embodiment.
- Train on exactly 50 clean demonstrations per task across 50 tasks (2,500 episodes total).
- Never train on randomized demonstrations.
- Evaluate both `demo_clean` and `demo_randomized`, 100 valid rollouts per task and setting.
- Preserve raw rollout logs, seeds, per-task results, and representative failure videos.
- Do not present the released RoboTwin post-trained checkpoint as a valid competition model: the official recipe trains it with clean and randomized data. It may be used only to diagnose the evaluation pipeline.

Any ambiguity about image augmentation, synthetic perturbation, external data, or initialization from a task-specific post-trained checkpoint must be resolved in writing with the organizer before use in a scored model.

## Pinned upstream revisions

- LingBot-VLA 2.0: `Robbyant/lingbot-vla-v2` commit `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` (observed 2026-10-01).
- RoboTwin 2.0: commit `13c3c47ff4312dd62484bcd51be034af55c062d1`, the revision specified by the LingBot RoboTwin setup guide.

Record any later upstream update as a new decision. Do not silently move either pin during an experiment series.

## Environment contract

- Training/inference reference: Python 3.12, PyTorch 2.8.0, FlashAttention 2.8.3.
- Keep separate Conda environments for LingBot inference/training and RoboTwin simulation.
- Hardware target: 4 x H100. Start evaluation with one FP32 inference server per GPU.
- Official release validation uses FP32 inference. BF16 is permitted for smoke tests only and must be labeled; never compare BF16 and FP32 success rates as if they were the same protocol.
- Store machine-specific paths in environment variables or an untracked `.env`, not committed YAML or scripts.

Required path variables for wrappers:

- `LINGBOT_VLA_ROOT`
- `ROBOTWIN_ROOT`
- `QWEN3VL_PATH`
- `CONDA_SH`
- `TRAIN_MANIFEST` and `COMPETITION_CONFIG` for training
- `MODEL_PATH` and `OUTPUT_BASE` for evaluation

## Data guardrails

- A training manifest path or manifest entry containing `randomized` is a hard error.
- Before training, produce a data audit with exactly 50 tasks and 50 clean episodes per task.
- Verify camera keys, state/action dimensions, episode lengths, NaN/Inf counts, timestamp alignment, and normalization-stat provenance.
- Hash the manifest and normalization statistics; record both in the experiment registry or run metadata.
- Do not modify or regenerate source demonstrations in place.

## Experiment protocol

Every executed experiment gets:

1. A unique ID such as `E000-baseline-smoke`.
2. A git commit and clean/dirty status.
3. An immutable config snapshot.
4. Seed, data-manifest hash, normalization-stat hash, base checkpoint, and upstream commits.
5. Output and checkpoint paths outside Git.
6. Training time, peak GPU memory, total/trainable parameters, and inference latency.
7. Clean SR, randomized SR, gap (`clean - randomized`), retention (`randomized / clean`), and per-task counts.
8. Raw logs and failure-case references.

Append runs to `experiments/registry.csv`; do not rewrite old results. If a run is invalid or aborted, keep it and record the reason.

## Verification ladder

Do not skip levels:

1. Static checks: config parses, imports work, data audit passes, model and robot dimensions match.
2. Open-loop check: one batch forward/loss and checkpoint load/export.
3. One-task evaluation smoke: released reference checkpoint, then local checkpoint.
4. One-task overfit: prove the training path can reduce loss and execute actions.
5. Short 50-task run: benchmark throughput, memory, loss stability, and checkpoint export.
6. Full clean-only baseline.
7. Sentinel closed-loop evaluation with fixed seeds.
8. Full 10,000-rollout evaluation only for the validated baseline and finalists.

## Method-development rule

- Establish a clean-only full-fine-tuning baseline before adding a method.
- Change one causal factor per comparison.
- First ablations should use existing controls: full fine-tuning, action-expert-only, and expert-first then full fine-tuning.
- Add feature/weight anchoring, LoRA, augmentation, or architectural changes only after simpler comparisons identify the failure mode.
- Never select a method from training loss alone; require closed-loop evidence.

## Results and submission

- Never overwrite an evaluation directory. Use experiment ID, checkpoint step, setting, seed, precision, and timestamp in its path.
- Generate the official `results.json` from parsed raw logs; do not hand-edit success counts.
- Validate exact task names, `attempts == 100`, `0 <= successes <= attempts`, and both settings before packaging.
- Keep large checkpoints and data out of Git. Submission packaging should copy only the selected checkpoint and required materials into a separate staging directory.
- Document every modification to the upstream LingBot-VLA or RoboTwin evaluation code.

## Current stop condition

The next action is environment/data/evaluation validation. Do not start a full 50-task training job until all earlier verification gates pass and a 100-step throughput measurement provides a defensible training budget.
