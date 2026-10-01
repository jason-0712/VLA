# LingBot-VLA 2.0 Challenge

This repository tracks the clean-only post-training and RoboTwin 2.0 evaluation workflow for the preliminary round of the Ant LingBot Embodied Foundation Model Challenge.

## Current status

- Stage: baseline infrastructure only
- Scope: required online simulation tasks; real-robot demo deferred
- Training data: 50 clean demonstrations for each of 50 tasks (2,500 total)
- Evaluation: 100 clean and 100 randomized rollouts per task (10,000 total)
- Hardware target: 4 x NVIDIA H100
- Deadline: 2026-10-26 24:00 (Asia/Hong_Kong)

No competition training has been launched from this repository yet. The first milestone is to validate the pinned environments, data manifest, and end-to-end evaluation path before starting a full 50-task run.

## Repository map

- `AGENTS.md`: operating constraints, environment pins, and verification gates
- `docs/EXPERIMENT_PLAN.md`: phased competition and research plan
- `docs/DECISIONS.md`: append-only decision log
- `experiments/registry.csv`: one row per executed run
- `configs/competition/`: competition-owned configuration snapshots
- `scripts/train_competition.sh`: guarded clean-only training entry point
- `scripts/eval_competition.sh`: clean/randomized evaluation entry point

## Immediate milestone

1. Pin and build the LingBot-VLA and RoboTwin environments.
2. Validate all 2,500 clean demonstrations and generate the LeRobot dataset plus normalization statistics.
3. Evaluate the released RoboTwin checkpoint on a one-task smoke run to verify infrastructure only.
4. Run one-task overfit and short 50-task clean-only training checks.
5. Launch the first full clean-only baseline only after the above gates pass.
