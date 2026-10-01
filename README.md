This repository tracks the clean-only post-training and RoboTwin 2.0 evaluation workflow for the preliminary round of the Ant LingBot Embodied Foundation Model Challenge.

## Current status

- Stage: baseline infrastructure only
- Scope: required online simulation tasks; real-robot demo deferred
- Training data: 50 clean demonstrations for each of 50 tasks (2,500 total)
- Evaluation: 100 clean and 100 randomized rollouts per task (10,000 total)

## Repository map

- `AGENTS.md`: operating constraints, environment pins, and verification gates
- `docs/EXPERIMENT_PLAN.md`: phased competition and research plan
- `docs/DECISIONS.md`: append-only decision log
- `experiments/registry.csv`: one row per executed run
- `configs/competition/`: competition-owned configuration snapshots
- `scripts/train_competition.sh`: guarded clean-only training entry point
- `scripts/eval_competition.sh`: clean/randomized evaluation entry point

