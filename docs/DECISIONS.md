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

## Open decisions

- Organizer ruling on image augmentation and synthetic clean-frame perturbations.
- Whether base-teacher feature distillation on clean frames is explicitly permitted.
- Whether randomized development seeds may be used for checkpoint selection and how final seeds are separated.
- Four-H100 micro-batch, effective batch, learning rate, and maximum-step budget after the 100-step throughput test.
- Final experiment-selection score and acceptable clean-SR trade-off.
