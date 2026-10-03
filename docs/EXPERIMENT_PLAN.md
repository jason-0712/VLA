# Experiment Plan

Last updated: 2026-10-03

## 1. Goal and research question

Competition objective: maximize clean and randomized success rates on all 50 RoboTwin 2.0 Aloha-AgileX tasks while using only the 2,500 clean demonstrations for post-training.

Research question:

> When a VLA is post-trained only on a narrow clean-demonstration distribution, can we preserve useful pretrained visual-semantic representations while still learning the downstream action distribution, thereby reducing the clean-to-randomized generalization gap?

The project should produce both a competitive checkpoint and an interpretable study of the clean-performance/retention trade-off.

## 2. Primary hypotheses

- H0: unrestricted full fine-tuning gives strong clean SR but causes measurable pretrained-feature drift and a larger randomized gap.
- H1: training the action expert while freezing the VLM preserves randomized robustness but may underfit task grounding.
- H2: action-expert-first training followed by low-rate full fine-tuning reaches a better clean/randomized Pareto point than either full fine-tuning or expert-only training.
- H3: if H2 still shows destructive drift, explicit anchoring to the base checkpoint (weight or feature regularization) improves retention.
- H4: checkpoint interpolation with the base model can recover robustness without another training run, provided architecture and parameter semantics remain compatible.

Image augmentation is not part of the initial study. It becomes eligible only after the organizer confirms that synthetic image perturbations of clean demonstrations do not count as randomized training data.

## 3. Evaluation design

### Official metrics

- Clean SR: total clean successes / 5,000 clean attempts.
- Randomized SR: total randomized successes / 5,000 randomized attempts.
- Generalization gap: Clean SR - Randomized SR.
- Retention: Randomized SR / Clean SR. Report `N/A` if Clean SR is zero.
- Per-task attempts, successes, and SR for both settings.
- Peak training VRAM, wall-clock time, total/trainable parameters, and policy latency.

### Diagnostic metrics

- Per-layer representation drift on a fixed clean probe set: cosine similarity and linear CKA between base and adapted activations.
- Parameter delta norms by module: vision encoder, VLM, action expert, depth/video alignment modules, MoE router/experts.
- Gradient norms by module during short runs.
- Open-loop normalized action error on held-out clean episodes.
- Failure taxonomy: perception/grounding, approach, grasp, transport, release, bimanual coordination, premature termination, oscillation.
- One-factor-at-a-time evaluation for background, clutter, lighting, table height, and language. These are diagnostic evaluation configs, never training data, and are reported separately from official combined-randomized results.

### Evaluation ladder

Use fixed development seeds and identical episode seeds for paired comparisons where possible.

1. API smoke: one batch, one inference request, checkpoint export/reload.
2. One-task smoke: stop after successful end-to-end requests; FP32 for any reported comparison.
3. Sentinel set: 8 tasks chosen after baseline failure analysis, 20 episodes per setting for coarse screening.
4. Expanded validation: 20 tasks, 40 episodes per setting for finalists.
5. Official evaluation: 50 tasks x 100 episodes x 2 settings = 10,000 rollouts.

For small-n screens, attach Wilson intervals and avoid treating differences below sampling noise as wins. Use paired outcome analysis when the same seeds are available.

## 4. Phased pipeline and gates

### Phase 0 - Rule and version freeze (Oct 1-2)

- Archive the official task list, result schema, material templates, and organizer clarifications.
- Pin LingBot-VLA and RoboTwin commits.
- Ask the organizer whether photometric/geometric augmentation, base-teacher feature distillation on clean frames, and randomized validation for model selection are allowed.

Gate: written rule matrix with allowed, forbidden, and unresolved techniques.

### Phase 1 - Environment and data audit (Oct 2-4)

- Build separate LingBot and RoboTwin environments.
- Download base model and required Qwen3-VL, MoGe/LingBot-Depth, and DINO-Video teacher assets.
- Strictly load the full 6.376B-parameter model in single-GPU FP32 and verify
  architecture dimensions, dtype, device placement, and peak memory. Completed
  2026-10-02; see `docs/ENVIRONMENT.md`.
- Acquire only the required clean demonstrations for training.
- Convert HDF5 to LeRobot v2.1, verify 50 x 50 episodes, and compute clean-only normalization statistics.
- Record hashes and a storage estimate before conversion.

The one-task source, conversion, and training-loader gates for
`beat_block_hammer` passed on 2026-10-02: the pinned archive, safe extraction,
50 raw episodes, deterministic LeRobot v2.1 conversion, clean-only
normalization, and the complete 55-D LingBot CPU batch were validated. The
storage layout is now fixed across the Mac, `/mnt/data1` staging, and the server
root filesystem.

The teacher-free core-VLA FP32 forward/loss sub-gate also passed on 2026-10-02
with the complete official model topology and fixed explicit noise/time. The
official fused MoE path produced finite, decomposable losses and required about
24.08 GiB peak allocated memory. A diagnostic localized small fixed-input
repeat jitter to the `robby_moe_forward` fast inference kernel. No optimizer,
backward pass, or checkpoint write occurred.

The complete official auxiliary depth/video teacher-target forward passed on
2026-10-03. Current depth, future depth, future DINO patch, and current DINO
patch targets were finite and shape-matched to policy predictions; all weighted
loss terms were active. No optimizer, backward pass, or checkpoint write
occurred.

The next sub-gate is one minimal backward/optimizer step followed by checkpoint
export and strict reload. Do not call Phase 1 complete or start training until
that check passes.

The E000 launcher and immutable override set are prepared for a reserved
four-H100 window. Its resource preflight refuses GPUs with more than 1 GiB
already allocated, so shared-server occupancy cannot turn the smoke into an
unrecorded oversubscription experiment. E000 has not passed until the resulting
DCP model/optimizer/extra state, HF export, and strict FP32 HF reload are all
present in one immutable run directory.

Gate: one-batch official loss and checkpoint export/reload pass; no randomized
path in the training manifest.

### Phase 2 - Evaluation pipeline validation (Oct 3-5)

- Use the released RoboTwin post-trained checkpoint only as an infrastructure oracle.
- Run one clean task and one randomized task in FP32 on one H100.
- Verify seed logging, success parsing, checkpoint path points to `hf_ckpt`, and outputs are never overwritten.
- Measure simulator CPU/RAM load and FP32 GPU memory before increasing concurrency.

Gate: raw logs can be deterministically converted into the official result schema.

### Phase 3 - Training-path validation (Oct 4-7)

- One-task overfit on clean data.
- Short 50-task run (100 steps, then 1,000 steps if stable).
- Benchmark 4 x H100 throughput with one candidate micro-batch; measure before deciding global batch, accumulation, or total steps.
- Export and reload an HF checkpoint; run the one-task evaluation smoke.

Gate: finite/stable losses, expected trainable parameter count, acceptable memory headroom, and executable exported checkpoint.

### Phase 4 - First clean-only baseline (Oct 7-11)

- Start from the released base LingBot-VLA 2.0 checkpoint, not the mixed-data RoboTwin checkpoint.
- Use full fine-tuning as the first baseline because that is the official reference recipe's default behavior.
- Save early checkpoints more frequently than the official 10k interval until the best stopping region is known.
- Evaluate early/mid/late checkpoints on the sentinel set before one full official evaluation.

Gate: one complete, reproducible baseline with clean/randomized per-task results and failure analysis.

### Phase 5 - Minimal causal ablations (Oct 11-18)

Run sequentially, not in parallel unless the previous result motivates the next:

1. Action-expert-only (`train_expert_only=true`, with required vision freeze).
2. Expert-first then full fine-tuning, analogous to LP-FT.
3. Lower VLM/vision learning rate or layer-wise learning-rate decay.
4. Explicit base anchoring: L2-SP-style weight penalty or clean-frame feature distillation.
5. Base/fine-tuned checkpoint interpolation (WiSE-FT-style) if state dictionaries are fully compatible.

Each experiment must state one expected mechanism and one falsification criterion. Promote only variants that improve randomized SR or retention without an unacceptable clean-SR loss.

### Phase 6 - Confirmation and final model (Oct 18-22)

- Re-run the best recipe with a second training seed if budget permits.
- Evaluate the selected checkpoint on the full official protocol.
- Confirm inference precision, action chunk length, and all evaluation code against the pinned baseline.
- Freeze the final commit, config, data hashes, and checkpoint hash.

Gate: final `results.json` is generated from logs and passes schema/invariant validation.

### Phase 7 - Reproduction and submission (Oct 22-25)

- Reproduce a short training run from the clean checkout using only documented commands.
- Run submission scripts in a fresh shell and verify there are no hidden local dependencies.
- Prepare code materials, checkpoint, configs, result JSON, and reproduction/tuning report.
- Perform a mock submission on Oct 25, leaving Oct 26 as recovery buffer.

## 5. Compute strategy for 4 x H100

The official RoboTwin configuration currently states micro-batch 32, global batch 1,024, 50,000 steps, and was benchmarked at 32 GPUs. Copying that schedule onto four GPUs before measuring accumulation cost would risk consuming most of the remaining competition window.

Therefore:

- Benchmark 100 steps first and project wall-clock time.
- Keep optimizer comparisons separate from representation-preservation comparisons.
- Preserve the official effective batch only if measured throughput makes it feasible; otherwise choose a smaller effective batch and retune learning rate based on short-run loss and closed-loop checks.
- Evaluate checkpoints early. More optimization steps are not automatically better under narrow clean-only post-training.
- Start evaluation at four FP32 servers (one per H100). Increase resident servers only after measuring memory and host simulation load.

No final batch size, learning rate, or step budget is approved until the throughput gate is complete.

## 6. Learning plan

### Read first: system and benchmark

1. LingBot-VLA 2.0, *From Foundation to Application: Improving VLA Models in Practice* - understand the 55-D unified action representation, MoE action expert, and depth/video dual-query distillation: https://arxiv.org/abs/2607.06403
2. RoboTwin 2.0 - understand the five randomization axes and what the randomized setting actually changes: https://arxiv.org/abs/2506.18088
3. Qwen3-VL Technical Report - focus on DeepStack and visual-token flow because this is the visual-semantic substrate being adapted: https://arxiv.org/abs/2511.21631
4. Flow Matching for Generative Modeling - connect the vector-field objective to LingBot's continuous action generation: https://arxiv.org/abs/2210.02747

### Read before choosing the method

5. *Knowledge Insulating Vision-Language-Action Models* - the closest prior work to this project's question; identify which gradients and interfaces cause VLM knowledge loss: https://arxiv.org/abs/2505.23705
6. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* - motivates expert/head-first then full fine-tuning: https://arxiv.org/abs/2202.10054
7. *Robust Fine-Tuning of Zero-Shot Models* - motivates post-hoc base/fine-tuned weight interpolation: https://arxiv.org/abs/2109.01903
8. *Learning without Forgetting* - motivates teacher anchoring when old pretraining data are unavailable: https://arxiv.org/abs/1606.09282

### Read as implementation references

9. OpenVLA and OpenVLA-OFT - compare full/LoRA fine-tuning, continuous actions, action chunking, and small-data adaptation: https://arxiv.org/abs/2406.09246 and https://arxiv.org/abs/2502.19645
10. LoRA - use only if freezing/staged fine-tuning is insufficient or full fine-tuning is too costly: https://arxiv.org/abs/2106.09685

For each paper, write a half-page note with: problem, mechanism, assumptions, one applicable idea, one incompatibility with this challenge, and the cheapest falsifying experiment.

## 7. Target project narrative

If supported by results, the final technical story should be:

> Built a reproducible 10,000-rollout VLA evaluation system, diagnosed how clean-only post-training shifts LingBot-VLA 2.0 representations, and developed a staged/anchored adaptation recipe that improves randomized robustness and retention while preserving clean task performance.

The resume claim must include the exact baseline, absolute SR changes, generalization-gap change, compute budget, and number of rollouts. Do not claim representation preservation without a drift measurement, or robustness without held-out randomized evaluation.
