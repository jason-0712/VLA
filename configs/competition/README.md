# Competition Configs

This directory will contain immutable configuration snapshots for executed competition runs.

`robotwin_smoke_clean.yml` is an infrastructure-only, one-episode simulator
check. It is not a training configuration and its output must not be added to
the competition training manifest.

`E000_training_step_smoke_overrides.yaml` is the immutable, infrastructure-only
override set for one clean batch: FP32 FSDP2 forward/backward, one Muon update,
DCP plus HF export, and strict HF reload. The preparation script overlays it on
the pinned upstream recipe and resolves machine paths only in the output
directory.

Naming convention:

```text
E###_<method>_<scope>_<seed>.yaml
```

The first runnable training config is created only after:

1. the clean-only data manifest and normalization statistics are verified;
2. model/teacher/checkpoint paths are known;
3. a one-batch forward pass succeeds; and
4. the complete auxiliary forward-loss check succeeds.

The E000 one-step transaction comes next. The 100-step four-H100 throughput run
is created only after E000, one-task evaluation smoke, and one-task overfit pass.

Every config must state its base checkpoint, training manifest, robot config, optimizer, batch/accumulation, precision, seed, save cadence, and trainable-module policy. Never edit a config after launching its experiment; copy it to a new experiment ID.
