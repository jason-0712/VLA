# Competition Configs

This directory will contain immutable configuration snapshots for executed competition runs.

`robotwin_smoke_clean.yml` is an infrastructure-only, one-episode simulator
check. It is not a training configuration and its output must not be added to
the competition training manifest.

Naming convention:

```text
E###_<method>_<scope>_<seed>.yaml
```

The first runnable training config is created only after:

1. the clean-only data manifest and normalization statistics are verified;
2. model/teacher/checkpoint paths are known;
3. a one-batch forward pass succeeds; and
4. a 100-step four-H100 throughput test determines a feasible batch and step budget.

Every config must state its base checkpoint, training manifest, robot config, optimizer, batch/accumulation, precision, seed, save cadence, and trainable-module policy. Never edit a config after launching its experiment; copy it to a new experiment ID.
