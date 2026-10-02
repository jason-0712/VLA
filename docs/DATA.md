# Competition Data Record

## Source catalog

The clean training source is pinned to the Hugging Face dataset
`TianxingChen/RoboTwin2.0` at revision
`3dc3b798668feb99ac61cc9086d84cbcc3d79186`.

At that revision, the strict path pattern
`dataset/<task>/aloha-agilex_clean_50.zip` selects exactly 50 unique tasks and
23,780,715,316 compressed bytes. The downloader rejects any selected path
containing `randomized`, checks repository revision and catalog invariants, and
validates each downloaded archive using its LFS SHA256 and ZIP CRC:

```bash
export CLEAN_DATA_ROOT=/path/outside/git
bash scripts/download_competition_clean_data.sh --task beat_block_hammer
# After the one-task audit is complete:
bash scripts/download_competition_clean_data.sh --all
```

Do not use upstream `assets/training_data/robotwin.txt` as a competition
manifest. At the pinned LingBot commit it has 99 rows: 50 explicitly
randomized rows, plus a clean Piper row in place of one required Aloha-AgileX
entry. A competition manifest must instead be generated from the audited 50
clean Aloha-AgileX archives.

## First task-level source validation

The `beat_block_hammer` source archive passed on 2026-10-02:

- local source archive:
  `/mnt/data1/hanyu/lingbot_vla/source_archives/dataset/beat_block_hammer/aloha-agilex_clean_50.zip`
- compressed bytes: 229,401,324
- SHA256: `a135ad233bdcffff65fb636780f95ec34abc36ccf2bd0ff26ab07c9c464cc6af`
- ZIP members: 207
- uncompressed member bytes: 319,531,834
- contents: 50 HDF5 episodes, 50 trajectory pickle files, 50 MP4 videos,
  50 instruction JSON files, `seed.txt`, and `scene_info.json`
- immutable audit:
  `/mnt/data1/hanyu/lingbot_vla/audits/clean_download_task_beat_block_hammer_20261002T111318.json`

The source archive remains unchanged. It was subsequently extracted using
`scripts/extract_competition_clean_data.sh`, which rejects path traversal and
symlink members, stages the extraction, and atomically installs it without
overwriting existing raw data. The extraction audit is:

```text
/mnt/data1/hanyu/lingbot_vla/audits/clean_extract_beat_block_hammer_20261002T113759.json
```

## First task-level episode audit

All 50 extracted `beat_block_hammer` episodes passed
`scripts/audit_competition_raw_data.py` on 2026-10-02:

- 50 unique source seeds and 50 clean scene metadata entries
- 5,732 raw frames; 109–126 frames per episode, mean 114.64
- 14-D raw joint vectors exactly equal left arm/gripper plus right
  arm/gripper concatenation (maximum absolute error 0)
- 1,100,544 numeric values checked with zero NaN/Inf values
- 22,928 encoded images decoded across four cameras, all 320 x 240 x 3
- all 50 diagnostic MP4 frame counts match their HDF5 episode lengths
- every episode has 100 `seen` and 100 `unseen` instructions
- official training inputs map head/left/right cameras to
  `cam_high`/`cam_left_wrist`/`cam_right_wrist`; the front camera is not used
- trajectory pickle files were checked for presence and size but deliberately
  not deserialized; the official conversion path reads the HDF5 files

The raw HDF5 files contain no explicit timestamp field. The pinned official
converter aligns samples by index as `state[t] -> action[t+1]`, dropping one
frame per episode, so the expected one-task LeRobot output is 5,682 frames.
Diagnostic MP4 files report 30 FPS while the official LeRobot converter writes
50 FPS metadata; the converter reads HDF5 frames rather than the MP4 files. Keep
the upstream 50 FPS behavior for the reproduction baseline and record this
fact instead of inferring control timing from diagnostic video metadata.

The immutable full audit, including per-episode rows and numeric ranges, is:

```text
/mnt/data1/hanyu/lingbot_vla/audits/raw_episode_audit_beat_block_hammer_20261002T114242.json
```

The next gate is a one-task official raw-to-LeRobot conversion followed by a
LeRobot metadata/frame audit. Do not acquire all 50 tasks until that conversion
path is verified.
