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

## First task-level LeRobot v2.1 conversion

The deterministic `beat_block_hammer` conversion and full integrity audit
passed on 2026-10-02. It uses RoboTwin commit
`13c3c47ff4312dd62484bcd51be034af55c062d1` and the LeRobot revision locked by
RoboTwin, `a445d9c9da6bea99a8972daa4fe1fdd053d711d2` (`codebase_version: v2.1`).

The wrapper preserves the official `data_transform`, `create_empty_dataset`,
and `populate_dataset` functions. It only makes two reproducibility controls
explicit: numeric episode order `0..49` instead of unsorted `os.walk`, and
NumPy seed 0 for selecting one `seen` instruction per episode.

Conversion result:

- 50 episodes and 5,682 frames
- 14-D float32 state and 14-D float32 action
- three 640 x 480 training cameras
- FPS metadata 50, with timestamps exactly `frame_index / 50`
- 50 Parquet files and four metadata files
- image-mode storage: image bytes embedded in Parquet, no duplicate MP4 files
- 50 deterministic `seen` instruction selections, all reproduced from seed 0
- pinned LeRobot reader successfully loaded samples 0, 124, 125, and 5,681
- all 17,046 converted training images decoded and exactly matched the
  corresponding processed-HDF5 pixels
- every converted state/action value exactly matched the processed HDF5

Immutable records and hashes:

```text
conversion audit:
/mnt/data1/hanyu/lingbot_vla/audits/lerobot_conversion_beat_block_hammer_seed0_20261002T125523.json

full integrity audit:
/mnt/data1/hanyu/lingbot_vla/audits/lerobot_audit_beat_block_hammer_seed0_20261002T131448.json

processed tree SHA256:
4c986aefce4e7dc073f24253559cbbc8c3581035d93f0fe6b091786ee08cadc8

LeRobot dataset tree SHA256:
9e351b7370e98e9322df1404f90a36fc674c438f07be10f127de86b67b4175f5
```

### Measured storage expansion

For this task:

| Stage | Bytes | Ratio to source ZIP |
| --- | ---: | ---: |
| Source ZIP | 229,401,324 | 1.00x |
| Uncompressed ZIP members | 319,531,834 | 1.39x |
| Processed Aloha HDF5 tree | 506,262,934 | 2.21x |
| LeRobot v2.1 image dataset | 1,255,148,320 | 5.47x |

If this task's byte ratios held over all 23,780,715,316 source bytes, the 50
tasks would require approximately 33.1 GB raw, 52.5 GB processed HDF5, and
130.1 GB LeRobot, or about 239.5 GB while all four stages coexist. Even keeping
only source ZIPs plus LeRobot would be about 153.9 GB. `/mnt/data1` had about
140 GB free at this gate, so the full acquisition must use staged conversion
and a revised cross-filesystem storage layout. Do not launch all-task download
until that layout is fixed.

The next functional gate is to open this one-task dataset through LingBot's
actual training-side data loader. That verifies v2.1 compatibility and the
RoboTwin 14-D-to-unified mapping before spending storage on all 50 tasks.
