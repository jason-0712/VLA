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

This gate is download/structure validation only. The archive remains unchanged
and has not yet been extracted or converted. The next gate is safe extraction
to a separate raw directory followed by an episode-level HDF5 schema audit.
