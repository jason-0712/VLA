#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/extract_competition_clean_data.sh --task TASK_NAME

Required environment:
  CLEAN_DATA_ROOT  Data root used by download_competition_clean_data.sh

The source ZIP is revalidated against the pinned Hugging Face revision, checked
for unsafe members, and extracted atomically into raw/<task>/. Existing raw
task data is never overwritten.
EOF
}

if [[ $# -ne 2 || "$1" != "--task" ]]; then
  usage >&2
  exit 2
fi

task_name="$2"
[[ "$task_name" =~ ^[a-z0-9_]+$ ]] || {
  echo "Invalid task name: $task_name" >&2
  exit 2
}
: "${CLEAN_DATA_ROOT:?Set CLEAN_DATA_ROOT to a data directory outside Git}"
export CLEAN_DATA_ROOT task_name

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi


REPO_ID = "TianxingChen/RoboTwin2.0"
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
ARCHIVE_NAME = "aloha-agilex_clean_50.zip"
EXTRACTED_NAME = "aloha-agilex_clean_50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(os.environ["CLEAN_DATA_ROOT"]).expanduser().resolve()
task = os.environ["task_name"]
if "randomized" in task.lower():
    raise RuntimeError(f"Refusing randomized task path: {task}")

relative_archive = f"dataset/{task}/{ARCHIVE_NAME}"
source = root / "source_archives" / relative_archive
target_parent = root / "raw" / task
target = target_parent / EXTRACTED_NAME
if not source.is_file():
    raise FileNotFoundError(
        f"Source archive is missing: {source}. Run the clean downloader first."
    )
if target.exists():
    raise FileExistsError(f"Refusing to overwrite existing raw data: {target}")

info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
if info.sha != REVISION:
    raise RuntimeError(f"Dataset revision mismatch: {info.sha} != {REVISION}")
matches = [item for item in info.siblings if item.rfilename == relative_archive]
if len(matches) != 1 or matches[0].lfs is None:
    raise RuntimeError(f"Pinned clean archive metadata not found: {relative_archive}")
metadata = matches[0]
if source.stat().st_size != metadata.size:
    raise RuntimeError(f"Source size mismatch: {source}")
source_sha256 = sha256(source)
if source_sha256 != metadata.lfs.sha256:
    raise RuntimeError(
        f"Source SHA256 mismatch: {source_sha256} != {metadata.lfs.sha256}"
    )

staging_root = root / "raw" / ".staging"
staging_root.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=f"{task}-", dir=staging_root)).resolve()
try:
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"Refusing ZIP symlink: {member.filename}")
            destination = (staging / Path(*path.parts)).resolve()
            if not destination.is_relative_to(staging):
                raise RuntimeError(f"ZIP member escapes staging root: {member.filename}")
        archive.extractall(staging)

    extracted = staging / EXTRACTED_NAME
    if not extracted.is_dir():
        raise RuntimeError(f"Expected one top-level directory named {EXTRACTED_NAME}")
    expected_sets = {
        "hdf5": sorted((extracted / "data").glob("episode*.hdf5")),
        "instructions": sorted((extracted / "instructions").glob("episode*.json")),
        "videos": sorted((extracted / "video").glob("episode*.mp4")),
        "trajectories": sorted((extracted / "_traj_data").glob("episode*.pkl")),
    }
    wrong_counts = {
        name: len(paths) for name, paths in expected_sets.items() if len(paths) != 50
    }
    if wrong_counts:
        raise RuntimeError(f"Expected 50 files per episode artifact type: {wrong_counts}")
    for required in (extracted / "seed.txt", extracted / "scene_info.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    target_parent.mkdir(parents=True, exist_ok=True)
    extracted.rename(target)
finally:
    if staging.exists():
        shutil.rmtree(staging)

audit = {
    "status": "ok",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "repository": REPO_ID,
    "revision": REVISION,
    "source_archive": str(source),
    "source_bytes": source.stat().st_size,
    "source_sha256": source_sha256,
    "raw_path": str(target),
    "artifact_counts": {name: len(paths) for name, paths in expected_sets.items()},
    "source_preserved": True,
    "scope": "safe_atomic_extraction_only",
}
audit_dir = root / "audits"
audit_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
audit_path = audit_dir / f"clean_extract_{task}_{timestamp}.json"
with audit_path.open("x") as stream:
    json.dump(audit, stream, indent=2, sort_keys=True)
    stream.write("\n")

print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
print(f"audit_path={audit_path}", flush=True)
print("CLEAN_DATA_EXTRACTION_OK", flush=True)
PY
