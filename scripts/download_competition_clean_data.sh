#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download_competition_clean_data.sh --task TASK_NAME
  scripts/download_competition_clean_data.sh --all

Required environment:
  CLEAN_DATA_ROOT  Data root outside Git; source archives go below it

The downloader pins the RoboTwin2.0 dataset revision, accepts only the 50
official aloha-agilex_clean_50.zip files, and validates size, LFS SHA256, and
ZIP integrity. It never extracts or modifies a source archive.
EOF
}

if [[ $# -eq 1 && "$1" == "--all" ]]; then
  mode="all"
  task_name=""
elif [[ $# -eq 2 && "$1" == "--task" ]]; then
  mode="task"
  task_name="$2"
  [[ "$task_name" =~ ^[a-z0-9_]+$ ]] || {
    echo "Invalid task name: $task_name" >&2
    exit 2
  }
else
  usage >&2
  exit 2
fi

: "${CLEAN_DATA_ROOT:?Set CLEAN_DATA_ROOT to a data directory outside Git}"
mkdir -p "$CLEAN_DATA_ROOT"
export CLEAN_DATA_ROOT mode task_name

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "TianxingChen/RoboTwin2.0"
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
EXPECTED_ARCHIVE_COUNT = 50
EXPECTED_TOTAL_BYTES = 23_780_715_316
ARCHIVE_PATTERN = re.compile(
    r"^dataset/(?P<task>[a-z0-9_]+)/aloha-agilex_clean_50\.zip$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(os.environ["CLEAN_DATA_ROOT"]).expanduser().resolve()
mode = os.environ["mode"]
requested_task = os.environ["task_name"]
api = HfApi()
info = api.dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
if info.sha != REVISION:
    raise RuntimeError(f"Dataset revision mismatch: {info.sha} != {REVISION}")

catalog = []
for sibling in info.siblings:
    match = ARCHIVE_PATTERN.fullmatch(sibling.rfilename)
    if match is None:
        continue
    if "randomized" in sibling.rfilename.lower():
        raise RuntimeError(f"Forbidden randomized path selected: {sibling.rfilename}")
    if sibling.lfs is None:
        raise RuntimeError(f"Clean archive lacks LFS metadata: {sibling.rfilename}")
    catalog.append(
        {
            "task": match.group("task"),
            "path": sibling.rfilename,
            "bytes": sibling.size,
            "sha256": sibling.lfs.sha256,
        }
    )

catalog.sort(key=lambda row: row["task"])
if len(catalog) != EXPECTED_ARCHIVE_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_ARCHIVE_COUNT} clean archives, found {len(catalog)}"
    )
total_bytes = sum(row["bytes"] for row in catalog)
if total_bytes != EXPECTED_TOTAL_BYTES:
    raise RuntimeError(
        f"Clean archive byte total changed: {total_bytes} != {EXPECTED_TOTAL_BYTES}"
    )
if len({row["task"] for row in catalog}) != EXPECTED_ARCHIVE_COUNT:
    raise RuntimeError("Clean archive catalog contains duplicate task names")

if mode == "all":
    selected = catalog
else:
    selected = [row for row in catalog if row["task"] == requested_task]
    if not selected:
        available = ", ".join(row["task"] for row in catalog)
        raise RuntimeError(
            f"Task {requested_task!r} is not in the 50-task catalog: {available}"
        )

archive_root = root / "source_archives"
archive_root.mkdir(parents=True, exist_ok=True)
validated = []
for row in selected:
    print(
        f"downloading task={row['task']} bytes={row['bytes']} "
        f"revision={REVISION}",
        flush=True,
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=row["path"],
            local_dir=archive_root,
        )
    ).resolve()
    if downloaded.stat().st_size != row["bytes"]:
        raise RuntimeError(f"Size mismatch for {downloaded}")
    actual_sha256 = sha256(downloaded)
    if actual_sha256 != row["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {downloaded}: {actual_sha256} != {row['sha256']}"
        )
    with zipfile.ZipFile(downloaded) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure in {downloaded}: {bad_member}")
        member_count = len(archive.infolist())
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())
    validated.append(
        {
            **row,
            "local_path": str(downloaded),
            "member_count": member_count,
            "uncompressed_bytes": uncompressed_bytes,
        }
    )
    print(
        f"validated task={row['task']} sha256={actual_sha256} "
        f"members={member_count} uncompressed_bytes={uncompressed_bytes}",
        flush=True,
    )

audit = {
    "status": "ok",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "repository": REPO_ID,
    "revision": REVISION,
    "catalog_archive_count": len(catalog),
    "catalog_total_bytes": total_bytes,
    "selection": mode if mode == "all" else requested_task,
    "selected_archive_count": len(selected),
    "selected_total_bytes": sum(row["bytes"] for row in selected),
    "validated_archives": validated,
    "scope": "download_and_validate_only_no_extraction",
}
audit_dir = root / "audits"
audit_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
audit_path = audit_dir / f"clean_download_{mode}_{requested_task or '50tasks'}_{timestamp}.json"
with audit_path.open("x") as stream:
    json.dump(audit, stream, indent=2, sort_keys=True)
    stream.write("\n")

print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
print(f"audit_path={audit_path}", flush=True)
print("CLEAN_DATA_DOWNLOAD_VALIDATION_OK", flush=True)
PY
