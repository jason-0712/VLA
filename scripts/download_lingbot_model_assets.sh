#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/download_lingbot_model_assets.sh {--base-only|--dependencies-only|--all}

Required environment:
  MODEL_ROOT  Destination directory for immutable model snapshots

Run this script from the lingbotvla Conda environment. Every repository is
pinned to an exact Hugging Face revision. Existing partial downloads resume.
EOF
}

[[ $# -eq 1 ]] || {
  usage >&2
  exit 2
}

case "$1" in
  --base-only | --dependencies-only | --all) mode="$1" ;;
  *)
    usage >&2
    exit 2
    ;;
esac

: "${MODEL_ROOT:?Set MODEL_ROOT to the model snapshot directory}"
mkdir -p "$MODEL_ROOT"
export MODEL_ROOT mode

python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_SOURCES = (
    {
        "group": "base",
        "repo_id": "robbyant/lingbot-vla-v2-6b",
        "revision": "11c703bf6a5c1f45b3b69168482da11fdbba53d7",
        "expected_bytes": 28_239_981_618,
        "directory": "lingbot-vla-v2-6b",
    },
    {
        "group": "dependency",
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "expected_bytes": 8_887_292_732,
        "directory": "Qwen3-VL-4B-Instruct",
    },
    {
        "group": "dependency",
        "repo_id": "Ruicheng/moge-2-vitb-normal",
        "revision": "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368",
        "expected_bytes": 419_111_765,
        "directory": "moge-2-vitb-normal",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


mode = os.environ["mode"]
selected_groups = {
    "--base-only": {"base"},
    "--dependencies-only": {"dependency"},
    "--all": {"base", "dependency"},
}[mode]
model_root = Path(os.environ["MODEL_ROOT"]).resolve()
api = HfApi()

for source in MODEL_SOURCES:
    if source["group"] not in selected_groups:
        continue

    info = api.model_info(
        source["repo_id"],
        revision=source["revision"],
        files_metadata=True,
    )
    if info.sha != source["revision"]:
        raise RuntimeError(
            f"Revision mismatch for {source['repo_id']}: {info.sha}"
        )
    total_bytes = sum(sibling.size or 0 for sibling in info.siblings)
    if total_bytes != source["expected_bytes"]:
        raise RuntimeError(
            f"Repository size changed for {source['repo_id']}: {total_bytes}"
        )

    target = model_root / source["directory"]
    print(
        f"downloading repo={source['repo_id']} revision={source['revision']} "
        f"bytes={total_bytes} target={target}",
        flush=True,
    )
    snapshot_download(
        repo_id=source["repo_id"],
        revision=source["revision"],
        local_dir=target,
        max_workers=4,
    )

    checked_files = 0
    checked_lfs_bytes = 0
    for sibling in info.siblings:
        path = target / sibling.rfilename
        if not path.is_file():
            raise RuntimeError(f"Missing downloaded file: {path}")
        actual_size = path.stat().st_size
        if actual_size != (sibling.size or 0):
            raise RuntimeError(
                f"Size mismatch for {path}: {actual_size} != {sibling.size}"
            )
        if sibling.lfs is not None:
            actual_sha256 = sha256(path)
            if actual_sha256 != sibling.lfs.sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {path}: {actual_sha256}"
                )
            checked_lfs_bytes += actual_size
        checked_files += 1

    print(
        f"MODEL_SNAPSHOT_VALIDATED repo={source['repo_id']} "
        f"revision={source['revision']} files={checked_files} "
        f"lfs_bytes={checked_lfs_bytes} target={target}",
        flush=True,
    )
PY
