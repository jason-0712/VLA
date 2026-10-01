#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/download_robotwin_assets.sh {--download-only|--extract-only|--all}

Required environment variables:
  ROBOTWIN_ROOT  Pinned RoboTwin checkout

Run this script from the RoboTwin conda environment. Downloads are pinned to a
specific Hugging Face dataset revision and verified before extraction.
EOF
}

mode="${1:-}"
case "$mode" in
  --download-only) do_download=1; do_extract=0 ;;
  --extract-only) do_download=0; do_extract=1 ;;
  --all) do_download=1; do_extract=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 1 ;;
esac

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the pinned RoboTwin checkout}"

readonly expected_robotwin_commit="13c3c47ff4312dd62484bcd51be034af55c062d1"
readonly dataset_revision="3dc3b798668feb99ac61cc9086d84cbcc3d79186"
readonly asset_dir="${ROBOTWIN_ROOT}/assets"

actual_robotwin_commit="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
[[ "$actual_robotwin_commit" == "$expected_robotwin_commit" ]] || {
  echo "Unexpected RoboTwin commit: $actual_robotwin_commit" >&2
  exit 2
}

declare -Ar expected_size=(
  [background_texture.zip]=10970687027
  [embodiments.zip]=219847741
  [objects.zip]=3737778549
)
declare -Ar expected_sha256=(
  [background_texture.zip]=54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070
  [embodiments.zip]=85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789
  [objects.zip]=6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27
)
readonly archives=(background_texture.zip embodiments.zip objects.zip)

mkdir -p "$asset_dir"

if [[ "$do_download" == "1" ]]; then
  export ROBOTWIN_ASSET_DIR="$asset_dir"
  export ROBOTWIN_DATASET_REVISION="$dataset_revision"
  python - <<'PY'
import os

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    revision=os.environ["ROBOTWIN_DATASET_REVISION"],
    allow_patterns=[
        "background_texture.zip",
        "embodiments.zip",
        "objects.zip",
    ],
    local_dir=os.environ["ROBOTWIN_ASSET_DIR"],
    local_dir_use_symlinks=False,
    max_workers=3,
    resume_download=True,
)
PY
fi

for archive in "${archives[@]}"; do
  archive_path="${asset_dir}/${archive}"
  [[ -f "$archive_path" ]] || {
    echo "Missing asset archive: $archive_path" >&2
    exit 3
  }

  actual_size="$(stat -c '%s' "$archive_path")"
  [[ "$actual_size" == "${expected_size[$archive]}" ]] || {
    echo "Unexpected size for $archive: $actual_size" >&2
    exit 3
  }

  printf '%s  %s\n' "${expected_sha256[$archive]}" "$archive_path" | sha256sum -c -
  unzip -tq "$archive_path" >/dev/null
  echo "archive_valid=$archive bytes=$actual_size"
done

echo "ROBOTWIN_ASSET_ARCHIVES_VALIDATED revision=$dataset_revision"

if [[ "$do_extract" == "1" ]]; then
  for target in background_texture embodiments objects; do
    [[ ! -e "${asset_dir}/${target}" ]] || {
      echo "Refusing to overwrite existing asset directory: ${asset_dir}/${target}" >&2
      exit 4
    }
  done

  for archive in "${archives[@]}"; do
    unzip -q "${asset_dir}/${archive}" -d "$asset_dir"
  done

  for target in background_texture embodiments objects; do
    [[ -d "${asset_dir}/${target}" ]] || {
      echo "Expected directory missing after extraction: ${asset_dir}/${target}" >&2
      exit 5
    }
  done

  (
    cd "$ROBOTWIN_ROOT"
    python script/update_embodiment_config_path.py
  )

  echo "ROBOTWIN_ASSETS_EXTRACTED"
fi
