#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_competition.sh

Required environment variables:
  LINGBOT_VLA_ROOT   Pinned LingBot-VLA 2.0 checkout
  COMPETITION_CONFIG Absolute path to an immutable experiment YAML
  TRAIN_MANIFEST     Clean-only training manifest
  OUTPUT_DIR         New output directory, or an existing run with ALLOW_RESUME=1

Optional:
  ALLOW_RESUME=1     Permit an existing OUTPUT_DIR for an intentional resume
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${LINGBOT_VLA_ROOT:?Set LINGBOT_VLA_ROOT to the pinned LingBot-VLA checkout}"
: "${COMPETITION_CONFIG:?Set COMPETITION_CONFIG to an immutable YAML snapshot}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the clean-only data manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique run directory}"

[[ -d "$LINGBOT_VLA_ROOT" ]] || { echo "Missing LingBot checkout: $LINGBOT_VLA_ROOT" >&2; exit 2; }
[[ -f "$COMPETITION_CONFIG" ]] || { echo "Missing config: $COMPETITION_CONFIG" >&2; exit 2; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "Missing training manifest: $TRAIN_MANIFEST" >&2; exit 2; }

if [[ "$TRAIN_MANIFEST" =~ [Rr][Aa][Nn][Dd][Oo][Mm][Ii][Zz][Ee][Dd] ]]; then
  echo "Refusing training manifest whose path contains 'randomized': $TRAIN_MANIFEST" >&2
  exit 3
fi

if grep -Eiq 'randomized' "$TRAIN_MANIFEST"; then
  echo "Refusing training manifest containing randomized data entries: $TRAIN_MANIFEST" >&2
  exit 3
fi

if [[ -e "$OUTPUT_DIR" && "${ALLOW_RESUME:-0}" != "1" ]]; then
  echo "OUTPUT_DIR already exists. Use a new path or set ALLOW_RESUME=1 for a recorded resume." >&2
  exit 4
fi

cd "$LINGBOT_VLA_ROOT"
exec bash train.sh \
  tasks/vla/train_lingbotvla.py \
  "$COMPETITION_CONFIG" \
  --data.train_path "$TRAIN_MANIFEST" \
  --train.output_dir "$OUTPUT_DIR"
