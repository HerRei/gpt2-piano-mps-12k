#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/anaconda3/envs/gpt2piano/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/Users/hermesreisner/gpt2-piano-mps-12k/data/augmented/train}
FORCE_REBUILD=${FORCE_REBUILD:-0}
DATASET_ARGS=${DATASET_ARGS:-}
TRAIN_ARGS=${TRAIN_ARGS:-}
META_PATH="$SCRIPT_DIR/artifacts/meta.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "source MIDI directory not found: $SOURCE_ROOT" >&2
  exit 1
fi

NEEDS_REBUILD=0
if [[ "$FORCE_REBUILD" == "1" || ! -f "$META_PATH" || -n "$DATASET_ARGS" ]]; then
  NEEDS_REBUILD=1
else
  if ! "$PYTHON_BIN" - <<'PY' "$META_PATH" "$SOURCE_ROOT"; then
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
source_root = Path(sys.argv[2]).expanduser().resolve()
meta = json.loads(meta_path.read_text())

if Path(meta.get("source_root", "")).expanduser().resolve() != source_root:
    raise SystemExit(1)
if int(meta.get("source_file_cap", 0) or 0) != 0:
    raise SystemExit(1)
PY
    NEEDS_REBUILD=1
  fi
fi

if [[ "$NEEDS_REBUILD" == "1" ]]; then
  if [[ -n "$DATASET_ARGS" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_melody_intensity_dataset.py" \
      --source-root "$SOURCE_ROOT" ${=DATASET_ARGS}
  else
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_melody_intensity_dataset.py" \
      --source-root "$SOURCE_ROOT"
  fi
fi

if [[ -n "$TRAIN_ARGS" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_melody_intensity_gpt.py" ${=TRAIN_ARGS}
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_melody_intensity_gpt.py"
fi
