#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/anaconda3/envs/gpt2piano/bin/python}
MAESTRO_SOURCE_ROOT=${MAESTRO_SOURCE_ROOT:-/Users/hermesreisner/gpt2-piano-mps-12k/data/raw/maestro-v3.0.0}
FORCE_REBUILD=${FORCE_REBUILD:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$MAESTRO_SOURCE_ROOT" ]]; then
  echo "MAESTRO source directory not found: $MAESTRO_SOURCE_ROOT" >&2
  exit 1
fi

if [[ "$FORCE_REBUILD" == "1" || ! -f "$SCRIPT_DIR/data/split_manifest.json" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_maestro_2k_split.py" --source-root "$MAESTRO_SOURCE_ROOT"
fi

if [[ "$FORCE_REBUILD" == "1" || ! -f "$SCRIPT_DIR/artifacts/meta.json" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/tokenize_maestro_2k.py"
fi

if [[ -n "$TRAIN_ARGS" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_gpt2_large_maestro_2k.py" ${=TRAIN_ARGS}
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_gpt2_large_maestro_2k.py"
fi
