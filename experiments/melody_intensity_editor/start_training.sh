#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
WORKSPACE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
RUN_ROOT=${RUN_ROOT:-$SCRIPT_DIR}
PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE_ROOT=${SOURCE_ROOT:-$WORKSPACE_ROOT/data/augmented/train}
FORCE_REBUILD=${FORCE_REBUILD:-0}
DATASET_ARGS=${DATASET_ARGS:-}
TRAIN_ARGS=${TRAIN_ARGS:-}
RESUME_FROM=${RESUME_FROM:-}
AUTO_RESUME=${AUTO_RESUME:-0}
ALLOW_DATASET_OVERWRITE=${ALLOW_DATASET_OVERWRITE:-0}
META_PATH="$RUN_ROOT/artifacts/meta.json"
CHECKPOINTS_DIR="$RUN_ROOT/checkpoints"
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
LOG_TEE=${LOG_TEE:-1}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  print -r -- "[$(timestamp)] $*"
}

mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname -- "$LOG_FILE")"

if [[ "$LOG_TEE" == "1" ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

log "starting melody training pipeline"
log "log_file=$LOG_FILE"
log "run_root=$RUN_ROOT"
log "python_bin=$PYTHON_BIN"
log "source_root=$SOURCE_ROOT"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    log "python interpreter not found: $PYTHON_BIN"
    exit 1
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "python interpreter not found on PATH: $PYTHON_BIN"
    exit 1
  fi
  PYTHON_BIN=$(command -v "$PYTHON_BIN")
  log "resolved_python_bin=$PYTHON_BIN"
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  log "source MIDI directory not found: $SOURCE_ROOT"
  exit 1
fi

if [[ -z "$RESUME_FROM" && "$AUTO_RESUME" == "1" ]]; then
  if [[ -L "$CHECKPOINTS_DIR/latest" || -e "$CHECKPOINTS_DIR/latest" ]]; then
    RESUME_FROM="$CHECKPOINTS_DIR/latest"
  elif [[ -f "$CHECKPOINTS_DIR/latest.txt" ]]; then
    RESUME_FROM=$(<"$CHECKPOINTS_DIR/latest.txt")
  fi
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
  if [[ "$ALLOW_DATASET_OVERWRITE" != "1" ]] && [[ -d "$CHECKPOINTS_DIR" ]] && find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 | read -r _; then
    log "refusing dataset rebuild because checkpoints already exist under run_root=$RUN_ROOT"
    log "use a fresh RUN_ROOT for new experiments, or set ALLOW_DATASET_OVERWRITE=1 if you really intend to replace this dataset"
    exit 1
  fi
  log "stage=dataset_build"
  if [[ -n "$DATASET_ARGS" ]]; then
    log "dataset_args=$DATASET_ARGS"
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_melody_intensity_dataset.py" \
      --run-root "$RUN_ROOT" \
      --source-root "$SOURCE_ROOT" ${=DATASET_ARGS}
  else
    log "dataset_args=<defaults>"
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_melody_intensity_dataset.py" \
      --run-root "$RUN_ROOT" \
      --source-root "$SOURCE_ROOT"
  fi
else
  log "stage=dataset_build skipped (existing dataset matches source_root)"
fi

log "stage=training"
resume_args=()
if [[ -n "$RESUME_FROM" ]]; then
  log "resume_from=$RESUME_FROM"
  resume_args=(--resume-from "$RESUME_FROM")
fi
if [[ -n "$TRAIN_ARGS" ]]; then
  log "train_args=$TRAIN_ARGS"
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_melody_intensity_gpt.py" \
    --run-root "$RUN_ROOT" \
    "${resume_args[@]}" \
    ${=TRAIN_ARGS}
else
  log "train_args=<defaults>"
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/train_melody_intensity_gpt.py" \
    --run-root "$RUN_ROOT" \
    "${resume_args[@]}"
fi
