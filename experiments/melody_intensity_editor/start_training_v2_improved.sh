#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
RUN_NAME=${RUN_NAME:-v2_improved_m16_overlap_e6}
RUN_ROOT=${RUN_ROOT:-$SCRIPT_DIR/runs/$RUN_NAME}
AUTO_RESUME=${AUTO_RESUME:-1}
FORCE_REBUILD=${FORCE_REBUILD:-0}

DEFAULT_DATASET_ARGS="--step-beats 8 --max-windows-per-file 8"
DEFAULT_TRAIN_ARGS="--epochs 6 --n-layer 16 --n-head 12 --n-embd 768 --lr 1.2e-4 --warmup-steps 800 --log-every-opt-steps 10"

if [[ ${+DATASET_ARGS} -eq 0 ]]; then
  DATASET_ARGS="$DEFAULT_DATASET_ARGS"
fi
if [[ ${+TRAIN_ARGS} -eq 0 ]]; then
  TRAIN_ARGS="$DEFAULT_TRAIN_ARGS"
fi

mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/run_preset.json" <<JSON
{
  "preset": "v2_improved",
  "run_root": "$RUN_ROOT",
  "dataset_args": "$DATASET_ARGS",
  "train_args": "$TRAIN_ARGS",
  "auto_resume": $AUTO_RESUME,
  "force_rebuild": $FORCE_REBUILD
}
JSON

print -r -- "[v2] run_root=$RUN_ROOT"
print -r -- "[v2] dataset_args=$DATASET_ARGS"
print -r -- "[v2] train_args=$TRAIN_ARGS"
print -r -- "[v2] auto_resume=$AUTO_RESUME"

exec env \
  RUN_ROOT="$RUN_ROOT" \
  AUTO_RESUME="$AUTO_RESUME" \
  FORCE_REBUILD="$FORCE_REBUILD" \
  DATASET_ARGS="$DATASET_ARGS" \
  TRAIN_ARGS="$TRAIN_ARGS" \
  zsh "$SCRIPT_DIR/start_training.sh"
