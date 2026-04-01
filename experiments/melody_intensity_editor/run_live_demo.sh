#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/run_live_demo.py" "$@"
