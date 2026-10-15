#!/usr/bin/env bash
# Download the data (first run only) and run the full analysis in order.
# Usage: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
for script in python/0*.py; do
  echo "Running $script"
  "$PY" "$script"
done
