#!/usr/bin/env bash
# Download the data (first run only), run the full analysis in order, and render the report.
# Usage: ./run_all.sh   (25-45 minutes on 8 cores; 03_models.py takes most of it)
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
for script in python/0*.py; do
  echo "Running $script"
  "$PY" "$script"
done

export PATH="$PWD/.venv/bin:$PATH" QUARTO_PYTHON="$PWD/$PY"
quarto render report.qmd
