#!/usr/bin/env bash
# Download the data (first run only), run the full analysis in order, and render the report.
# Usage: ./run_all.sh   (25-45 minutes on 8 cores; 03_models.py takes most of it)
set -euo pipefail
cd "$(dirname "$0")"

# The virtual environment keeps its programs in bin/ on Mac and Linux, Scripts/ on Windows.
if [[ -d .venv/bin ]]; then VENV_BIN=.venv/bin; PY=$VENV_BIN/python
else VENV_BIN=.venv/Scripts; PY=$VENV_BIN/python.exe; fi

for script in python/0*.py; do
  echo "Running $script"
  "$PY" "$script"
done

export PATH="$PWD/$VENV_BIN:$PATH" QUARTO_PYTHON="$PWD/$PY"
quarto render report.qmd
