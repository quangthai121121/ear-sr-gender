#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate .venv only if you haven't already activated an environment
# yourself (checks $VIRTUAL_ENV). If you ran `source .venv/bin/activate`
# (or activated any other venv/conda env) before calling this script,
# it is left alone and whatever's already active is used as-is.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
python -m src.data.splits --protocol both
