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
python -m src.train.hp_search --fold 0 --epochs 8 --final-epochs 20
echo ""
echo "Locked config written to configs/locked_config.yaml -- inspect it,"
echo "then proceed to scripts/07_train_all.sh."
