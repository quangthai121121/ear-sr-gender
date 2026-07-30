#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.train.hp_search --fold 0 --epochs 8 --final-epochs 20
echo ""
echo "Locked config written to configs/locked_config.yaml -- inspect it,"
echo "then proceed to scripts/07_train_all.sh."
