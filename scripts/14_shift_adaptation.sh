#!/usr/bin/env bash
# Label-free test-time adaptation of the Original-trained Protocol B
# checkpoints to SR inputs (prior/bias matching and AdaBN), plus the CPU-only
# geometry-shortcut check and paired statistics used in the paper text.
# Needs the Protocol B same-checkpoint checkpoints from scripts/07.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m src.eval.shift_adaptation --all
python -m src.eval.shortcut_lr
python -m src.eval.paired_stats

echo ""
echo "Results -> results/table_shift_adaptation{,_raw}.csv, results/table_shortcut_lr.csv,"
echo "           results/table_matched_domain_{paired,pooled}.csv, results/table_operating_point_shift.csv"
