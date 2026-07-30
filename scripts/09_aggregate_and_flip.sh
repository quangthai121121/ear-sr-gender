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

echo "=== Flip analysis (Table 5 raw) ==="
python -m src.eval.flip_analysis --all

echo ""
echo "=== Native-size breakdown (Figure 2) ==="
python -m src.eval.native_size --all

echo ""
echo "=== Aggregating final tables (3, 4, 5) ==="
python -m src.eval.aggregate

echo ""
echo "Final tables written to:"
echo "  results/table3_same_checkpoint.csv"
echo "  results/table4_protocolA_vs_B.csv"
echo "  results/table5_flips.csv"
echo "  results/table_native_size.csv + figure2_native_size__*.png"
