#!/usr/bin/env bash
# For every trained checkpoint, evaluate that SAME checkpoint on
# Original/Bicubic/Real-ESRGAN/SwinIR test inputs (Table 3's raw data).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS=(vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t)
N_FOLDS=$(python -c "from src.config import CFG; print(CFG.protocol_b.n_folds)")

for model in "${MODELS[@]}"; do
  echo ""
  echo "== Protocol A: $model =="
  python -m src.eval.same_checkpoint_eval --model "$model" --protocol a

  for ((fold=0; fold<N_FOLDS; fold++)); do
    echo ""
    echo "== Protocol B fold $fold: $model =="
    python -m src.eval.same_checkpoint_eval --model "$model" --protocol b --fold "$fold"
  done
done

echo ""
echo "Raw results -> results/table3_same_checkpoint_raw.csv"
echo "Per-image predictions -> results/predictions/*.csv"
echo "Next: scripts/09_aggregate_and_flip.sh"
