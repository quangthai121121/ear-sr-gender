#!/usr/bin/env bash
# Evaluates every matched-domain checkpoint (trained by
# scripts/11_train_matched_domain.sh) on its own training variant, then
# aggregates into results/table_matched_domain.csv. Does not touch
# table3_same_checkpoint(.raw).csv at all.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

MODELS=(vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t)
VARIANTS=(bicubic realesrgan swinir)
N_FOLDS=$(python -c "from src.config import CFG; print(CFG.protocol_b.n_folds)")
PRED_DIR=$(python -c "from src.config import CFG; print(CFG.predictions_dir)")

for model in "${MODELS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    for ((fold=0; fold<N_FOLDS; fold++)); do
      tag="${model}_protoB_fold${fold}_retrain_${variant}"
      pred_path="${PRED_DIR}/${tag}__${variant}.csv"
      if [ -f "$pred_path" ]; then
        echo "[skip] $tag (already evaluated -- delete ${pred_path} and its row in "
        echo "       results/table_matched_domain_raw.csv if you want to re-run it)"
        continue
      fi
      echo ""
      echo "== Matched-domain eval: $model / $variant / fold $fold =="
      python -m src.eval.same_checkpoint_eval --model "$model" --protocol b --fold "$fold" \
        --retrain --variant "$variant"
    done
  done
done

echo ""
echo "=== Aggregating (also refreshes table3/4/5 if their raw files exist) ==="
python -m src.eval.aggregate

echo ""
echo "Result -> results/table_matched_domain.csv"
