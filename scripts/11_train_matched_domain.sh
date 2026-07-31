#!/usr/bin/env bash
# Matched-domain retraining (secondary analysis): train each backbone on
# each SR/bicubic variant (not just Original), using the SAME locked
# config as the main same-checkpoint runs -- this is what makes it a valid
# comparison, unlike the old uncontrolled Table 2 survey which changed
# optimizer/WD at the same time as the input.
#
# COST WARNING: this is 6 models x 5 folds x N variants ADDITIONAL full
# training runs on top of the 36 already done in scripts/07. Includes all
# 3 non-Original variants (bicubic, realesrgan, swinir) = 90 extra runs,
# so all 4 conditions (orig/bicubic/realesrgan/swinir, each trained AND
# tested on itself) can be compared side by side in
# results/table_matched_domain.csv. "train orig, test orig" is NOT
# retrained separately here -- it's already exactly Table 3's "orig"
# column (src/eval/aggregate.py pulls it in for you), so retraining it
# again would just add noise from a different random init, not new
# information. Comment out variants below / edit MODELS if compute is tight.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

MODELS=(vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t)
VARIANTS=(bicubic realesrgan swinir)
N_FOLDS=$(python -c "from src.config import CFG; print(CFG.protocol_b.n_folds)")

for model in "${MODELS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    for ((fold=0; fold<N_FOLDS; fold++)); do
      echo ""
      echo "=================================================================="
      echo "Matched-domain: $model / $variant / fold $fold"
      echo "=================================================================="
      python -m src.train.train --model "$model" --protocol b --fold "$fold" \
        --retrain --variant "$variant"
    done
  done
done

echo ""
echo "Matched-domain checkpoints saved under checkpoints/classifiers/*_retrain_*.pt"
echo "Next: scripts/12_eval_matched_domain.sh"
