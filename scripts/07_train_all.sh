#!/usr/bin/env bash
# Trains all 6 backbones on Original images only:
#   - Protocol B: every model x every fold (main same-checkpoint benchmark)
#   - Protocol A: every model, once (needed for Table 4's A-vs-B comparison)
# Always trains on variant=orig -- SR is only ever used at TEST time
# (src/eval/same_checkpoint_eval.py), per the same-checkpoint design.
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate .venv only if you haven't already activated an environment
# yourself (checks $VIRTUAL_ENV). If you ran `source .venv/bin/activate`
# (or activated any other venv/conda env) before calling this script,
# it is left alone and whatever's already active is used as-is.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

MODELS=(vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t)
N_FOLDS=$(python -c "from src.config import CFG; print(CFG.protocol_b.n_folds)")

for model in "${MODELS[@]}"; do
  echo ""
  echo "=================================================================="
  echo "Protocol A: $model"
  echo "=================================================================="
  python -m src.train.train --model "$model" --protocol a

  for ((fold=0; fold<N_FOLDS; fold++)); do
    echo ""
    echo "=================================================================="
    echo "Protocol B fold $fold: $model"
    echo "=================================================================="
    python -m src.train.train --model "$model" --protocol b --fold "$fold"
  done
done

echo ""
echo "All checkpoints saved under checkpoints/classifiers/."
echo "Next: scripts/08_eval_all.sh"
