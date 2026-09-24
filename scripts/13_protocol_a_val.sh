#!/usr/bin/env bash
# Re-runs Protocol A WITHOUT early-stopping on the test set.
#
# The original Protocol A runs (scripts/07) have no val split and therefore
# early-stop on TEST, which makes the Protocol A numbers (Table 4) optimistic.
# Protocol "av" keeps the exact same test half as protocol_a.csv and carves
# a val subset (configs/paths.yaml -> protocol_a.val_fraction_of_train) out of
# the train half. Checkpoints are tagged <model>_protoAV and never overwrite
# the published <model>_protoA ones.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

MODELS=(vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t)

python -m src.data.splits --protocol av

for model in "${MODELS[@]}"; do
  echo ""
  echo "=================================================================="
  echo "Protocol AV: $model"
  echo "=================================================================="
  python -m src.train.train --model "$model" --protocol av
  python -m src.eval.same_checkpoint_eval --model "$model" --protocol av
done

python -m src.eval.aggregate

echo ""
echo "Protocol AV vs B -> results/table4_protocolAV_vs_B.csv"
