#!/usr/bin/env bash
# Cross-dataset evaluation on AWE (1,000 images, 100 subjects).
# Applies the existing EarVN1.0 Protocol B same-checkpoint classifiers
# (scripts/07) unchanged to AWE Original / Bicubic / Real-ESRGAN / SwinIR.
# Edit configs/paths_awe.yaml -> paths.data_root first.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
export EAR_SR_CONFIG=configs/paths_awe.yaml

echo "=== AWE metadata ==="
python -m src.data.build_awe_metadata

echo "=== Real-ESRGAN x4 on AWE ==="
python -m src.sr.run_realesrgan
echo "=== SwinIR-Large x4 on AWE ==="
# SwinIR writes into its own repo's results/ before we move files out; clear
# leftovers so a stale output from an earlier crashed run is never picked up.
rm -rf "$(python -c 'from src.config import CFG; print(CFG.swinir_repo)')/results"
# Untiled (as on EarVN1.0) for crops with long side < 512; the few larger AWE
# images run out of GPU memory untiled, so they get 256-px tiles.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.sr.run_swinir --max-long-side 512 --batch-size 100
python -m src.sr.run_swinir --min-long-side 512 --tile 256 --batch-size 20
echo "SwinIR outputs: $(find data/processed_awe/swinir_large -name '*.png' | wc -l) / 1000"

echo "=== Same-checkpoint evaluation (6 backbones x 5 folds), one process per checkpoint ==="
for m in vgg19 mobilenet_v2 resnet50 efficientnet_b0 swin_t maxvit_t; do
  for f in 0 1 2 3 4; do
    python -m src.eval.awe_eval --model "$m" --fold "$f" > /dev/null
  done
done
python -m src.eval.awe_eval --summarize

echo ""
echo "Results -> results_awe/table_awe_{raw,summary,flips,bands}.csv"
