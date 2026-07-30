#!/usr/bin/env bash
# Precomputes Real-ESRGAN and SwinIR x4 outputs for EVERY image in the
# dataset. This is the slow, GPU-heavy step (tens of thousands of images
# through two deep SR networks) -- expect this to take hours depending on
# GPU. Both wrapper scripts are resumable (they skip already-written files),
# so it's safe to Ctrl-C and re-run.
#
# Runs from the CLASSIFIER venv (has pandas/tqdm/yaml), but internally
# calls out to each SR repo's OWN interpreter (see configs/paths.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate .venv only if you haven't already activated an environment
# yourself (checks $VIRTUAL_ENV). If you ran `source .venv/bin/activate`
# (or activated any other venv/conda env) before calling this script,
# it is left alone and whatever's already active is used as-is.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

echo "=== Real-ESRGAN x4 (this can take a while) ==="
python -m src.sr.run_realesrgan

echo ""
echo "=== SwinIR-Large x4 (this can take a while) ==="
python -m src.sr.run_swinir

echo ""
echo "Done. Outputs cached under:"
python -c "from src.config import CFG; print(' ', CFG.processed_root / 'realesrgan'); print(' ', CFG.processed_root / 'swinir_large')"
