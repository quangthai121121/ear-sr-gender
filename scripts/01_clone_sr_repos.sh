#!/usr/bin/env bash
# Clones the two SR repos used as preprocessing (Real-ESRGAN, SwinIR) into
# external/. Each gets its OWN virtualenv because their pinned deps
# (basicsr, old torchvision, etc.) can conflict with the classifier env.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p external

# ---------------------------------------------------------------------
# Real-ESRGAN (xinntao/Real-ESRGAN)
# ---------------------------------------------------------------------
if [ ! -d "external/Real-ESRGAN" ]; then
  git clone https://github.com/xinntao/Real-ESRGAN.git external/Real-ESRGAN
else
  echo "external/Real-ESRGAN already exists, skipping clone."
fi

python3 -m venv external/Real-ESRGAN/.venv
source external/Real-ESRGAN/.venv/bin/activate
pip install --upgrade pip
pip install basicsr facexlib gfpgan
pip install -r external/Real-ESRGAN/requirements.txt
(cd external/Real-ESRGAN && python setup.py develop)

# --- Known compatibility fix -------------------------------------------
# basicsr (unmaintained since ~2022) imports
# `torchvision.transforms.functional_tensor`, which was removed in
# torchvision >= 0.17 (the function it needs, rgb_to_grayscale, still
# exists, just moved to `torchvision.transforms.functional`). Without
# this patch, inference_realesrgan.py fails immediately with:
#   ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
#
# We locate basicsr on disk via `find` rather than `import basicsr` --
# importing it is exactly what triggers the broken import chain we're
# trying to patch (chicken-and-egg).
BASICSR_DIR=$(find external/Real-ESRGAN/.venv -maxdepth 6 -type d -name basicsr -path "*/site-packages/*" | head -n1)
if [ -n "$BASICSR_DIR" ]; then
  echo "Patching basicsr at $BASICSR_DIR for torchvision >= 0.17 compatibility..."
  grep -rl "torchvision.transforms.functional_tensor" "$BASICSR_DIR" 2>/dev/null \
    | xargs -r sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g'
fi
deactivate

# ---------------------------------------------------------------------
# SwinIR (JingyunLiang/SwinIR)
# ---------------------------------------------------------------------
if [ ! -d "external/SwinIR" ]; then
  git clone https://github.com/JingyunLiang/SwinIR.git external/SwinIR
else
  echo "external/SwinIR already exists, skipping clone."
fi

python3 -m venv external/SwinIR/.venv
source external/SwinIR/.venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm opencv-python numpy requests
deactivate

echo ""
echo "Cloned. src/sr/run_realesrgan.py and src/sr/run_swinir.py call these"
echo "repos via subprocess using their OWN interpreters -- edit"
echo "CFG.realesrgan_repo / CFG.swinir_repo in configs/paths.yaml if you"
echo "move them, and point src/sr/run_*.py's sys.executable calls at"
echo "external/Real-ESRGAN/.venv/bin/python / external/SwinIR/.venv/bin/python"
echo "if you keep the envs separate from your main classifier env."
