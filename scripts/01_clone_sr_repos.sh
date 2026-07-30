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
