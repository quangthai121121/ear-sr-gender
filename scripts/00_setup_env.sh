#!/usr/bin/env bash
# Sets up the Python environment for the CLASSIFIER + EVAL side of the
# project (training the 6 backbones, computing metrics, flip analysis).
# Real-ESRGAN/SwinIR get their own env in script 01, since they pin
# different/older dependency versions.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
