#!/usr/bin/env bash
# Downloads the exact pretrained SR checkpoints referenced in the paper draft
# (Section 3.2: "public checkpoints [exact Real-ESRGAN and SwinIR weight
# names/versions]"). Fills that TBD with these two:
#
#   Real-ESRGAN: RealESRGAN_x4plus.pth        (general x4 RRDBNet, v0.1.0)
#   SwinIR:      003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
#                (SwinIR-Large, real-world SR, x4, GAN-trained, v0.0)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p checkpoints/sr/realesrgan checkpoints/sr/swinir

REALESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_OUT="checkpoints/sr/realesrgan/RealESRGAN_x4plus.pth"

SWINIR_URL="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
SWINIR_OUT="checkpoints/sr/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

if [ ! -f "$REALESRGAN_OUT" ]; then
  echo "Downloading Real-ESRGAN checkpoint..."
  wget -q --show-progress "$REALESRGAN_URL" -O "$REALESRGAN_OUT"
else
  echo "Real-ESRGAN checkpoint already present, skipping."
fi

if [ ! -f "$SWINIR_OUT" ]; then
  echo "Downloading SwinIR-Large checkpoint..."
  wget -q --show-progress "$SWINIR_URL" -O "$SWINIR_OUT"
else
  echo "SwinIR-Large checkpoint already present, skipping."
fi

echo ""
echo "Verify against configs/paths.yaml -> paths.realesrgan_ckpt / swinir_large_ckpt"
echo "(filenames must match exactly, they are used as-is)."
sha256sum "$REALESRGAN_OUT" "$SWINIR_OUT" 2>/dev/null || true
