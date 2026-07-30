#!/usr/bin/env bash
# Standalone fix for an already-created external/Real-ESRGAN/.venv (e.g. if
# you ran scripts/01_clone_sr_repos.sh before this patch step was added).
# Safe to re-run; idempotent.
#
# basicsr (unmaintained) imports torchvision.transforms.functional_tensor,
# which was removed in torchvision >= 0.17. rgb_to_grayscale (the only
# thing basicsr needs from it) still exists at torchvision.transforms.functional.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY="external/Real-ESRGAN/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: $VENV_PY not found. Run scripts/01_clone_sr_repos.sh first."
  exit 1
fi

BASICSR_DIR=$("$VENV_PY" -c "import basicsr, os; print(os.path.dirname(basicsr.__file__))")
if [ -z "$BASICSR_DIR" ]; then
  echo "ERROR: could not locate basicsr install. Is it installed in that venv?"
  exit 1
fi

echo "Patching basicsr at $BASICSR_DIR ..."
MATCHES=$(grep -rl "torchvision.transforms.functional_tensor" "$BASICSR_DIR" 2>/dev/null || true)
if [ -z "$MATCHES" ]; then
  echo "No matching imports found -- already patched, or your basicsr version doesn't need it."
else
  echo "$MATCHES" | xargs sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g'
  echo "Patched files:"
  echo "$MATCHES"
fi

echo ""
echo "Done. Re-run: python -m src.sr.run_realesrgan --subjects 1 --limit 3"
