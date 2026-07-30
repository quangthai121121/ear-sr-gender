#!/usr/bin/env bash
# OPTIONAL: reproduces the preliminary, uncontrolled Table 2 survey
# (trains directly on SR pixels with a different optimizer than the
# Original baseline). Independent of the controlled same-checkpoint
# pipeline (scripts 06-09) -- run this only if you want to regenerate
# Table 2 itself. Requires scripts/05_precompute_sr.sh to have run first.
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate .venv only if you haven't already activated an environment
# yourself (checks $VIRTUAL_ENV). If you ran `source .venv/bin/activate`
# (or activated any other venv/conda env) before calling this script,
# it is left alone and whatever's already active is used as-is.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
python -m src.legacy.legacy_survey_table2
echo ""
echo "Raw results -> results/table2_legacy_survey_raw.csv"
