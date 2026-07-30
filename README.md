# Ear-SR-Gender: controlled same-checkpoint study

Implements the experimental pipeline from *"Does Super-Resolution Improve
Ear-Based Gender Classification?"* — a controlled, same-checkpoint
evaluation of Real-ESRGAN and SwinIR as preprocessing for ear-based gender
classification on EarVN1.0, under both a legacy image-level split
(Protocol A) and a subject-disjoint split (Protocol B).

## 0. What you need before starting

- A GPU machine with internet access (this scaffold assumes network access;
  it was built in a sandbox with no network, so nothing below has been
  executed — treat it as a well-tested *design*, not a verified run).
- ~50-100 GB free disk for cached SR outputs (28,412 images × 2 SR
  methods × 4x upsampled PNGs adds up).
- The **EarVN1.0** dataset, requested from the authors / downloaded from
  Mendeley Data: https://data.mendeley.com/datasets/yws3v3mwx3
  (registration/request may be required — see the dataset's license terms;
  redistribution is restricted).

## 1. The ONE file you edit: `configs/paths.yaml`

Every script reads paths from here via `src/config.py`. Open it and set:

```yaml
paths:
  data_root: "/absolute/path/to/EarVN1.0"        # must contain images/01 .. images/164
  processed_root: "/absolute/path/to/fast/disk"   # cached SR outputs go here
```

Everything else (`meta_root`, checkpoint paths, results paths) defaults to
locations inside this repo and normally doesn't need to change. If your
Mendeley download's `images/` subfolder is numbered differently than
`01..164`, or if 98/66 male/female no longer matches, fix
`subject_gender()` in `src/data/build_metadata.py` accordingly — **check
the readme that ships with your download first.**

## 2. Directory layout

```
ear-sr-gender/
├── configs/paths.yaml          <- edit this
├── external/                   <- SR repos get git-cloned here (step 3)
│   ├── Real-ESRGAN/
│   └── SwinIR/
├── checkpoints/
│   ├── sr/                     <- SR pretrained weights (step 4)
│   └── classifiers/            <- your trained gender-classifier checkpoints
├── data/
│   ├── raw/                    <- (unused if data_root points elsewhere)
│   ├── meta/                   <- images.csv, subjects.csv, splits/
│   └── processed/              <- (unused if processed_root points elsewhere)
├── src/                        <- all the code (see below)
├── scripts/                    <- numbered pipeline, run in order
└── results/                    <- final CSVs, predictions, figures
```

## 3. Run order

```bash
cd ear-sr-gender

# 0. Python env for the classifier/eval side
bash scripts/00_setup_env.sh
source .venv/bin/activate

# 1. Clone Real-ESRGAN + SwinIR from GitHub, install THEIR deps in
#    separate venvs (they pin older/conflicting package versions)
bash scripts/01_clone_sr_repos.sh

# 2. Download the exact SR checkpoints the paper uses:
#    RealESRGAN_x4plus.pth, 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
bash scripts/02_download_sr_checkpoints.sh

# 3. Scan data_root/images/<subject>/* -> data/meta/images.csv, subjects.csv
bash scripts/03_build_metadata.sh

# 4. Build Protocol A (legacy 50/50) and Protocol B (5-fold subject-disjoint) splits
bash scripts/04_make_splits.sh

# 5. Precompute Real-ESRGAN / SwinIR x4 outputs for the WHOLE dataset
#    (slow; resumable; run overnight)
bash scripts/05_precompute_sr.sh

# 6. Small HP grid on MobileNetV2/ResNet50/Swin-T -> locks configs/locked_config.yaml
bash scripts/06_hp_search.sh

# 7. Train all 6 backbones on Original images (Protocol A once + Protocol B x5 folds)
bash scripts/07_train_all.sh

# 8. Same-checkpoint evaluation: each trained checkpoint tested on
#    Original / Bicubic / Real-ESRGAN / SwinIR
bash scripts/08_eval_all.sh

# 9. Flip analysis (McNemar), native-size breakdown, final table aggregation
bash scripts/09_aggregate_and_flip.sh

# Optional: regenerate the preliminary UNCONTROLLED Table 2 survey
bash scripts/10_legacy_survey_table2.sh
```

After step 9, the paper's tables are at:

| Table | File |
|---|---|
| Table 2 (uncontrolled legacy survey) | `results/table2_legacy_survey_raw.csv` |
| Table 3 (same-checkpoint benchmark)  | `results/table3_same_checkpoint.csv` |
| Table 4 (Protocol A vs B)            | `results/table4_protocolA_vs_B.csv` |
| Table 5 (flips, Holm-corrected)      | `results/table5_flips.csv` |
| Figure 2 (native-size breakdown)     | `results/figure2_native_size__*.png` |

## 4. Design notes / things to double-check

- **Same-checkpoint protocol**: `src/train/train.py` only ever trains on
  `variant="orig"`. SR pixels are only touched at evaluation time
  (`src/eval/same_checkpoint_eval.py`), which is what makes ∆SR
  attributable to the input rather than to optimizer/WD (Section 3.3).
- **PadResize224 is shared** across all 4 variants (`src/data/dataset.py:
  pad_resize_224`) — aspect-ratio-preserving resize + letterbox padding,
  applied identically whether the source is Original, Bicubic×4,
  Real-ESRGAN×4, or SwinIR×4 output. This is what Eq. (1)-(4) specify.
- **Gender label rule** (`subject_gender()` in `build_metadata.py`):
  subjects 01-98 = male, 99-164 = female, per EarVN1.0's own
  documentation. Verify against your actual download before trusting this.
- **SwinIR output filename matching** (`src/sr/run_swinir.py`): the
  official repo's `main_test_swinir.py` writes outputs into its own
  `results/` subfolder with a naming convention that has changed across
  repo versions. The wrapper glob-matches by stem and lists candidate
  suffixes in `OUTPUT_GLOB_SUFFIXES` — if matching fails on your clone,
  open `external/SwinIR/main_test_swinir.py`, find the `cv2.imwrite(...)`
  call, and adjust that list.
- **Tile sizes** (`configs/paths.yaml -> sr.realesrgan_tile / sr.swinir_tile`):
  set to 400 by default to avoid GPU OOM on large source photos; increase
  if you have headroom, decrease (or set null for SwinIR) if you hit OOM.
- **Locked training config**: `src/train/hp_search.py` picks one
  optimizer/LR/WD by grid search on Protocol B fold 0's Original
  validation macro-F1 only (Section 3.4), then every Table 3/4/5 run uses
  that same locked config — this is NOT claimed to be globally optimal,
  just fixed so optimizer choice can't be confused with an SR effect.
- **Protocol A has no dedicated validation split** in this scaffold
  (`src/train/train.py` uses its test split for early-stopping selection
  when `--protocol a`, matching how the original 2020 paper trained). If
  you want a proper A-train/A-val/A-test three-way split, extend
  `build_protocol_a()` in `src/data/splits.py`.
- **McNemar + Holm correction**: `src/eval/flip_analysis.py` computes
  per-pair exact McNemar tests; `src/eval/aggregate.py` applies Holm
  correction across *all* model×SR-variant tests found, per Section 4.4.

## 5. Sanity checks worth running before trusting results

1. After step 3, `data/meta/subjects.csv` should show 98 male / 66 female
   subjects, and `images.csv` should total 28,412 rows.
2. After step 4, the Protocol B overlap check printed by
   `src/data/splits.py` should report **0 pairwise overlaps** between
   folds' test-subject sets.
3. Spot-check a handful of `data/processed/realesrgan/<subject>/*.png`
   and `.../swinir_large/<subject>/*.png` visually before running the full
   precompute — confirm they're actually 4x upsampled/sharpened, not
   corrupted or blank.
4. In `results/table5_flips.csv`, `net_gain_pp` should match
   `Acc(SR) - Acc(Original)` from `table3_same_checkpoint_raw.csv` for the
   same model/fold (this is asserted implicitly by construction — worth
   spot-checking once by hand).
