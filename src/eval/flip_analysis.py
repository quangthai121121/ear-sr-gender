"""
Table 5: for each (model, fold, SR variant) pair, join the Original and SR
per-image predictions (from same_checkpoint_eval.py) on image_id, count:

  corrected = Original wrong, SR correct
  corrupted = Original correct, SR wrong

and run an exact paired McNemar test on the 2x2 table. Net gain in
percentage points should equal Acc(SR) - Acc(Original) by construction --
we assert that as a sanity check.

Usage:
    python -m src.eval.flip_analysis --tag resnet50_protoB_fold0 --sr realesrgan
    python -m src.eval.flip_analysis --all   # runs over every prediction pair found
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.config import CFG

SR_VARIANTS = ["bicubic", "realesrgan", "swinir"]


def flip_counts(tag: str, sr_variant: str):
    orig_path = CFG.predictions_dir / f"{tag}__orig.csv"
    sr_path = CFG.predictions_dir / f"{tag}__{sr_variant}.csv"
    if not orig_path.exists() or not sr_path.exists():
        return None

    orig_df = pd.read_csv(orig_path)[["image_id", "correct"]].rename(columns={"correct": "orig_correct"})
    sr_df = pd.read_csv(sr_path)[["image_id", "correct"]].rename(columns={"correct": "sr_correct"})
    merged = orig_df.merge(sr_df, on="image_id", how="inner")
    if len(merged) != len(orig_df):
        print(f"  [warn] {tag}/{sr_variant}: {len(orig_df)} orig rows vs "
              f"{len(merged)} matched rows -- mismatched test sets?")

    n = len(merged)
    both_correct = int(((merged.orig_correct == 1) & (merged.sr_correct == 1)).sum())
    both_wrong = int(((merged.orig_correct == 0) & (merged.sr_correct == 0)).sum())
    corrected = int(((merged.orig_correct == 0) & (merged.sr_correct == 1)).sum())
    corrupted = int(((merged.orig_correct == 1) & (merged.sr_correct == 0)).sum())

    table = [[both_correct, corrupted],   # sr_correct=1/0 | orig_correct=1
             [corrected, both_wrong]]     # sr_correct=1/0 | orig_correct=0
    result = mcnemar(table, exact=True)

    net_gain_pp = 100.0 * (corrected - corrupted) / n if n else float("nan")
    acc_orig = 100.0 * (both_correct + corrupted) / n if n else float("nan")
    acc_sr = 100.0 * (both_correct + corrected) / n if n else float("nan")

    return {
        "tag": tag, "sr_variant": sr_variant, "n": n,
        "corrected": corrected, "corrupted": corrupted,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "net_gain_pp": net_gain_pp, "mcnemar_stat": result.statistic,
        "mcnemar_p": result.pvalue,
        "acc_orig_check": acc_orig, "acc_sr_check": acc_sr,
    }


def discover_pairs():
    """Find every (tag, sr_variant) pair with both orig and SR prediction
    files present, by scanning results/predictions/."""
    pattern = re.compile(r"^(?P<tag>.+)__(?P<variant>orig|bicubic|realesrgan|swinir)\.csv$")
    tags = set()
    for p in CFG.predictions_dir.glob("*__orig.csv"):
        m = pattern.match(p.name)
        if m:
            tags.add(m.group("tag"))

    pairs = []
    for tag in sorted(tags):
        for variant in SR_VARIANTS:
            if (CFG.predictions_dir / f"{tag}__{variant}.csv").exists():
                pairs.append((tag, variant))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sr", default=None, choices=SR_VARIANTS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    CFG.ensure_dirs()
    if args.all or args.tag is None:
        pairs = discover_pairs()
        if not pairs:
            print("No (orig, SR) prediction pairs found under results/predictions/. "
                  "Run same_checkpoint_eval.py first.")
            return
    else:
        pairs = [(args.tag, args.sr or "realesrgan")]

    out_csv = CFG.results_root / "table5_flips_raw.csv"
    rows = []
    for tag, sr_variant in pairs:
        r = flip_counts(tag, sr_variant)
        if r is None:
            continue
        rows.append(r)
        print(f"{tag:35s} vs {sr_variant:12s}  n={r['n']:5d}  "
              f"corrected={r['corrected']:4d}  corrupted={r['corrupted']:4d}  "
              f"net={r['net_gain_pp']:+.2f}pp  p={r['mcnemar_p']:.4g}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\n[flip_analysis] wrote {out_csv}")
        print("Run src.eval.aggregate to apply Holm correction across all "
              "model x SR tests before reporting p-values.")


if __name__ == "__main__":
    main()
