"""
Paired statistics for the paper's text (CPU only, reads existing CSVs).

1. Matched-domain gaps (Table 6): for each model x variant, the per-fold
   paired difference Acc(train=test=variant) - Acc(train=test=Original),
   with its 95% t-interval, a paired t-test, and a TOST equivalence test
   against +-`--margin` pp. Also pools the 30 model x fold pairs per
   variant, and the 5 fold means (the more conservative unit, since the
   six models share each fold's test subjects).

2. Same-checkpoint operating-point shift (Table 3): mean change vs
   Original in accuracy, macro-F1, balanced accuracy and per-class recall,
   per protocol, and how many backbones show male recall falling /
   female recall rising.

Usage:
    python -m src.eval.paired_stats
    python -m src.eval.paired_stats --results-root ./result-ear-sr-gender
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config import CFG

VARIANTS = ["bicubic", "realesrgan", "swinir"]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "recall_male", "recall_female"]


def paired_summary(d: np.ndarray, margin: float) -> dict:
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    p_lo = stats.t.sf((mean + margin) / se, n - 1)   # H0: diff <= -margin
    p_hi = stats.t.cdf((mean - margin) / se, n - 1)  # H0: diff >= +margin
    return {"n": n, "mean_pp": mean, "sd_diff_pp": sd,
            "ci_lo_pp": mean - half, "ci_hi_pp": mean + half,
            "p_ttest": stats.ttest_1samp(d, 0).pvalue,
            "p_tost": max(p_lo, p_hi)}


def matched_domain(root: Path, margin: float):
    sc = pd.read_csv(root / "table3_same_checkpoint_raw.csv")
    md = pd.read_csv(root / "table_matched_domain_raw.csv")
    orig = (sc[(sc.protocol == "b") & (sc.variant == "orig")]
            .set_index(["model", "fold"])["accuracy"])
    md = md[(md.protocol == "b") & (md.train_variant == md.test_variant)]
    acc = md.set_index(["model", "fold", "train_variant"])["accuracy"].unstack()
    diff = acc[VARIANTS].sub(orig.loc[acc.index], axis=0) * 100

    rows = []
    for model, g in diff.groupby(level="model"):
        for v in VARIANTS:
            rows.append({"model": model, "variant": v, **paired_summary(g[v].values, margin)})
    cells = pd.DataFrame(rows)
    cells["p_ttest_holm"] = multipletests(cells["p_ttest"], method="holm")[1]
    cells["cell_sd_pp"] = [100 * md[(md.model == r.model) & (md.train_variant == r.variant)]
                           ["accuracy"].std(ddof=1) for r in cells.itertuples()]

    pooled = []
    fold_means = diff.groupby(level="fold").mean()
    for v in VARIANTS:
        d = diff[v].values
        pooled.append({"variant": v, "unit": "model x fold (30)", **paired_summary(d, margin),
                       "p_wilcoxon": stats.wilcoxon(d).pvalue, "n_positive": int((d > 0).sum())})
        pooled.append({"variant": v, "unit": "fold mean (5)",
                       **paired_summary(fold_means[v].values, margin)})
    # SR retrain vs Bicubic retrain: both are "second training runs"
    for v in ["realesrgan", "swinir"]:
        d = (diff[v] - diff["bicubic"]).groupby(level="fold").mean().values
        pooled.append({"variant": f"{v} - bicubic", "unit": "fold mean (5)",
                       **paired_summary(d, margin)})
    pooled = pd.DataFrame(pooled)

    print(f"=== Matched-domain paired gaps vs Original (pp), TOST margin +-{margin} pp")
    print(cells.round(3).to_markdown(index=False))
    print(f"\n|gap| < cell sd in {(cells.mean_pp.abs() < cells.cell_sd_pp).sum()}/{len(cells)} cells; "
          f"TOST p<0.05 in {(cells.p_tost < 0.05).sum()}/{len(cells)}; "
          f"t-test p<0.05 in {(cells.p_ttest < 0.05).sum()} (after Holm: "
          f"{(cells.p_ttest_holm < 0.05).sum()})")
    print("\n=== Pooled")
    print(pooled.round(3).to_markdown(index=False))
    cells.to_csv(root / "table_matched_domain_paired.csv", index=False)
    pooled.to_csv(root / "table_matched_domain_pooled.csv", index=False)


def operating_point(root: Path):
    sc = pd.read_csv(root / "table3_same_checkpoint_raw.csv")
    out = []
    for proto, g in sc.groupby("protocol"):
        per_model = g.groupby(["model", "variant"])[METRICS].mean() * 100
        base = per_model.xs("orig", level="variant")
        delta = per_model.sub(base, level="model").drop(index="orig", level="variant")
        mean = delta.groupby(level="variant").mean()
        mean["protocol"] = proto
        mean["n_models_male_recall_down"] = (delta["recall_male"] < 0).groupby(level="variant").sum()
        mean["n_models_female_recall_up"] = (delta["recall_female"] > 0).groupby(level="variant").sum()
        out.append(mean.reset_index())
    out = pd.concat(out, ignore_index=True)
    print("\n=== Same-checkpoint change vs Original (pp, mean over models)")
    print(out.round(2).to_markdown(index=False))
    out.to_csv(root / "table_operating_point_shift.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=None,
                     help="defaults to configs/paths.yaml results_root")
    ap.add_argument("--margin", type=float, default=1.0, help="TOST equivalence margin, pp")
    args = ap.parse_args()
    root = args.results_root or CFG.results_root
    matched_domain(root, args.margin)
    operating_point(root)


if __name__ == "__main__":
    main()
