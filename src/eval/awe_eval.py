"""
Cross-dataset same-checkpoint evaluation on AWE.

Each Protocol B checkpoint trained on EarVN1.0 Original images (6 backbones
x 5 folds) is applied, unchanged, to every AWE image under the four inputs
(Original, Bicubic, Real-ESRGAN, SwinIR). Nothing is trained on AWE, so the
five fold checkpoints of a backbone are five independently trained models
scored on the same 1,000 images.

Writes under the AWE config's results_root:
  table_awe_raw.csv                 one row per checkpoint x input
  predictions/<tag>__<variant>.csv  per-image predictions (for McNemar/bands)
  table_awe_summary.csv             per backbone x input, mean+-std over folds
  table_awe_flips.csv               McNemar per checkpoint x SR input, Holm-corrected
  table_awe_bands.csv               accuracy by native short-side band

Usage (from the repo root, after build_awe_metadata and SR precompute):
    EAR_SR_CONFIG=configs/paths_awe.yaml python -m src.eval.awe_eval --all
    EAR_SR_CONFIG=configs/paths_awe.yaml python -m src.eval.awe_eval --model resnet50 --fold 0
    EAR_SR_CONFIG=configs/paths_awe.yaml python -m src.eval.awe_eval --summarize
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.config import CFG
from src.eval.same_checkpoint_eval import load_checkpoint, evaluate_variant, VARIANTS
from src.eval.native_size import bucket_of, BUCKET_LABELS
from src.train.models import MODEL_NAMES

# Many short-lived DataLoaders in one process exhausted file descriptors in
# shift_adaptation.py; the file_system strategy avoids holding one fd per tensor.
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True

RAW = "table_awe_raw.csv"
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "recall_male", "recall_female"]


def split_csv():
    return CFG.splits_dir / "awe_test.csv"


def run_one(model_name, fold, device, batch_size):
    tag = f"{model_name}_protoB_fold{fold}"
    model, ckpt = load_checkpoint(tag)
    if ckpt.get("retrain", False):
        raise ValueError(f"{tag} is a matched-domain checkpoint; expected same-checkpoint.")
    rows = []
    for variant in VARIANTS:
        df, m = evaluate_variant(model, split_csv(), "test", variant, device, batch_size)
        df.to_csv(CFG.predictions_dir / f"{tag}__{variant}.csv", index=False)
        rows.append({"tag": tag, "model": model_name, "fold": fold, "variant": variant,
                     **m, "n_test": len(df)})
        print(f"  {tag:30s} {variant:10s} acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
              f"recM={m['recall_male']:.3f} recF={m['recall_female']:.3f}")
    new = pd.DataFrame(rows)
    path = CFG.results_root / RAW
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["tag"] != tag], new], ignore_index=True)
    new.to_csv(path, index=False)


def summarize():
    raw = pd.read_csv(CFG.results_root / RAW)
    g = raw.groupby(["model", "variant"])[METRICS]
    summary = (g.mean() * 100).round(2).join((g.std(ddof=1) * 100).round(2), rsuffix="_std")
    summary.to_csv(CFG.results_root / "table_awe_summary.csv")
    print("\n=== AWE, mean over the 5 fold checkpoints (%)")
    print(summary[METRICS].unstack("variant")["accuracy"].to_markdown())

    per = raw.set_index(["model", "fold", "variant"])[METRICS] * 100
    base = per.xs("orig", level="variant")
    delta = (per - base.loc[per.index.droplevel("variant")].values).drop(index="orig", level="variant")
    print("\n=== Change vs Original (pp), mean over backbones x folds")
    print(delta.groupby(level="variant").mean().round(2).to_markdown())
    print("backbones with male recall down / female recall up:")
    for v in ["realesrgan", "swinir"]:
        d = delta.xs(v, level="variant").groupby(level="model").mean()
        print(f"  {v}: {(d['recall_male'] < 0).sum()}/6, {(d['recall_female'] > 0).sum()}/6, "
              f"accuracy down {(d['accuracy'] < 0).sum()}/6")

    # Fold-level mean over backbones -> 95% CI of the accuracy change.
    print("\n=== 95% CI of the accuracy change (fold-level mean over backbones)")
    for v in ["bicubic", "realesrgan", "swinir"]:
        f = delta.xs(v, level="variant")["accuracy"].groupby(level="fold").mean()
        h = stats.t.ppf(0.975, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f))
        print(f"  {v:10s} {f.mean():+.2f} [{f.mean() - h:+.2f}, {f.mean() + h:+.2f}]")

    # McNemar per checkpoint x input, Holm over all of them.
    flips = []
    for tag in sorted(raw["tag"].unique()):
        o = pd.read_csv(CFG.predictions_dir / f"{tag}__orig.csv")[["image_id", "correct"]]
        for v in ["bicubic", "realesrgan", "swinir"]:
            s = pd.read_csv(CFG.predictions_dir / f"{tag}__{v}.csv")[["image_id", "correct"]]
            m = o.merge(s, on="image_id", suffixes=("_o", "_s"))
            b = int(((m.correct_o == 1) & (m.correct_s == 0)).sum())
            c = int(((m.correct_o == 0) & (m.correct_s == 1)).sum())
            a = int(((m.correct_o == 1) & (m.correct_s == 1)).sum())
            d = int(((m.correct_o == 0) & (m.correct_s == 0)).sum())
            flips.append({"tag": tag, "variant": v, "corrected": c, "corrupted": b,
                          "p": mcnemar([[a, b], [c, d]], exact=True).pvalue})
    flips = pd.DataFrame(flips)
    flips["p_holm"] = multipletests(flips["p"], method="holm")[1]
    flips.to_csv(CFG.results_root / "table_awe_flips.csv", index=False)
    print("\n=== Flips summed over checkpoints; checkpoints significant after Holm")
    print(flips.groupby("variant").agg(corrected=("corrected", "sum"), corrupted=("corrupted", "sum"),
                                       sig=("p_holm", lambda p: int((p < 0.05).sum())),
                                       n=("p_holm", "size")).to_markdown())

    # Native short-side bands, pooled over all checkpoints.
    bands = []
    for tag in sorted(raw["tag"].unique()):
        for v in VARIANTS:
            p = pd.read_csv(CFG.predictions_dir / f"{tag}__{v}.csv")
            p["bucket"] = p["short_side"].map(bucket_of)
            bands.append(p.assign(variant=v))
    bands = pd.concat(bands)
    table = (bands.groupby(["bucket", "variant"])["correct"].mean().unstack() * 100).reindex(BUCKET_LABELS)
    share = bands[bands.variant == "orig"].groupby("bucket").size() / (bands.variant == "orig").sum() * 100
    table["share_%"] = share.reindex(BUCKET_LABELS)
    table.round(2).to_csv(CFG.results_root / "table_awe_bands.csv")
    print("\n=== Accuracy by native short-side band (%), pooled")
    print(table.round(2).to_markdown())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every backbone x Protocol B fold")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--summarize", action="store_true", help="only aggregate existing results")
    args = ap.parse_args()

    if not args.summarize:
        CFG.ensure_dirs()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jobs = ([(m, f) for m in MODEL_NAMES for f in range(CFG.protocol_b.n_folds)]
                if args.all else [(args.model, args.fold)] if args.model else None)
        if jobs is None:
            ap.error("pass --model (+ --fold), --all, or --summarize")
        for m, f in jobs:
            print(f"[awe_eval] {m} fold {f}")
            run_one(m, f, device, args.batch_size)
    if args.all or args.summarize:
        summarize()


if __name__ == "__main__":
    main()
