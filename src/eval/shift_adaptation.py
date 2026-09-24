"""
Label-free test-time adaptation of the SAME Original-trained checkpoint to
SR inputs -- a direct test of the distribution-shift explanation for the
same-checkpoint SR penalty (Table 3). No weights are trained and no labels
are used; the adaptation set is each fold's held-out VALIDATION subjects
(never used for weight updates, only for early stopping), read as
unlabeled images.

Methods, per (checkpoint, SR variant):

  none        the checkpoint as-is (reproduces Table 3's SR column)
  prior       shift the female-vs-male logit margin by a scalar b so that,
              on the unlabeled val images, the fraction predicted female
              under SR equals the fraction predicted female under Original
              (i.e. restore the classifier's own Original operating point)
  adabn       re-estimate every BatchNorm layer's running mean/var on the
              unlabeled SR val images (Li et al., AdaBN); models without
              BatchNorm (VGG19, Swin-T) are skipped
  adabn+prior both, prior computed after AdaBN

Writes results/table_shift_adaptation_raw.csv (one row per
tag x variant x method; re-running a tag replaces its rows).

Usage:
    python -m src.eval.shift_adaptation --model resnet50 --fold 0
    python -m src.eval.shift_adaptation --all
    python -m src.eval.shift_adaptation --summarize
"""
from __future__ import annotations
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import CFG
from src.data.dataset import make_loader, GENDER_TO_IDX
from src.eval.metrics import compute_metrics
from src.eval.same_checkpoint_eval import load_checkpoint
from src.train.models import MODEL_NAMES

torch.backends.cudnn.benchmark = True  # fixed 224x224 input shape throughout

SR_VARIANTS = ["realesrgan", "swinir"]
MALE, FEMALE = GENDER_TO_IDX["male"], GENDER_TO_IDX["female"]
RAW_CSV_NAME = "table_shift_adaptation_raw.csv"


@torch.no_grad()
def collect_margins(model, split_csv, split_name, variant, device, batch_size):
    """Returns (margin, y) with margin = logit_female - logit_male, so
    predicting female <=> margin > 0."""
    loader = make_loader(split_csv, split_name, variant=variant,
                          batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    margins, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        logits = logits.float().cpu()
        margins.append(logits[:, FEMALE] - logits[:, MALE])
        ys.append(y)
    return torch.cat(margins).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def adabn(model, split_csv, split_name, variant, device, batch_size):
    """Copy of `model` whose BatchNorm running stats are re-estimated on
    `variant` images (cumulative average over one pass). Returns None if
    the model has no BatchNorm layers."""
    m = copy.deepcopy(model).to(device)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return None
    m.eval()  # dropout / stochastic depth stay off
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average
        bn.train()
    loader = make_loader(split_csv, split_name, variant=variant,
                          batch_size=batch_size, shuffle=True)
    for x, _ in loader:
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            m(x.to(device, non_blocking=True))
    return m.eval()


def matching_bias(margin_src, target_female_rate):
    """Scalar b such that mean(margin_src + b > 0) ~= target_female_rate."""
    return -float(np.quantile(margin_src, 1.0 - target_female_rate))


def score(margin, y, bias=0.0):
    pred = np.where(margin + bias > 0, FEMALE, MALE)
    m = compute_metrics(y, pred)
    m["female_pred_rate"] = float((pred == FEMALE).mean())
    return m


def run_one(model_name, fold, variants, device, batch_size):
    tag = f"{model_name}_protoB_fold{fold}"
    model, ckpt = load_checkpoint(tag)
    if ckpt.get("retrain", False):
        raise ValueError(f"{tag} is a matched-domain checkpoint; expected same-checkpoint.")
    split_csv = CFG.protocol_b_csv(fold)

    val_orig, _ = collect_margins(model, split_csv, "val", "orig", device, batch_size)
    target_rate = float((val_orig > 0).mean())
    test_orig, y_orig = collect_margins(model, split_csv, "test", "orig", device, batch_size)

    rows = [dict(variant="orig", method="none", bias=0.0, **score(test_orig, y_orig))]
    for variant in variants:
        test_m, y = collect_margins(model, split_csv, "test", variant, device, batch_size)
        val_m, _ = collect_margins(model, split_csv, "val", variant, device, batch_size)
        b = matching_bias(val_m, target_rate)
        rows.append(dict(variant=variant, method="none", bias=0.0, **score(test_m, y)))
        rows.append(dict(variant=variant, method="prior", bias=b, **score(test_m, y, b)))

        m_bn = adabn(model, split_csv, "val", variant, device, batch_size)
        if m_bn is not None:
            test_bn, y_bn = collect_margins(m_bn, split_csv, "test", variant, device, batch_size)
            val_bn, _ = collect_margins(m_bn, split_csv, "val", variant, device, batch_size)
            b_bn = matching_bias(val_bn, target_rate)
            rows.append(dict(variant=variant, method="adabn", bias=0.0, **score(test_bn, y_bn)))
            rows.append(dict(variant=variant, method="adabn+prior", bias=b_bn,
                             **score(test_bn, y_bn, b_bn)))
            del m_bn

    for r in rows:
        r.update(tag=tag, model=model_name, fold=fold, n_test=len(y_orig),
                 val_orig_female_rate=target_rate)
        print(f"  {tag:32s} {r['variant']:10s} {r['method']:12s} "
              f"acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} "
              f"recM={r['recall_male']:.3f} recF={r['recall_female']:.3f} "
              f"femRate={r['female_pred_rate']:.3f} b={r['bias']:+.3f}")
    return pd.DataFrame(rows)


def save_rows(new: pd.DataFrame):
    path = CFG.results_root / RAW_CSV_NAME
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["tag"].isin(new["tag"].unique())], new], ignore_index=True)
    cols = ["tag", "model", "fold", "variant", "method", "accuracy", "macro_f1",
            "balanced_accuracy", "recall_male", "recall_female", "female_pred_rate",
            "bias", "val_orig_female_rate", "n_test"]
    new[cols].to_csv(path, index=False)
    print(f"[shift_adaptation] wrote {path}")


def summarize():
    path = CFG.results_root / RAW_CSV_NAME
    df = pd.read_csv(path)
    orig = df[df["variant"] == "orig"].set_index(["model", "fold"])["accuracy"]
    sr = df[df["variant"] != "orig"].copy()
    sr["delta_pp"] = 100 * (sr["accuracy"] - orig.loc[list(zip(sr["model"], sr["fold"]))].values)
    for metric in ["delta_pp", "balanced_accuracy", "recall_male", "recall_female"]:
        t = (sr.groupby(["model", "variant", "method"])[metric].mean()
             .unstack("method"))
        if metric != "delta_pp":
            t = 100 * t
        print(f"\n{metric} (mean over folds):")
        print(t.round(2).to_markdown())
    print("\nMean delta_pp over models (only models present for every method in that column):")
    print(sr.groupby(["variant", "method"])["delta_pp"].mean().unstack("method").round(2).to_markdown())
    out = CFG.results_root / "table_shift_adaptation.csv"
    (sr.groupby(["model", "variant", "method"])
       [["delta_pp", "accuracy", "balanced_accuracy", "recall_male", "recall_female"]]
       .agg(["mean", "std"]).to_csv(out))
    print(f"\n[shift_adaptation] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every model x every Protocol B fold")
    ap.add_argument("--variants", nargs="*", default=SR_VARIANTS,
                     choices=["bicubic", "realesrgan", "swinir"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--summarize", action="store_true", help="only aggregate existing raw CSV")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        jobs = [(m, f) for m in MODEL_NAMES for f in range(CFG.protocol_b.n_folds)]
    elif args.model:
        jobs = [(args.model, args.fold)]
    else:
        ap.error("pass --model (+ --fold), --all, or --summarize")

    for model_name, fold in jobs:
        print(f"[shift_adaptation] {model_name} fold {fold}")
        save_rows(run_one(model_name, fold, args.variants, device, args.batch_size))
    summarize()


if __name__ == "__main__":
    main()
