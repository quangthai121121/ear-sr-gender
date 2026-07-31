"""
Evaluate ONE trained checkpoint. Two output destinations depending on how
the checkpoint was trained (src/train/train.py's --retrain flag):

  Same-checkpoint checkpoints (retrain=False, the main protocol):
    load the checkpoint (trained on Original only) and evaluate it,
    unmodified, on {Original, Bicubic, Real-ESRGAN, SwinIR} test inputs.
    Writes to results/table3_same_checkpoint_raw.csv -- SAME schema as
    always, so this file and everything already aggregated from it stays
    valid no matter how many matched-domain runs you do elsewhere.

  Matched-domain checkpoints (retrain=True, secondary analysis):
    by default, evaluate ONLY on the variant the checkpoint was trained
    on (train_variant == test_variant, the core "matched-domain" question)
    -- pass --variants explicitly to also test cross-domain generalization
    if you want that too. Writes to a SEPARATE file,
    results/table_matched_domain_raw.csv, which table3_same_checkpoint_raw.csv
    never sees rows from and vice versa.

Both modes also write per-image predictions to:
  results/predictions/<tag>__<variant>.csv

Usage:
    python -m src.eval.same_checkpoint_eval --tag resnet50_protoB_fold0
    python -m src.eval.same_checkpoint_eval --model resnet50 --protocol b --fold 0
    python -m src.eval.same_checkpoint_eval --model resnet50 --protocol b --fold 0 \
        --retrain --variant realesrgan
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import torch
import pandas as pd

from src.config import CFG
from src.data.dataset import make_loader, IDX_TO_GENDER
from src.train.models import build_model
from src.eval.metrics import compute_metrics

torch.backends.cudnn.benchmark = True  # fixed 224x224 input shape throughout

VARIANTS = ["orig", "bicubic", "realesrgan", "swinir"]


def load_checkpoint(tag: str):
    ckpt_path = CFG.classifier_ckpt_root / f"{tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Train it first with src.train.train.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


@torch.no_grad()
def evaluate_variant(model, split_csv, split_name, variant, device, batch_size=32):
    loader = make_loader(split_csv, split_name, variant=variant,
                          batch_size=batch_size, shuffle=False, return_meta=True)
    model.eval().to(device)
    use_amp = device.type == "cuda"

    rows = []
    for x, y, image_id, short_side in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(1).cpu().tolist()
        y = y.tolist()
        for iid, yt, yp, ss in zip(image_id, y, pred, short_side.tolist()):
            rows.append({"image_id": iid, "y_true": yt, "y_pred": yp,
                         "correct": int(yt == yp), "short_side": ss})

    df = pd.DataFrame(rows)
    metrics = compute_metrics(df["y_true"], df["y_pred"])
    return df, metrics


def resolve_tag(args) -> str:
    if args.tag:
        return args.tag
    if args.model is None or args.protocol is None:
        raise ValueError("Pass either --tag, or --model + --protocol (+ --fold, "
                          "+ --retrain/--variant for a matched-domain checkpoint).")
    base_tag = (f"{args.model}_protoA" if args.protocol == "a"
                else f"{args.model}_protoB_fold{args.fold}")
    if args.retrain:
        return f"{base_tag}_retrain_{args.variant}"
    return base_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                     help="checkpoint tag as saved by train.py, e.g. resnet50_protoB_fold0 "
                          "or resnet50_protoB_fold0_retrain_realesrgan")
    ap.add_argument("--model", default=None)
    ap.add_argument("--protocol", default=None, choices=["a", "b"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                     help="only used to help reconstruct --tag when --tag isn't "
                          "passed directly -- must match how the checkpoint was trained")
    ap.add_argument("--variant", default="orig",
                     choices=["orig", "bicubic", "realesrgan", "swinir"],
                     help="only used to help reconstruct --tag for a --retrain checkpoint")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--variants", nargs="*", default=None, choices=VARIANTS,
                     help="which variants to TEST on. Default: all 4 for a "
                          "same-checkpoint model; just the model's own training "
                          "variant for a matched-domain model (pass this "
                          "explicitly to override, e.g. to test cross-domain).")
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = resolve_tag(args)
    model, ckpt = load_checkpoint(tag)
    protocol = ckpt["protocol"]
    fold = ckpt["fold"]
    # .get(...) defaults keep this compatible with checkpoints saved before
    # --retrain existed (they're always same-checkpoint / retrain=False).
    ckpt_retrain = ckpt.get("retrain", False)
    ckpt_train_variant = ckpt.get("train_variant", "orig")
    split_csv = CFG.protocol_a_csv if protocol == "a" else CFG.protocol_b_csv(fold)

    if args.variants is not None:
        variants_to_test = args.variants
    elif ckpt_retrain:
        variants_to_test = [ckpt_train_variant]
    else:
        variants_to_test = VARIANTS

    if ckpt_retrain:
        raw_csv = CFG.results_root / "table_matched_domain_raw.csv"
        header = ["tag", "model", "protocol", "fold", "train_variant", "test_variant",
                  "accuracy", "macro_f1", "balanced_accuracy",
                  "recall_male", "recall_female", "n_test"]
    else:
        raw_csv = CFG.results_root / "table3_same_checkpoint_raw.csv"
        header = ["tag", "model", "protocol", "fold", "variant",
                  "accuracy", "macro_f1", "balanced_accuracy",
                  "recall_male", "recall_female", "n_test"]

    write_header = not raw_csv.exists()
    print(f"[eval] tag={tag}  retrain={ckpt_retrain}  train_variant={ckpt_train_variant}")
    print(f"[eval] testing on variants={variants_to_test}")
    print(f"[eval] writing raw rows -> {raw_csv}")

    with open(raw_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for variant in variants_to_test:
            print(f"[eval] {tag} on variant={variant} ...")
            df, metrics = evaluate_variant(model, split_csv, "test", variant,
                                            device, args.batch_size)

            pred_path = CFG.predictions_dir / f"{tag}__{variant}.csv"
            df.to_csv(pred_path, index=False)

            if ckpt_retrain:
                row = [tag, ckpt["model_name"], protocol, fold, ckpt_train_variant, variant,
                       metrics["accuracy"], metrics["macro_f1"], metrics["balanced_accuracy"],
                       metrics["recall_male"], metrics["recall_female"], len(df)]
            else:
                row = [tag, ckpt["model_name"], protocol, fold, variant,
                       metrics["accuracy"], metrics["macro_f1"], metrics["balanced_accuracy"],
                       metrics["recall_male"], metrics["recall_female"], len(df)]
            writer.writerow(row)
            print(f"  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"bal_acc={metrics['balanced_accuracy']:.4f}  -> {pred_path}")

    print(f"[eval] appended to {raw_csv}")


if __name__ == "__main__":
    main()
