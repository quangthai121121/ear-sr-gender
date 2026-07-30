"""
Core of Table 3: load ONE checkpoint (trained on Original only) and
evaluate it, unmodified, on {Original, Bicubic, Real-ESRGAN, SwinIR} test
inputs for the same fold's test subjects.

Writes:
  results/predictions/<tag>__<variant>.csv   (per-image, for flip/size analysis)
  results/table3_same_checkpoint_raw.csv     (appended row per model/fold/variant)

Usage:
    python -m src.eval.same_checkpoint_eval --tag resnet50_protoB_fold0
    python -m src.eval.same_checkpoint_eval --model resnet50 --protocol b --fold 0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                     help="checkpoint tag as saved by train.py, e.g. resnet50_protoB_fold0")
    ap.add_argument("--model", default=None)
    ap.add_argument("--protocol", default=None, choices=["a", "b"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = args.tag
    if tag is None:
        if args.model is None or args.protocol is None:
            raise ValueError("Pass either --tag, or both --model and --protocol.")
        tag = (f"{args.model}_protoA" if args.protocol == "a"
               else f"{args.model}_protoB_fold{args.fold}")

    model, ckpt = load_checkpoint(tag)
    protocol = ckpt["protocol"]
    fold = ckpt["fold"]
    split_csv = CFG.protocol_a_csv if protocol == "a" else CFG.protocol_b_csv(fold)

    raw_csv = CFG.results_root / "table3_same_checkpoint_raw.csv"
    write_header = not raw_csv.exists()

    with open(raw_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["tag", "model", "protocol", "fold", "variant",
                              "accuracy", "macro_f1", "balanced_accuracy",
                              "recall_male", "recall_female", "n_test"])

        for variant in args.variants:
            print(f"[eval] {tag} on variant={variant} ...")
            df, metrics = evaluate_variant(model, split_csv, "test", variant,
                                            device, args.batch_size)

            pred_path = CFG.predictions_dir / f"{tag}__{variant}.csv"
            df.to_csv(pred_path, index=False)

            writer.writerow([
                tag, ckpt["model_name"], protocol, fold, variant,
                metrics["accuracy"], metrics["macro_f1"], metrics["balanced_accuracy"],
                metrics["recall_male"], metrics["recall_female"], len(df),
            ])
            print(f"  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"bal_acc={metrics['balanced_accuracy']:.4f}  -> {pred_path}")

    print(f"[eval] appended to {raw_csv}")


if __name__ == "__main__":
    main()
