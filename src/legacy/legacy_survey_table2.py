"""
Reproduces Table 2: the PRELIMINARY, UNCONTROLLED legacy-recipe survey.
Kept isolated from the controlled same-checkpoint pipeline (src/train,
src/eval) on purpose -- do not mix its numbers into Table 3/4/5.

For each of the 6 backbones, on Protocol A (legacy image-level split):
  - trains on Original images with SGDM, wd=0        (the "Reproduce" column)
  - trains on Real-ESRGAN images with AdamW, wd=1e-4  (uncontrolled SR run)
  - trains on SwinIR images with AdamW, wd=1e-4       (uncontrolled SR run)

Note this trains directly on SR pixels (not same-checkpoint) and changes
optimizer+wd at the same time as the input -- that confound is the whole
point of this table; see Section 4.1's framing.

This requires realesrgan/swinir pixels to already be precomputed
(scripts/05_precompute_sr.sh) since here we train ON them, not just
evaluate on them.

Usage:
    python -m src.legacy.legacy_survey_table2
    python -m src.legacy.legacy_survey_table2 --models vgg19 resnet50
"""
from __future__ import annotations
import argparse
import csv

import torch

from src.config import CFG
from src.data.dataset import make_loader
from src.train.models import build_model, MODEL_NAMES
from src.train.train import build_optimizer, run_epoch

torch.backends.cudnn.benchmark = True  # fixed 224x224 input shape throughout

RUNS = [
    {"name": "orig_sgdm", "variant": "orig", "optimizer": "sgdm", "lr": 1e-3, "wd": 0.0},
    {"name": "realesrgan_adamw", "variant": "realesrgan", "optimizer": "adamw", "lr": 1e-4, "wd": 1e-4},
    {"name": "swinir_adamw", "variant": "swinir", "optimizer": "adamw", "lr": 1e-4, "wd": 1e-4},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = CFG.protocol_a_csv

    out_csv = CFG.results_root / "table2_legacy_survey_raw.csv"
    write_header = not out_csv.exists()

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "run_name", "variant", "optimizer", "lr", "wd",
                              "test_accuracy"])

        for model_name in args.models:
            for run in RUNS:
                print(f"\n[legacy] {model_name} / {run['name']}")
                train_loader = make_loader(split_csv, "train", variant=run["variant"],
                                            batch_size=args.batch_size, shuffle=True)
                test_loader = make_loader(split_csv, "test", variant=run["variant"],
                                           batch_size=args.batch_size, shuffle=False)

                model = build_model(model_name, pretrained=True).to(device)
                optimizer = build_optimizer(model, run["optimizer"], run["lr"], run["wd"])
                use_amp = device.type == "cuda"
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

                for epoch in range(args.epochs):
                    train_loss, train_f1 = run_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler,
                                                      desc=f"{model_name}/{run['name']} epoch {epoch+1}/{args.epochs} [train]")
                    print(f"    epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f}")

                _, test_f1 = run_epoch(model, test_loader, optimizer, device, train=False, scaler=scaler,
                                        desc=f"{model_name}/{run['name']} [test]")
                # accuracy for Table 2 (matches paper's reporting for this table)
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                        pred = model(x).argmax(1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                test_acc = correct / total

                writer.writerow([model_name, run["name"], run["variant"],
                                  run["optimizer"], run["lr"], run["wd"], test_acc])
                print(f"  -> test accuracy = {100*test_acc:.2f}%")

    print(f"\n[legacy] wrote {out_csv}")


if __name__ == "__main__":
    main()
