"""
Section 3.4: "one common training configuration selected on Original
validation macro-F1 only, using an optimizer/LR/WD search on MobileNetV2,
ResNet50, and Swin-T, and then locked across test variants."

This script runs that small grid (on Protocol B, fold 0, Original inputs
only) and writes the winning config to configs/locked_config.yaml, which
src/train.train.py reads by default for every subsequent run (Table 3/4/5).

Usage:
    python -m src.train.hp_search
    python -m src.train.hp_search --epochs 8   # shorter grid, then re-train
                                                # the winner for full epochs
                                                # via src.train.train
"""
from __future__ import annotations
import argparse
import itertools
import json

import torch
import yaml

from src.config import CFG
from src.data.dataset import make_loader
from src.train.models import build_model
from src.train.train import build_optimizer, run_epoch

torch.backends.cudnn.benchmark = True  # fixed 224x224 input shape throughout

GRID_MODELS = ["mobilenet_v2", "resnet50", "swin_t"]
GRID_OPTIMIZERS = ["sgdm", "adamw"]
GRID_LR = {"sgdm": [1e-2, 1e-3], "adamw": [1e-3, 1e-4]}
GRID_WD = [0.0, 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8, help="short grid epochs")
    ap.add_argument("--final-epochs", type=int, default=20,
                     help="value written into locked_config.yaml for the real runs")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = CFG.protocol_b_csv(args.fold)

    results = []
    for model_name, optimizer_name in itertools.product(GRID_MODELS, GRID_OPTIMIZERS):
        for lr in GRID_LR[optimizer_name]:
            for wd in GRID_WD:
                print(f"\n[hp_search] {model_name} / {optimizer_name} / lr={lr} / wd={wd}")
                train_loader = make_loader(split_csv, "train", variant="orig",
                                            batch_size=args.batch_size, shuffle=True)
                val_loader = make_loader(split_csv, "val", variant="orig",
                                          batch_size=args.batch_size, shuffle=False)

                model = build_model(model_name, pretrained=True).to(device)
                optimizer = build_optimizer(model, optimizer_name, lr, wd)
                use_amp = device.type == "cuda"
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

                best_f1 = -1.0
                for epoch in range(args.epochs):
                    _, train_f1 = run_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler,
                                             desc=f"{model_name}/{optimizer_name}/lr={lr}/wd={wd} ep{epoch+1} [train]")
                    _, val_f1 = run_epoch(model, val_loader, optimizer, device, train=False, scaler=scaler,
                                           desc=f"{model_name}/{optimizer_name}/lr={lr}/wd={wd} ep{epoch+1} [val]")
                    best_f1 = max(best_f1, val_f1)
                    print(f"    epoch {epoch}: train_f1={train_f1:.4f} val_f1={val_f1:.4f}")

                results.append({"model": model_name, "optimizer": optimizer_name,
                                 "lr": lr, "wd": wd, "best_val_macro_f1": best_f1})

    results.sort(key=lambda r: r["best_val_macro_f1"], reverse=True)
    print("\n[hp_search] ranked results:")
    for r in results:
        print(f"  {r}")

    winner = results[0]
    locked = {
        "optimizer": winner["optimizer"],
        "lr": winner["lr"],
        "wd": winner["wd"],
        "epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "selected_from": {"model": winner["model"], "fold": args.fold,
                           "grid_epochs": args.epochs,
                           "best_val_macro_f1": winner["best_val_macro_f1"]},
    }
    with open(CFG.locked_config_path, "w") as f:
        yaml.safe_dump(locked, f, sort_keys=False)

    log_path = CFG.results_root / "hp_search_grid.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[hp_search] locked config -> {CFG.locked_config_path}")
    print(f"[hp_search] full grid log -> {log_path}")
    print("NOTE: this locked config is used for every Table 3/4/5 run. It is "
          "NOT claimed to be globally optimal (see Section 3.4).")


if __name__ == "__main__":
    main()
