"""
Train one backbone on Original images only. The resulting checkpoint is
later evaluated on Original/Bicubic/RealESRGAN/SwinIR test inputs by
src/eval/same_checkpoint_eval.py -- SR is never seen during training,
by design (Section 3.3, "same-checkpoint protocol").

Usage (Protocol B, fold 2, locked config):
    python -m src.train.train --model resnet50 --protocol b --fold 2

Usage (Protocol A, one-off, custom config -- used by the legacy survey):
    python -m src.train.train --model vgg19 --protocol a \
        --optimizer sgdm --lr 1e-3 --wd 0.0 --epochs 15 --batch-size 32 \
        --tag legacy_orig_sgdm
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import CFG
from src.data.dataset import make_loader
from src.train.models import build_model, MODEL_NAMES

# All inputs are PadResize224'd to a FIXED 224x224 shape (see
# src/data/dataset.py), so cuDNN can safely auto-tune the fastest
# convolution algorithms for that exact shape once and reuse them for
# every batch -- free speedup, no effect on results. (Would be unsafe /
# counter-productive only if input shapes varied between batches.)
torch.backends.cudnn.benchmark = True


def build_optimizer(model, optimizer_name: str, lr: float, wd: float):
    optimizer_name = optimizer_name.lower()
    if optimizer_name in ("sgdm", "sgd"):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def load_locked_config() -> dict:
    if not CFG.locked_config_path.exists():
        raise FileNotFoundError(
            f"{CFG.locked_config_path} not found. Run scripts/06_hp_search.sh first, "
            f"or pass --optimizer/--lr/--wd/--epochs/--batch-size explicitly."
        )
    with open(CFG.locked_config_path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, optimizer, device, train: bool, scaler=None, desc=""):
    """scaler: a torch.cuda.amp.GradScaler, or None to disable mixed precision
    (e.g. on CPU, or if you hit numerical-stability issues with a
    particular model/optimizer combo)."""
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss, all_y, all_pred = 0.0, [], []
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.set_grad_enabled(train):
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            all_y.extend(y.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

    n = len(all_y)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return total_loss / n, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--protocol", required=True, choices=["a", "b"])
    ap.add_argument("--fold", type=int, default=0, help="only used for protocol b")
    ap.add_argument("--tag", default=None, help="override checkpoint filename tag")

    # If any of these are passed explicitly, they override the locked config
    # (used by the legacy uncontrolled survey in src/legacy/).
    ap.add_argument("--optimizer", default=None, choices=["sgdm", "adamw"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                     help="mixed precision training (default: on, CUDA only)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                     help="disable mixed precision (e.g. if you hit NaN/instability)")
    ap.add_argument("--patience", type=int, default=7,
                     help="stop if val macro-F1 doesn't improve for this many "
                          "consecutive epochs (0 disables early stopping)")
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- resolve config: explicit CLI args override the locked config ----
    explicit = all(v is not None for v in
                    [args.optimizer, args.lr, args.wd, args.epochs, args.batch_size])
    if explicit:
        cfg = {"optimizer": args.optimizer, "lr": args.lr, "wd": args.wd,
               "epochs": args.epochs, "batch_size": args.batch_size}
    else:
        cfg = load_locked_config()
        for k, v in vars(args).items():
            if k in cfg and v is not None:
                cfg[k] = v

    print(f"[train] model={args.model} protocol={args.protocol} fold={args.fold}")
    print(f"[train] config={cfg}")

    # ---- data ----
    if args.protocol == "a":
        split_csv = CFG.protocol_a_csv
        tag = args.tag or f"{args.model}_protoA"
    else:
        split_csv = CFG.protocol_b_csv(args.fold)
        tag = args.tag or f"{args.model}_protoB_fold{args.fold}"

    train_loader = make_loader(split_csv, "train", variant="orig",
                                batch_size=cfg["batch_size"], shuffle=True,
                                num_workers=args.num_workers)
    val_split = "val" if args.protocol == "b" else "test"
    # Protocol A has no dedicated val split in this scaffold; if you need one,
    # add a --val-fraction option to src/data/splits.py's build_protocol_a.
    val_loader = make_loader(split_csv, val_split, variant="orig",
                              batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=args.num_workers)

    # ---- model / optimizer ----
    model = build_model(args.model, pretrained=True).to(device)
    optimizer = build_optimizer(model, cfg["optimizer"], cfg["lr"], cfg["wd"])

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    print(f"[train] mixed precision (AMP): {'ON' if use_amp else 'off'}")

    best_f1, best_state = -1.0, None
    epochs_since_improve = 0
    history = []
    t0 = time.time()
    stopped_early_at = None

    for epoch in range(cfg["epochs"]):
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler,
                                          desc=f"{tag} epoch {epoch+1}/{cfg['epochs']} [train]")
        val_loss, val_f1 = run_epoch(model, val_loader, optimizer, device, train=False, scaler=scaler,
                                      desc=f"{tag} epoch {epoch+1}/{cfg['epochs']} [val]")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1,
                         "val_loss": val_loss, "val_f1": val_f1})
        print(f"  epoch {epoch:03d}  train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if args.patience > 0 and epochs_since_improve >= args.patience:
                stopped_early_at = epoch
                print(f"  [train] early stopping: val macro-F1 hasn't improved for "
                      f"{args.patience} epochs (best={best_f1:.4f} at epoch "
                      f"{epoch - epochs_since_improve:03d})")
                break

    elapsed = time.time() - t0
    n_epochs_ran = len(history)
    ckpt_path = CFG.classifier_ckpt_root / f"{tag}.pt"
    torch.save({
        "model_name": args.model,
        "state_dict": best_state,
        "config": cfg,
        "protocol": args.protocol,
        "fold": args.fold,
        "best_val_macro_f1": best_f1,
        "stopped_early_at_epoch": stopped_early_at,
        "n_epochs_ran": n_epochs_ran,
    }, ckpt_path)

    log_path = CFG.train_logs_dir / f"{tag}.json"
    with open(log_path, "w") as f:
        json.dump({"tag": tag, "config": cfg, "history": history,
                    "elapsed_sec": elapsed, "best_val_macro_f1": best_f1,
                    "stopped_early_at_epoch": stopped_early_at,
                    "n_epochs_ran": n_epochs_ran}, f, indent=2)

    print(f"[train] done in {elapsed/60:.1f} min ({n_epochs_ran}/{cfg['epochs']} epochs ran). "
          f"best val macro-F1={best_f1:.4f}")
    print(f"[train] checkpoint -> {ckpt_path}")
    print(f"[train] log -> {log_path}")


if __name__ == "__main__":
    main()
