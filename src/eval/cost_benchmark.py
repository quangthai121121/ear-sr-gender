"""
Deployment cost of the options in the paper's decision rule (Reviewer 1.3).

Measures, on the current GPU and the default EarVN1.0 config:
  1. SR preprocessing per image (Real-ESRGAN x4, SwinIR-L x4), end to end
     through the same precompute code as the paper (read, SR, write PNG).
     A 1-image run is timed separately and subtracted, so model loading and
     process start-up are reported apart from the per-image cost.
  2. Classifier inference per image for each backbone (224x224, AMP, as in
     evaluation), at batch size 1 (latency) and 64 (throughput).
  3. Label-free adaptation on one fold's unlabeled validation images:
     threshold recalibration (one forward pass + a quantile) and AdaBN
     (one BatchNorm re-estimation pass), for the BatchNorm backbones.

Writes results/table_cost.csv and prints a summary.

Usage:
    python -m src.eval.cost_benchmark
    python -m src.eval.cost_benchmark --n-sr 100 --fold 0
"""
from __future__ import annotations
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

from src.config import CFG
from src.eval.native_size import bucket_of, BUCKET_LABELS
from src.eval.same_checkpoint_eval import load_checkpoint
from src.eval import shift_adaptation as sa
from src.sr import run_realesrgan, run_swinir
from src.train.models import MODEL_NAMES

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
BN_MODELS = ["mobilenet_v2", "resnet50", "efficientnet_b0", "maxvit_t"]


def sample_images(n: int, seed: int = 0) -> list[str]:
    """n test-set images spread evenly over the four native-size bands."""
    df = pd.read_csv(CFG.images_csv)
    df["bucket"] = df["short_side"].map(bucket_of)
    per = n // len(BUCKET_LABELS)
    parts = [df[df.bucket == b].sample(min(per, (df.bucket == b).sum()), random_state=seed)
             for b in BUCKET_LABELS]
    return list(pd.concat(parts)["rel_path"])


def time_sr(module, rel_paths: list[str]) -> dict:
    """Wall time for 1 image and for all images, each into a fresh output dir."""
    out = {}
    for label, paths in [("one", rel_paths[:1]), ("all", rel_paths)]:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            module.run_one_batch(paths, Path(tmp))
            out[label] = time.perf_counter() - t0
    per_image = (out["all"] - out["one"]) / (len(rel_paths) - 1)
    return {"n_images": len(rel_paths), "startup_s": out["one"] - per_image,
            "per_image_ms": 1000 * per_image}


@torch.no_grad()
def time_classifier(model, device, batch_size: int, iters: int = 50) -> float:
    """Milliseconds per image for a forward pass at the given batch size."""
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    model.eval().to(device)
    for _ in range(10):
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / (iters * batch_size)


def time_adaptation(model_name: str, fold: int, device, batch_size: int) -> dict:
    model, _ = load_checkpoint(f"{model_name}_protoB_fold{fold}")
    split_csv = CFG.protocol_b_csv(fold)
    n_val = int((pd.read_csv(split_csv)["split"] == "val").sum())
    t0 = time.perf_counter()
    margins, _ = sa.collect_margins(model, split_csv, "val", "realesrgan", device, batch_size)
    sa.matching_bias(margins, 0.38)
    t_thr = time.perf_counter() - t0
    row = {"model": model_name, "n_val_images": n_val, "threshold_recal_s": t_thr}
    if model_name in BN_MODELS:
        t0 = time.perf_counter()
        sa.adabn(model, split_csv, "val", "realesrgan", device, batch_size)
        row["adabn_s"] = time.perf_counter() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sr", type=int, default=100, help="images for the SR timing")
    ap.add_argument("--fold", type=int, default=0, help="fold whose val set is used for adaptation")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    CFG.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rows = []

    paths = sample_images(args.n_sr)
    for name, module in [("Real-ESRGAN", run_realesrgan), ("SwinIR-L", run_swinir)]:
        print(f"[cost] SR timing: {name} on {len(paths)} images ...")
        r = time_sr(module, paths)
        rows.append({"item": f"SR {name}", **r})
        print(f"  per image {r['per_image_ms']:.0f} ms, start-up {r['startup_s']:.1f} s")

    for m in MODEL_NAMES:
        model, _ = load_checkpoint(f"{m}_protoB_fold{args.fold}")
        b1 = time_classifier(model, device, 1)
        b64 = time_classifier(model, device, args.batch_size)
        rows.append({"item": f"classifier {m}", "per_image_ms": b1, "per_image_ms_batched": b64})
        print(f"[cost] classifier {m}: {b1:.2f} ms/img (bs 1), {b64:.3f} ms/img (bs {args.batch_size})")
        del model

    for m in MODEL_NAMES:
        r = time_adaptation(m, args.fold, device, args.batch_size)
        rows.append({"item": f"adaptation {m}", **r})
        print(f"[cost] adaptation {m}: threshold {r['threshold_recal_s']:.1f} s"
              + (f", AdaBN {r['adabn_s']:.1f} s" if "adabn_s" in r else "")
              + f" on {r['n_val_images']} unlabeled val images")

    df = pd.DataFrame(rows)
    df["gpu"] = gpu
    out = CFG.results_root / "table_cost.csv"
    df.to_csv(out, index=False)
    print(f"\n[cost] GPU: {gpu}\n[cost] wrote {out}")
    print(df.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
