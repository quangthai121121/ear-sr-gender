"""
Figure 2: same-checkpoint accuracy by native short-side of the crop
(before any SR/bicubic upsampling), bucketed as <32, 32-63, 64-127, >=128.

Reads results/predictions/<tag>__<variant>.csv (which already carries
short_side per image from same_checkpoint_eval.py) and produces:

  results/table_native_size.csv
  results/figure2_native_size.png

Usage:
    python -m src.eval.native_size --tags resnet50_protoB_fold0 mobilenet_v2_protoB_fold0
    python -m src.eval.native_size --all
"""
from __future__ import annotations
import argparse
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CFG

BUCKETS = [(0, 32), (32, 64), (64, 128), (128, float("inf"))]
BUCKET_LABELS = ["<32", "32-63", "64-127", ">=128"]
VARIANTS = ["orig", "bicubic", "realesrgan", "swinir"]


def bucket_of(short_side: int) -> str:
    for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS):
        if lo <= short_side < hi:
            return label
    return BUCKET_LABELS[-1]


def discover_tags():
    pattern = re.compile(r"^(?P<tag>.+)__orig\.csv$")
    return sorted(
        m.group("tag") for p in CFG.predictions_dir.glob("*__orig.csv")
        if (m := pattern.match(p.name))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    CFG.ensure_dirs()
    tags = args.tags or (discover_tags() if args.all else [])
    if not tags:
        print("No tags given/found. Pass --tags ... or --all.")
        return

    rows = []
    for tag in tags:
        for variant in VARIANTS:
            path = CFG.predictions_dir / f"{tag}__{variant}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["bucket"] = df["short_side"].apply(bucket_of)
            grouped = df.groupby("bucket")["correct"].agg(["mean", "count"])
            for bucket_label in BUCKET_LABELS:
                if bucket_label in grouped.index:
                    rows.append({
                        "tag": tag, "variant": variant, "bucket": bucket_label,
                        "accuracy": grouped.loc[bucket_label, "mean"],
                        "n": int(grouped.loc[bucket_label, "count"]),
                    })

    if not rows:
        print("No matching prediction files found.")
        return

    out_df = pd.DataFrame(rows)
    out_csv = CFG.results_root / "table_native_size.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[native_size] wrote {out_csv}")

    # one figure per tag, variants as separate lines
    for tag in out_df["tag"].unique():
        sub = out_df[out_df["tag"] == tag]
        fig, ax = plt.subplots(figsize=(6, 4))
        for variant in VARIANTS:
            vsub = sub[sub["variant"] == variant].set_index("bucket").reindex(BUCKET_LABELS)
            if vsub["accuracy"].isna().all():
                continue
            ax.plot(BUCKET_LABELS, vsub["accuracy"] * 100, marker="o", label=variant)
        ax.set_xlabel("Native short side (px)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Same-checkpoint accuracy by native size — {tag}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig_path = CFG.results_root / f"figure2_native_size__{tag}.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[native_size] wrote {fig_path}")


if __name__ == "__main__":
    main()
