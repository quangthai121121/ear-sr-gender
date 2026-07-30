"""
Figure 2: same-checkpoint accuracy by native short-side of the crop
(before any SR/bicubic upsampling), bucketed as <32, 32-63, 64-127, >=128.

Reads results/predictions/<tag>__<variant>.csv (which already carries
short_side per image from same_checkpoint_eval.py) and produces:

  results/table_native_size.csv
  results/figure2_native_size.png

NOTE on the >=128 bucket: src/data/dataset.py CAN skip the explicit
SR/bicubic x4 step (falling back to Original pixels) for images whose
native long side is already large, via
configs/paths.yaml -> dataset.skip_upsample_if_native_long_side_at_least
-- but this is DISABLED BY DEFAULT (null) for the main benchmark, precisely
so that this bucket reflects genuine measured SR effects rather than a
value forced to zero by construction (see that config's comments for why).
If you deliberately enable it for a separate robustness re-run, note that
its threshold is based on LONG side while this bucketing uses SHORT side,
so the two don't align exactly (e.g. a 130x180 crop has short_side>=128
but long_side<224, so SR would still run normally for it even with the
threshold enabled).

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
