"""
Figure 1: same-checkpoint accuracy by native short-side band, pooled over the
6 backbones x 5 Protocol B folds (image-weighted), for the four inputs.

Reads  ../../result-ear-sr-gender/table_native_size.csv
Writes native_size.pdf (vector, for the paper) and native_size.png (preview)

Usage (from anywhere):
    python paper/figures/make_fig_native_size.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[1] / "result-ear-sr-gender" / "table_native_size.csv"

BANDS = ["<32", "32-63", "64-127", ">=128"]
BAND_LABELS = ["<32", "32–63", "64–127", "≥128"]
SERIES = [  # variant, label, color, marker, linestyle, x-dodge
    # Original/Bicubic and Real-ESRGAN/SwinIR nearly coincide, so each pair is
    # dodged by +-0.05 on the categorical x-axis to keep every marker visible.
    ("orig", "Original", "#2a78d6", "o", "-", -0.05),
    ("bicubic", "Bicubic", "#eb6834", "s", "--", 0.05),
    ("realesrgan", "Real-ESRGAN", "#1baf7a", "^", "-", -0.05),
    ("swinir", "SwinIR", "#eda100", "D", "--", 0.05),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def pooled_accuracy() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["tag"].str.contains("protoB")].copy()
    df["correct"] = df["accuracy"] * df["n"]
    g = df.groupby(["bucket", "variant"])[["correct", "n"]].sum()
    return (100 * g["correct"] / g["n"]).unstack("variant").loc[BANDS]


def main():
    acc = pooled_accuracy()
    print(acc[[s[0] for s in SERIES]].round(2).to_string())

    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    x = range(len(BANDS))
    for variant, label, color, marker, ls, dx in SERIES:
        ax.plot([i + dx for i in x], acc[variant].values, color=color, marker=marker, linestyle=ls,
                linewidth=1.4, markersize=4, markeredgecolor="white",
                markeredgewidth=0.6, label=label, zorder=3)

    ax.set_xticks(list(x), BAND_LABELS)
    ax.set_xlabel("Native short side (px)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(82, 95)
    ax.set_xlim(-0.25, len(BANDS) - 0.75)
    ax.grid(axis="y", color="#d9d9d6", linewidth=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#52514e")
    ax.tick_params(width=0.6, length=2.5, colors="#0b0b0b")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, handlelength=2.0, columnspacing=0.9,
              handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout(pad=0.2)
    fig.savefig(HERE / "native_size.pdf")
    fig.savefig(HERE / "native_size.png", dpi=300)
    print("->", HERE / "native_size.pdf")


if __name__ == "__main__":
    main()
