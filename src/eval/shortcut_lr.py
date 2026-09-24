"""
Geometry-shortcut check (CPU only). Can gender be predicted from crop
geometry alone -- the cue PadResize224's letterbox padding exposes to every
recognizer? Fits a logistic regression on geometry features under the SAME
Protocol B folds (train on each fold's train split, test on its test split)
and compares against the majority-class baseline.

Feature sets:
  padding     aspect ratio (w/h) and letterbox padding fraction (1 - short/long)
  padding+size  the above plus log native short and long side

Note the padding is identical across Original/Bicubic/SR inputs, so this
cue can affect absolute accuracy but cannot explain Delta_SR.

Usage:
    python -m src.eval.shortcut_lr
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG

FEATURE_SETS = {
    "padding": ["aspect", "pad_frac"],
    "padding+size": ["aspect", "pad_frac", "log_short", "log_long"],
}


def geometry_features(images: pd.DataFrame) -> pd.DataFrame:
    df = images.copy()
    long_side = df[["width", "height"]].max(axis=1)
    df["aspect"] = df["width"] / df["height"]
    df["pad_frac"] = 1.0 - df["short_side"] / long_side
    df["log_short"] = np.log(df["short_side"])
    df["log_long"] = np.log(long_side)
    return df


def main():
    images = geometry_features(pd.read_csv(CFG.images_csv))
    rows = []
    for fold in range(CFG.protocol_b.n_folds):
        split = pd.read_csv(CFG.protocol_b_csv(fold))[["image_id", "split"]]
        df = split.merge(images, on="image_id")
        train, test = df[df.split == "train"], df[df.split == "test"]
        y_tr, y_te = (train.gender == "female").astype(int), (test.gender == "female").astype(int)
        rows.append({"fold": fold, "features": "majority",
                     "accuracy": accuracy_score(y_te, np.zeros_like(y_te)),
                     "balanced_accuracy": 0.5})
        for name, cols in FEATURE_SETS.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
            clf.fit(train[cols], y_tr)
            pred = clf.predict(test[cols])
            rows.append({"fold": fold, "features": name,
                         "accuracy": accuracy_score(y_te, pred),
                         "balanced_accuracy": balanced_accuracy_score(y_te, pred)})

    res = pd.DataFrame(rows)
    out = CFG.results_root / "table_shortcut_lr.csv"
    res.to_csv(out, index=False)
    summary = res.groupby("features")[["accuracy", "balanced_accuracy"]].agg(["mean", "std"]) * 100
    print(summary.round(2).to_markdown())
    print(f"\n[shortcut_lr] wrote {out}")


if __name__ == "__main__":
    main()
