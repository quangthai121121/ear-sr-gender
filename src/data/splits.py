"""
Build the two partitioning protocols described in the paper draft:

  Protocol A (legacy, image-level): random 50/50 split, stratified by
  gender only. Subjects CAN appear in both train and test (this is the
  known leakage risk the paper flags -- kept only for comparison with [1]).

  Protocol B (subject-disjoint, main setting): 5-fold StratifiedGroupKFold
  with groups=subject_id, stratified by gender, so every subject appears
  in exactly one fold's test set and never leaks into that fold's train/val.
  Within each fold's training subjects we carve out a validation subset
  (subject-level, stratified by gender) for early stopping / HP selection.

Usage:
    python -m src.data.splits --protocol a
    python -m src.data.splits --protocol b
    python -m src.data.splits --protocol both
"""
from __future__ import annotations
import argparse
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import CFG


def load_images_df() -> pd.DataFrame:
    if not CFG.images_csv.exists():
        raise FileNotFoundError(
            f"{CFG.images_csv} not found. Run `python -m src.data.build_metadata` first."
        )
    return pd.read_csv(CFG.images_csv)


# ---------------------------------------------------------------------------
# Protocol A: legacy image-level 50/50 split, stratified by gender only.
# ---------------------------------------------------------------------------
def build_protocol_a(df: pd.DataFrame) -> pd.DataFrame:
    seed = CFG.protocol_a.seed
    test_frac = CFG.protocol_a.test_fraction

    train_ids, test_ids = train_test_split(
        df["image_id"],
        test_size=test_frac,
        random_state=seed,
        stratify=df["gender"],
    )
    train_ids, test_ids = set(train_ids), set(test_ids)

    out = df[["image_id", "subject_id", "gender"]].copy()
    out["split"] = out["image_id"].apply(lambda i: "train" if i in train_ids else "test")
    return out


# ---------------------------------------------------------------------------
# Protocol B: subject-disjoint, 5-fold stratified group CV.
# ---------------------------------------------------------------------------
def build_protocol_b(df: pd.DataFrame):
    n_folds = CFG.protocol_b.n_folds
    val_frac = CFG.protocol_b.val_fraction_of_train_subjects
    seed = CFG.protocol_b.seed

    subjects = df.drop_duplicates("subject_id")[["subject_id", "gender"]].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # sklearn wants X, y, groups all at the *image* level so group sizes are
    # respected: pass the images themselves, grouped by subject_id.
    X = df["image_id"].values
    y = df["gender"].values
    groups = df["subject_id"].values

    fold_frames = {}
    for fold_idx, (trainval_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        test_subjects = set(df.iloc[test_idx]["subject_id"])
        trainval_subjects_df = subjects[~subjects["subject_id"].isin(test_subjects)]

        train_subj, val_subj = train_test_split(
            trainval_subjects_df["subject_id"],
            test_size=val_frac,
            random_state=seed + fold_idx,
            stratify=trainval_subjects_df["gender"],
        )
        train_subj, val_subj = set(train_subj), set(val_subj)

        def assign(sid):
            if sid in test_subjects:
                return "test"
            elif sid in train_subj:
                return "train"
            elif sid in val_subj:
                return "val"
            else:
                raise RuntimeError(f"subject {sid} not assigned to any split")

        fold_df = df[["image_id", "subject_id", "gender"]].copy()
        fold_df["split"] = fold_df["subject_id"].apply(assign)
        fold_frames[fold_idx] = fold_df

    return fold_frames


def _print_split_summary(name: str, split_df: pd.DataFrame):
    print(f"\n{name}")
    summary = split_df.groupby(["split", "gender"]).size().unstack(fill_value=0)
    n_subj = split_df.groupby("split")["subject_id"].nunique()
    summary["n_subjects"] = n_subj
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    CFG.ensure_dirs()
    df = load_images_df()

    if args.protocol in ("a", "both"):
        a_df = build_protocol_a(df)
        a_df.to_csv(CFG.protocol_a_csv, index=False)
        _print_split_summary(f"Protocol A -> {CFG.protocol_a_csv}", a_df)

    if args.protocol in ("b", "both"):
        fold_frames = build_protocol_b(df)
        for fold_idx, fold_df in fold_frames.items():
            out_path = CFG.protocol_b_csv(fold_idx)
            fold_df.to_csv(out_path, index=False)
            _print_split_summary(f"Protocol B fold {fold_idx} -> {out_path}", fold_df)

        # Sanity check: every subject appears in exactly one fold's test set
        test_subject_sets = [
            set(fold_frames[f][fold_frames[f]["split"] == "test"]["subject_id"])
            for f in fold_frames
        ]
        all_test_subjects = set().union(*test_subject_sets)
        overlaps = sum(
            len(test_subject_sets[i] & test_subject_sets[j])
            for i in range(len(test_subject_sets))
            for j in range(i + 1, len(test_subject_sets))
        )
        print(f"\nProtocol B check: {len(all_test_subjects)} unique subjects covered "
              f"across {len(fold_frames)} test folds, {overlaps} pairwise overlaps "
              f"(should be 0).")


if __name__ == "__main__":
    main()
