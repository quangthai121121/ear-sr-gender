"""
Turns the raw per-fold CSVs into the paper's final tables.

Table 3 (same-checkpoint benchmark, mean+/-std over Protocol B folds):
    reads results/table3_same_checkpoint_raw.csv
    writes results/table3_same_checkpoint.csv  (+ printed as markdown)

Table 5 (flips, Holm-corrected across all model x SR tests):
    reads results/table5_flips_raw.csv
    writes results/table5_flips.csv

Table 4 (legacy Protocol A vs subject-disjoint Protocol B) needs both a
Protocol-A run and Protocol-B fold runs for the same model/variant; this
script pulls whatever matching rows exist from table3_same_checkpoint_raw.csv.

Matched-domain retraining (secondary analysis, src.train.train --retrain):
    reads results/table_matched_domain_raw.csv (only exists if you've run
    --retrain checkpoints; safe to skip otherwise)
    writes results/table_matched_domain.csv
    Kept in entirely separate input/output files from Table 3 on purpose --
    this NEVER reads from or writes into table3_same_checkpoint(.raw).csv,
    so re-running matched-domain experiments can't silently change the
    main same-checkpoint results.

Usage:
    python -m src.eval.aggregate
"""
from __future__ import annotations
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.config import CFG


def aggregate_table3():
    raw_path = CFG.results_root / "table3_same_checkpoint_raw.csv"
    if not raw_path.exists():
        print(f"[table3] {raw_path} not found, skipping.")
        return
    df = pd.read_csv(raw_path)
    proto_b = df[df["protocol"] == "b"]
    if proto_b.empty:
        print("[table3] no protocol='b' rows found, skipping.")
        return

    agg = (proto_b.groupby(["model", "variant"])
           .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                bal_acc_mean=("balanced_accuracy", "mean"), bal_acc_std=("balanced_accuracy", "std"),
                n_folds=("fold", "nunique"))
           .reset_index())

    agg["acc_str"] = agg.apply(lambda r: f"{100*r.acc_mean:.2f}±{100*(r.acc_std or 0):.2f}", axis=1)
    agg["f1_str"] = agg.apply(lambda r: f"{100*r.f1_mean:.2f}±{100*(r.f1_std or 0):.2f}", axis=1)

    out_path = CFG.results_root / "table3_same_checkpoint.csv"
    agg.to_csv(out_path, index=False)
    print(f"[table3] wrote {out_path}\n")

    pivot = agg.pivot(index="model", columns="variant", values="acc_str")
    order = [c for c in ["orig", "bicubic", "realesrgan", "swinir"] if c in pivot.columns]
    print("Table 3 (Acc %, mean±std over folds):")
    print(pivot[order].to_markdown())
    print()


def aggregate_table4():
    raw_path = CFG.results_root / "table3_same_checkpoint_raw.csv"
    if not raw_path.exists():
        return
    df = pd.read_csv(raw_path)
    a = df[df["protocol"] == "a"]
    b = df[df["protocol"] == "b"]
    if a.empty or b.empty:
        print("[table4] need both protocol='a' and protocol='b' rows, skipping.")
        return

    b_agg = (b.groupby(["model", "variant"])
             .agg(acc_b_mean=("accuracy", "mean"), acc_b_std=("accuracy", "std"),
                  f1_b_mean=("macro_f1", "mean"), f1_b_std=("macro_f1", "std"))
             .reset_index())
    a_agg = a.groupby(["model", "variant"]).agg(acc_a=("accuracy", "mean")).reset_index()

    merged = a_agg.merge(b_agg, on=["model", "variant"], how="inner")
    merged["acc_a_pct"] = 100 * merged["acc_a"]
    merged["acc_b_str"] = merged.apply(
        lambda r: f"{100*r.acc_b_mean:.2f}±{100*(r.acc_b_std or 0):.2f}", axis=1)
    merged["f1_b_str"] = merged.apply(
        lambda r: f"{100*r.f1_b_mean:.2f}±{100*(r.f1_b_std or 0):.2f}", axis=1)
    merged["a_to_b_gap_pp"] = merged["acc_a_pct"] - 100 * merged["acc_b_mean"]

    out_path = CFG.results_root / "table4_protocolA_vs_B.csv"
    merged[["model", "variant", "acc_a_pct", "acc_b_str", "f1_b_str", "a_to_b_gap_pp"]].to_csv(
        out_path, index=False)
    print(f"[table4] wrote {out_path}\n")
    print("Table 4 (legacy A vs subject-disjoint B):")
    print(merged[["model", "variant", "acc_a_pct", "acc_b_str", "a_to_b_gap_pp"]].to_markdown(index=False))
    print()


def aggregate_table5():
    raw_path = CFG.results_root / "table5_flips_raw.csv"
    if not raw_path.exists():
        print(f"[table5] {raw_path} not found, skipping.")
        return
    df = pd.read_csv(raw_path)

    # Holm correction across ALL model x SR tests reported together.
    reject, p_adj, _, _ = multipletests(df["mcnemar_p"], method="holm")
    df["mcnemar_p_holm"] = p_adj
    df["significant_holm_0.05"] = reject

    out_path = CFG.results_root / "table5_flips.csv"
    df.to_csv(out_path, index=False)
    print(f"[table5] wrote {out_path}\n")
    print("Table 5 (corrected/corrupted, Holm-corrected p):")
    cols = ["tag", "sr_variant", "n", "corrected", "corrupted", "net_gain_pp",
            "mcnemar_p", "mcnemar_p_holm", "significant_holm_0.05"]
    print(df[cols].to_markdown(index=False))


def aggregate_matched_domain():
    """Matched-domain retraining (secondary analysis, src.train.train --retrain):
    for each model, accuracy when trained AND tested on the same SR/bicubic
    variant, alongside the SAME model's "train Original, test Original"
    numbers (reused from Table 3, not retrained again -- see below) so all
    four conditions sit in one table:

        orig:       train orig      / test orig       (from Table 3)
        bicubic:    train bicubic   / test bicubic     (retrained)
        realesrgan: train realesrgan/ test realesrgan  (retrained)
        swinir:     train swinir    / test swinir      (retrained)

    "train orig, test orig" is NOT retrained separately -- it's already
    exactly Table 3's "orig" column (same locked_config, same protocol),
    so reusing it avoids adding pure random-init noise as if it were new
    information. Only the train_variant == test_variant "diagonal" of the
    matched-domain runs is included (the core question); rows where a
    checkpoint was explicitly evaluated cross-domain (test_variant !=
    train_variant) are left out of this summary but remain in the raw CSV.
    """
    rows = []

    # "orig" condition: reused from the same-checkpoint Table 3 raw data.
    sc_path = CFG.results_root / "table3_same_checkpoint_raw.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        sc_orig = sc[(sc["protocol"] == "b") & (sc["variant"] == "orig")]
        if not sc_orig.empty:
            g = (sc_orig.groupby("model")
                 .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                      f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                      n_folds=("fold", "nunique"))
                 .reset_index())
            g["variant"] = "orig"
            g["source"] = "same_checkpoint (train=orig, test=orig)"
            rows.append(g)

    # "bicubic"/"realesrgan"/"swinir": retrained matched-domain checkpoints.
    raw_path = CFG.results_root / "table_matched_domain_raw.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        df = df[df["protocol"] == "b"]
        diag = df[df["train_variant"] == df["test_variant"]]
        if not diag.empty:
            g = (diag.groupby(["model", "train_variant"])
                 .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                      f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                      n_folds=("fold", "nunique"))
                 .reset_index()
                 .rename(columns={"train_variant": "variant"}))
            g["source"] = "retrained (train==test==variant)"
            rows.append(g)

    if not rows:
        print("[matched_domain] no Table 3 orig data and no "
              "table_matched_domain_raw.csv found, skipping.")
        return

    agg = pd.concat(rows, ignore_index=True)
    agg["acc_str"] = agg.apply(lambda r: f"{100*r.acc_mean:.2f}±{100*(r.acc_std or 0):.2f}", axis=1)
    agg["f1_str"] = agg.apply(lambda r: f"{100*r.f1_mean:.2f}±{100*(r.f1_std or 0):.2f}", axis=1)

    variant_order = ["orig", "bicubic", "realesrgan", "swinir"]
    agg["variant"] = pd.Categorical(agg["variant"], categories=variant_order, ordered=True)
    agg = agg.sort_values(["model", "variant"])

    # Gap vs. this model's own "orig" row (pp), for interpretation --
    # ~0 suggests representation drift (SR domain is learnable once
    # retrained); a persistent negative gap suggests SR destroys signal
    # that retraining can't recover.
    orig_acc = agg[agg["variant"] == "orig"].set_index("model")["acc_mean"] * 100
    agg["gap_vs_orig_pp"] = agg.apply(
        lambda r: 100 * r.acc_mean - orig_acc.get(r.model, float("nan")), axis=1).round(2)

    out_path = CFG.results_root / "table_matched_domain.csv"
    agg.to_csv(out_path, index=False)
    print(f"[matched_domain] wrote {out_path}\n")

    print("Matched-domain comparison (Acc %, mean±std over folds; each "
          "condition trained AND tested on its own variant):")
    print(agg.pivot(index="model", columns="variant", values="acc_str")
          .reindex(columns=[v for v in variant_order if v in agg["variant"].unique()])
          .to_markdown())
    print()
    print("Same, Macro-F1 (%):")
    print(agg.pivot(index="model", columns="variant", values="f1_str")
          .reindex(columns=[v for v in variant_order if v in agg["variant"].unique()])
          .to_markdown())
    print()
    print("Gap vs. this model's own 'orig' row (pp):")
    print(agg.pivot(index="model", columns="variant", values="gap_vs_orig_pp")
          .reindex(columns=[v for v in variant_order if v in agg["variant"].unique()])
          .to_markdown())
    print()
    missing = [v for v in variant_order if v not in agg["variant"].unique()]
    if missing:
        print(f"[matched_domain] note: no data yet for variant(s) {missing} -- "
              f"run scripts/11+12_*.sh (or src.train.train --retrain --variant ...) "
              f"to fill them in.")
    print()


def main():
    aggregate_table3()
    aggregate_table4()
    aggregate_table5()
    aggregate_matched_domain()


if __name__ == "__main__":
    main()
