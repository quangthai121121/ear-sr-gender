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


def main():
    aggregate_table3()
    aggregate_table4()
    aggregate_table5()


if __name__ == "__main__":
    main()
