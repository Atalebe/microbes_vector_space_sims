from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cpm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0)
    return counts.divide(lib, axis=1) * 1_000_000.0


def log2_cpm(counts: pd.DataFrame, prior_count: float = 1.0) -> pd.DataFrame:
    cpm = cpm_normalize(counts)
    return np.log2(cpm + prior_count)


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return (x - np.nanmean(x)) / sd
    return 0.67448975 * (x - med) / mad


def cohens_d_two_groups(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n1 + n2 - 2))
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(x2) - np.mean(x1)) / np.sqrt(pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/gse188790")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    base = Path(args.processed_dir)
    matrix_path = base / "gse188790_gene_count_matrix.csv"
    meta_path = base / "gse188790_sample_metadata.csv"
    state_path = base / "gse188790_state_table.csv"

    if not matrix_path.exists() or not meta_path.exists() or not state_path.exists():
        raise SystemExit("Missing prior outputs. Run ingest, audit, and state-table steps first.")

    matrix = pd.read_csv(matrix_path)
    meta = pd.read_csv(meta_path)
    state = pd.read_csv(state_path)

    sample_cols = [c for c in matrix.columns if c != "gene_id"]
    counts = matrix.set_index("gene_id")[sample_cols].copy()

    l2cpm = log2_cpm(counts, prior_count=1.0)

    aged_samples = meta.loc[meta["age_class"].eq("aged"), "sample_name"].tolist()
    unaged_samples = meta.loc[meta["age_class"].eq("unaged"), "sample_name"].tolist()

    if len(aged_samples) != 2 or len(unaged_samples) != 2:
        raise SystemExit("Expected exactly 2 aged and 2 unaged samples for GSE188790.")

    gene_stats = pd.DataFrame({"gene_id": l2cpm.index.astype(str)})
    gene_stats["mean_log2cpm_unaged"] = l2cpm[unaged_samples].mean(axis=1).values
    gene_stats["mean_log2cpm_aged"] = l2cpm[aged_samples].mean(axis=1).values
    gene_stats["delta_log2cpm_aged_minus_unaged"] = (
        gene_stats["mean_log2cpm_aged"] - gene_stats["mean_log2cpm_unaged"]
    )
    gene_stats["sd_log2cpm_unaged"] = l2cpm[unaged_samples].std(axis=1, ddof=1).values
    gene_stats["sd_log2cpm_aged"] = l2cpm[aged_samples].std(axis=1, ddof=1).values

    gene_stats["cohens_d_aged_minus_unaged"] = [
        cohens_d_two_groups(
            l2cpm.loc[g, unaged_samples].values,
            l2cpm.loc[g, aged_samples].values,
        )
        for g in l2cpm.index
    ]

    gene_stats["mean_count_unaged"] = counts[unaged_samples].mean(axis=1).values
    gene_stats["mean_count_aged"] = counts[aged_samples].mean(axis=1).values
    gene_stats["detected_in_unaged_n"] = (counts[unaged_samples] > 0).sum(axis=1).values
    gene_stats["detected_in_aged_n"] = (counts[aged_samples] > 0).sum(axis=1).values

    gene_stats["abs_delta_log2cpm"] = gene_stats["delta_log2cpm_aged_minus_unaged"].abs()
    gene_stats["delta_log2cpm_z"] = robust_z(gene_stats["delta_log2cpm_aged_minus_unaged"])

    gene_stats = gene_stats.sort_values(
        ["abs_delta_log2cpm", "mean_count_aged", "mean_count_unaged"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    gene_stats.to_csv(base / "gse188790_gene_age_contrast.csv", index=False)

    top_n = max(1, int(args.top_n))
    top_aged = (
        gene_stats.sort_values(
            ["delta_log2cpm_aged_minus_unaged", "mean_count_aged"],
            ascending=[False, False]
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    top_unaged = (
        gene_stats.sort_values(
            ["delta_log2cpm_aged_minus_unaged", "mean_count_unaged"],
            ascending=[True, False]
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    top_aged.to_csv(base / "gse188790_top_aged_up_genes.csv", index=False)
    top_unaged.to_csv(base / "gse188790_top_unaged_up_genes.csv", index=False)

    sample_age_summary = state[[
        "gsm_id", "sample_name", "age_class", "replicate",
        "library_size", "detected_genes", "pc1", "pc2", "H", "S", "M", "R"
    ]].copy()

    age_centroids = (
        sample_age_summary.groupby("age_class")[["pc1", "pc2", "H", "S", "M", "R"]]
        .mean()
        .reset_index()
    )
    age_centroids.to_csv(base / "gse188790_age_class_centroids.csv", index=False)

    age_summary = (
        sample_age_summary.groupby("age_class")
        .agg(
            n_samples=("sample_name", "size"),
            mean_library_size=("library_size", "mean"),
            mean_detected_genes=("detected_genes", "mean"),
            mean_pc1=("pc1", "mean"),
            mean_pc2=("pc2", "mean"),
            mean_H=("H", "mean"),
            mean_S=("S", "mean"),
            mean_M=("M", "mean"),
            mean_R=("R", "mean"),
        )
        .reset_index()
    )
    age_summary["delta_vs_unaged_mean_pc1"] = age_summary["mean_pc1"] - float(
        age_summary.loc[age_summary["age_class"].eq("unaged"), "mean_pc1"].iloc[0]
    )
    age_summary["delta_vs_unaged_mean_H"] = age_summary["mean_H"] - float(
        age_summary.loc[age_summary["age_class"].eq("unaged"), "mean_H"].iloc[0]
    )
    age_summary["delta_vs_unaged_mean_S"] = age_summary["mean_S"] - float(
        age_summary.loc[age_summary["age_class"].eq("unaged"), "mean_S"].iloc[0]
    )
    age_summary["delta_vs_unaged_mean_M"] = age_summary["mean_M"] - float(
        age_summary.loc[age_summary["age_class"].eq("unaged"), "mean_M"].iloc[0]
    )
    age_summary["delta_vs_unaged_mean_R"] = age_summary["mean_R"] - float(
        age_summary.loc[age_summary["age_class"].eq("unaged"), "mean_R"].iloc[0]
    )
    age_summary.to_csv(base / "gse188790_age_class_summary.csv", index=False)

    summary = {
        "dataset": "GSE188790",
        "n_genes": int(len(gene_stats)),
        "n_aged_samples": int(len(aged_samples)),
        "n_unaged_samples": int(len(unaged_samples)),
        "top_n_exported": top_n,
        "largest_positive_delta_log2cpm": float(top_aged["delta_log2cpm_aged_minus_unaged"].iloc[0]),
        "largest_negative_delta_log2cpm": float(top_unaged["delta_log2cpm_aged_minus_unaged"].iloc[0]),
        "notes": [
            "Gene-level contrast is descriptive and effect-size based.",
            "No formal differential-expression significance testing is claimed in this first pass.",
            "Aged-minus-unaged summaries are intended for vector-space interpretation and figure construction."
        ],
    }
    (base / "gse188790_residual_field_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
