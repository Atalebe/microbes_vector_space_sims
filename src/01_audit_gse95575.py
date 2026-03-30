#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    interim_root = Path(cfg.get("interim_root", "data/interim/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))

    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("interim_root", str(interim_root))
    cfg.setdefault("logs_root", str(logs_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    interim_root = Path(cfg["interim_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(interim_root, logs_root)

    master = pd.read_parquet(interim_root / "master_table.parquet")
    sample_meta = pd.read_csv(interim_root / "sample_metadata.csv")

    if master.empty:
        raise ValueError("master_table.parquet is empty.")

    master["expression_value"] = pd.to_numeric(master["expression_value"], errors="coerce")
    master["feature_id"] = master["feature_id"].astype(str)

    if master["sample_accession"].nunique() != 30:
        raise ValueError(
            f"Expected 30 samples in GSE95575 master table, found {master['sample_accession'].nunique()}."
        )

    if master["feature_id"].nunique() < 1000:
        raise ValueError(
            f"Expected at least 1000 features in GSE95575, found {master['feature_id'].nunique()}."
        )

    missing_raw_count_values = int(master["expression_value"].isna().sum())

    wide = master.pivot_table(
        index="sample_accession",
        columns="feature_id",
        values="expression_value",
        aggfunc="mean",
    )

    wide_log = np.log1p(wide)
    wide_filled = wide_log.apply(lambda col: col.fillna(col.median()), axis=0)

    pca = PCA(n_components=min(3, wide_filled.shape[0], wide_filled.shape[1]))
    pcs = pca.fit_transform(wide_filled.values)

    pca_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pca_scores = pd.DataFrame(pcs, columns=pca_cols)
    pca_scores.insert(0, "sample_accession", wide_filled.index)
    pca_scores = pca_scores.merge(
        sample_meta[
            [
                "sample_accession",
                "raw_sample_label",
                "geo_accession",
                "treatment",
                "time_min",
                "condition",
                "replicate",
            ]
        ],
        on="sample_accession",
        how="left",
    )

    condition_counts = (
        pca_scores.groupby(["treatment", "time_min", "condition"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["treatment", "time_min", "condition"])
    )

    expression_summary = (
        master.groupby("sample_accession")["expression_value"]
        .agg(
            n_rows="size",
            n_missing=lambda x: x.isna().sum(),
            expr_min="min",
            expr_median="median",
            expr_mean="mean",
            expr_std="std",
            expr_max="max",
        )
        .reset_index()
        .merge(
            sample_meta[
                [
                    "sample_accession",
                    "raw_sample_label",
                    "geo_accession",
                    "treatment",
                    "time_min",
                    "condition",
                    "replicate",
                ]
            ],
            on="sample_accession",
            how="left",
        )
    )

    condition_centroids = (
        pca_scores.groupby(["treatment", "time_min", "condition"], dropna=False)[pca_cols]
        .mean()
        .reset_index()
        .sort_values(["treatment", "time_min"])
    )

    replicate_tightness = (
        pca_scores.groupby("condition")[pca_cols]
        .std()
        .reset_index()
        .rename(columns={c: f"{c}_std" for c in pca_cols})
    )

    top_feature_variances = (
        wide_log.var(axis=0, skipna=True)
        .sort_values(ascending=False)
        .head(50)
        .rename("variance_log1p_counts")
        .reset_index()
    )

    out_counts = interim_root / "audit_condition_counts.csv"
    out_expr = interim_root / "audit_expression_summary.csv"
    out_pca = interim_root / "audit_pca_scores.csv"
    out_centroids = interim_root / "audit_condition_centroids.csv"
    out_reps = interim_root / "audit_replicate_tightness.csv"
    out_feat = interim_root / "audit_top_feature_variances.csv"
    out_summary = logs_root / "audit_summary.json"

    condition_counts.to_csv(out_counts, index=False)
    expression_summary.to_csv(out_expr, index=False)
    pca_scores.to_csv(out_pca, index=False)
    condition_centroids.to_csv(out_centroids, index=False)
    replicate_tightness.to_csv(out_reps, index=False)
    top_feature_variances.to_csv(out_feat, index=False)

    summary = {
        "dataset_name": dataset_name,
        "n_rows": int(len(master)),
        "n_features": int(master["feature_id"].nunique()),
        "n_samples": int(master["sample_accession"].nunique()),
        "missing_raw_count_values": missing_raw_count_values,
        "pca_explained_variance_ratio": {
            f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)
        },
        "conditions": sorted(master["condition"].dropna().astype(str).unique().tolist()),
        "treatments": sorted(master["treatment"].dropna().astype(str).unique().tolist()),
        "timepoints_min": sorted(master["time_min"].dropna().astype(float).unique().tolist()),
        "note": "Audit uses log1p(raw counts) with per-feature median fill for PCA.",
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_counts}")
    print(f"[ok] wrote: {out_expr}")
    print(f"[ok] wrote: {out_pca}")
    print(f"[ok] wrote: {out_centroids}")
    print(f"[ok] wrote: {out_reps}")
    print(f"[ok] wrote: {out_feat}")
    print(f"[ok] wrote: {out_summary}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
