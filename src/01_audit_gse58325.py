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

    interim_root = Path(cfg.get("interim_root", "data/interim/gse58325"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse58325"))

    cfg.setdefault("dataset_name", "gse58325")
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
        raise ValueError("master_table.parquet is empty. Ingestion did not succeed.")

    expr = pd.to_numeric(master["expression_value"], errors="coerce")
    master = master.assign(expression_value=expr)

    missing_expression_values = int(master["expression_value"].isna().sum())

    wide = master.pivot_table(
        index="sample_accession",
        columns="feature_id",
        values="expression_value",
        aggfunc="mean",
    )

    if wide.shape[0] == 0 or wide.shape[1] == 0:
        raise ValueError(f"Pivoted audit matrix is empty: shape={wide.shape}")

    # FPKM can be skewed. Use log1p then median fill.
    wide_log = np.log1p(wide)
    wide_filled = wide_log.apply(lambda col: col.fillna(col.median()), axis=0)

    n_components = min(3, wide_filled.shape[0], wide_filled.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(wide_filled.values)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_scores = pd.DataFrame(pcs, columns=pca_cols)
    pca_scores.insert(0, "sample_accession", wide_filled.index)
    pca_scores = pca_scores.merge(sample_meta, on="sample_accession", how="left")

    condition_counts = (
        sample_meta.groupby(["conditioning_class", "stage_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["conditioning_class", "stage_label"])
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
        .merge(sample_meta, on="sample_accession", how="left")
    )

    top_feature_variances = (
        wide_log.var(axis=0, skipna=True)
        .sort_values(ascending=False)
        .head(50)
        .rename("variance_log1p_fpkm")
        .reset_index()
    )

    out_condition_counts = interim_root / "audit_condition_counts.csv"
    out_expression_summary = interim_root / "audit_expression_summary.csv"
    out_pca_scores = interim_root / "audit_pca_scores.csv"
    out_feature_var = interim_root / "audit_top_feature_variances.csv"
    out_summary = logs_root / "audit_summary.json"

    condition_counts.to_csv(out_condition_counts, index=False)
    expression_summary.to_csv(out_expression_summary, index=False)
    pca_scores.to_csv(out_pca_scores, index=False)
    top_feature_variances.to_csv(out_feature_var, index=False)

    explained = {
        f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)
    }

    summary = {
        "dataset_name": dataset_name,
        "n_rows": int(len(master)),
        "n_features": int(master["feature_id"].nunique()),
        "n_samples": int(master["sample_accession"].nunique()),
        "missing_expression_values": missing_expression_values,
        "pca_explained_variance_ratio": explained,
        "conditioning_classes": sorted(sample_meta["conditioning_class"].dropna().astype(str).unique().tolist()),
        "stage_labels": sorted(sample_meta["stage_label"].dropna().astype(str).unique().tolist()),
        "expression_unit": "FPKM",
        "note": "Audit uses log1p(FPKM) with per-feature median fill for PCA.",
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_condition_counts}")
    print(f"[ok] wrote: {out_expression_summary}")
    print(f"[ok] wrote: {out_pca_scores}")
    print(f"[ok] wrote: {out_feature_var}")
    print(f"[ok] wrote: {out_summary}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
