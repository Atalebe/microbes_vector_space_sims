from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    master_path = cfg["output"]["master_table"]
    sample_meta_path = cfg["output"]["sample_metadata"]
    interim_dir = cfg["paths"]["interim_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(interim_dir)
    ensure_dir(logs_dir)

    master = pd.read_parquet(master_path)
    sample_meta = pd.read_csv(sample_meta_path)

    # ----------------------------
    # 1. Basic audit summaries
    # ----------------------------
    rows = len(master)
    n_features = master["feature_id"].nunique()
    n_samples = master["sample_accession"].nunique()
    missing_expr = int(master["expression_value"].isna().sum())

    by_condition = (
        sample_meta.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            n_samples=("sample_accession", "nunique"),
            replicates=("replicate", lambda x: sorted(x.dropna().tolist())),
        )
        .sort_values("condition_order")
    )

    expr_summary = (
        master.groupby("sample_accession", as_index=False)
        .agg(
            expr_min=("expression_value", "min"),
            expr_median=("expression_value", "median"),
            expr_mean=("expression_value", "mean"),
            expr_std=("expression_value", "std"),
            expr_max=("expression_value", "max"),
            n_missing=("expression_value", lambda x: int(x.isna().sum())),
        )
        .merge(
            sample_meta[
                ["sample_accession", "sample_title", "condition", "condition_order", "replicate"]
            ],
            on="sample_accession",
            how="left",
        )
        .sort_values(["condition_order", "replicate"])
    )

    # ----------------------------
    # 2. Wide matrix for geometry
    # ----------------------------
    wide = master.pivot_table(
        index="feature_id",
        columns="sample_accession",
        values="expression_value",
        aggfunc="mean",
    )

    # align columns to metadata order
    ordered_samples = (
        sample_meta.sort_values(["condition_order", "replicate"])["sample_accession"].tolist()
    )
    wide = wide[ordered_samples]

    # feature variance ranking
    feature_var = wide.var(axis=1).sort_values(ascending=False)
    top_var_features = feature_var.head(1000).index.tolist()

    wide_top = wide.loc[top_var_features].T  # samples x features

    # ----------------------------
    # 3. PCA
    # ----------------------------
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(wide_top.fillna(wide_top.mean()))

    pca_df = pd.DataFrame(
        {
            "sample_accession": ordered_samples,
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "PC3": pcs[:, 2],
        }
    ).merge(
        sample_meta[
            ["sample_accession", "sample_title", "condition", "condition_order", "replicate"]
        ],
        on="sample_accession",
        how="left",
    ).sort_values(["condition_order", "replicate"])

    explained = pca.explained_variance_ratio_

    # ----------------------------
    # 4. Condition centroids in PCA space
    # ----------------------------
    centroid_df = (
        pca_df.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
        .sort_values("condition_order")
    )

    # step lengths between ordered condition centroids
    centroid_df["next_condition"] = centroid_df["condition"].shift(-1)
    centroid_df["step_length_PC12"] = np.sqrt(
        (centroid_df["PC1_centroid"].shift(-1) - centroid_df["PC1_centroid"]) ** 2
        + (centroid_df["PC2_centroid"].shift(-1) - centroid_df["PC2_centroid"]) ** 2
    )
    centroid_df["step_length_PC123"] = np.sqrt(
        (centroid_df["PC1_centroid"].shift(-1) - centroid_df["PC1_centroid"]) ** 2
        + (centroid_df["PC2_centroid"].shift(-1) - centroid_df["PC2_centroid"]) ** 2
        + (centroid_df["PC3_centroid"].shift(-1) - centroid_df["PC3_centroid"]) ** 2
    )

    # ----------------------------
    # 5. Replicate tightness by condition
    # ----------------------------
    rep_tightness = (
        pca_df.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1_std=("PC1", "std"),
            PC2_std=("PC2", "std"),
            PC3_std=("PC3", "std"),
        )
        .sort_values("condition_order")
    )

    # ----------------------------
    # 6. Save outputs
    # ----------------------------
    by_condition.to_csv(f"{interim_dir}/audit_condition_counts.csv", index=False)
    expr_summary.to_csv(f"{interim_dir}/audit_expression_summary.csv", index=False)
    pca_df.to_csv(f"{interim_dir}/audit_pca_scores.csv", index=False)
    centroid_df.to_csv(f"{interim_dir}/audit_condition_centroids.csv", index=False)
    rep_tightness.to_csv(f"{interim_dir}/audit_replicate_tightness.csv", index=False)
    feature_var.head(200).rename("variance").to_csv(f"{interim_dir}/audit_top_feature_variances.csv")

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_rows": int(rows),
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "missing_expression_values": int(missing_expr),
        "pca_explained_variance_ratio": {
            "PC1": float(explained[0]),
            "PC2": float(explained[1]),
            "PC3": float(explained[2]),
        },
        "ordered_conditions": by_condition["condition"].tolist(),
    }

    with open(f"{logs_dir}/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{interim_dir}/audit_condition_counts.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_expression_summary.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_pca_scores.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_condition_centroids.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_replicate_tightness.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_top_feature_variances.csv")
    print("[ok] wrote:", f"{logs_dir}/audit_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit GSE36599 and compute first-pass PCA geometry.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
