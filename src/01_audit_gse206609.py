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
    interim_dir = cfg["paths"]["interim_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(interim_dir)
    ensure_dir(logs_dir)

    master = pd.read_parquet(f"{interim_dir}/master_table.parquet")
    sample_meta = pd.read_csv(f"{interim_dir}/sample_metadata.csv")

    cond_counts = (
        sample_meta.groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(n_samples=("sample_accession", "nunique"))
        .sort_values(["phase", "treatment"])
    )

    expr_summary = (
        master.groupby("sample_accession", as_index=False)
        .agg(
            n_rows=("feature_id", "size"),
            n_missing=("raw_count", lambda s: int(s.isna().sum())),
            expr_min=("raw_count", "min"),
            expr_median=("raw_count", "median"),
            expr_mean=("raw_count", "mean"),
            expr_std=("raw_count", "std"),
            expr_max=("raw_count", "max"),
        )
        .merge(sample_meta, on="sample_accession", how="left")
    )

    wide = master.pivot(index="feature_id", columns="sample_accession", values="raw_count").fillna(0.0)
    wide_log = np.log1p(wide)

    pca = PCA(n_components=min(5, wide_log.shape[1]))
    pcs = pca.fit_transform(wide_log.T)

    pca_scores = pd.DataFrame(
        pcs[:, :3],
        columns=["PC1", "PC2", "PC3"],
    )
    pca_scores["sample_accession"] = wide_log.columns.values
    pca_scores = pca_scores.merge(sample_meta, on="sample_accession", how="left")

    centroids = (
        pca_scores.groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
        .sort_values(["phase", "treatment"])
    )

    replicate_tightness = (
        pca_scores.merge(centroids, on=["condition", "phase", "treatment"], how="left")
        .assign(
            dist_to_condition_centroid=lambda d: np.sqrt(
                (d["PC1"] - d["PC1_centroid"]) ** 2
                + (d["PC2"] - d["PC2_centroid"]) ** 2
                + (d["PC3"] - d["PC3_centroid"]) ** 2
            )
        )
        .groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(
            mean_replicate_dist=("dist_to_condition_centroid", "mean"),
            max_replicate_dist=("dist_to_condition_centroid", "max"),
        )
        .sort_values(["phase", "treatment"])
    )

    top_feature_vars = (
        wide_log.var(axis=1)
        .sort_values(ascending=False)
        .head(50)
        .reset_index()
    )
    top_feature_vars.columns = ["feature_id", "log1p_variance"]

    cond_counts.to_csv(f"{interim_dir}/audit_condition_counts.csv", index=False)
    expr_summary.to_csv(f"{interim_dir}/audit_expression_summary.csv", index=False)
    pca_scores.to_csv(f"{interim_dir}/audit_pca_scores.csv", index=False)
    centroids.to_csv(f"{interim_dir}/audit_condition_centroids.csv", index=False)
    replicate_tightness.to_csv(f"{interim_dir}/audit_replicate_tightness.csv", index=False)
    top_feature_vars.to_csv(f"{interim_dir}/audit_top_feature_variances.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_rows": int(len(master)),
        "n_features": int(master["feature_id"].nunique()),
        "n_samples": int(master["sample_accession"].nunique()),
        "missing_raw_count_values": int(master["raw_count"].isna().sum()),
        "pca_explained_variance_ratio": {
            f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_[:3])
        },
        "conditions": cond_counts["condition"].tolist(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
