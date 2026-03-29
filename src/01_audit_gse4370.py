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

    rows = len(master)
    n_features = master["feature_id"].nunique()
    n_samples = master["sample_accession"].nunique()
    missing_expr = int(master["expression_value"].isna().sum())

    time_summary = (
        sample_meta[["sample_accession", "sample_title", "condition", "time_min", "time_order"]]
        .sort_values("time_order")
        .reset_index(drop=True)
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
            sample_meta[["sample_accession", "sample_title", "condition", "time_min", "time_order"]],
            on="sample_accession",
            how="left",
        )
        .sort_values("time_order")
    )

    wide = master.pivot_table(
        index="feature_id",
        columns="sample_accession",
        values="expression_value",
        aggfunc="mean",
    )

    ordered_samples = sample_meta.sort_values("time_order")["sample_accession"].tolist()
    wide = wide[ordered_samples]

    feature_var = wide.var(axis=1).sort_values(ascending=False)
    top_var_features = feature_var.head(1000).index.tolist()
    wide_top = wide.loc[top_var_features].T

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
        sample_meta[["sample_accession", "sample_title", "condition", "time_min", "time_order"]],
        on="sample_accession",
        how="left",
    ).sort_values("time_order")

    # step lengths between timepoints
    step_records = []
    for i in range(len(pca_df) - 1):
        a = pca_df.iloc[i][["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        b = pca_df.iloc[i + 1][["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        step = b - a
        step_records.append(
            {
                "from_condition": pca_df.iloc[i]["condition"],
                "to_condition": pca_df.iloc[i + 1]["condition"],
                "from_time_min": float(pca_df.iloc[i]["time_min"]),
                "to_time_min": float(pca_df.iloc[i + 1]["time_min"]),
                "dPC1": float(step[0]),
                "dPC2": float(step[1]),
                "dPC3": float(step[2]),
                "step_length_3d": float(np.linalg.norm(step)),
            }
        )
    steps = pd.DataFrame(step_records)

    time_summary.to_csv(f"{interim_dir}/audit_time_summary.csv", index=False)
    expr_summary.to_csv(f"{interim_dir}/audit_expression_summary.csv", index=False)
    pca_df.to_csv(f"{interim_dir}/audit_pca_scores.csv", index=False)
    steps.to_csv(f"{interim_dir}/audit_time_steps.csv", index=False)
    feature_var.head(200).rename("variance").to_csv(f"{interim_dir}/audit_top_feature_variances.csv")

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_rows": int(rows),
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "missing_expression_values": int(missing_expr),
        "pca_explained_variance_ratio": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "PC3": float(pca.explained_variance_ratio_[2]),
        },
        "ordered_timepoints_min": [float(x) for x in pca_df["time_min"].tolist()],
    }

    with open(f"{logs_dir}/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{interim_dir}/audit_time_summary.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_expression_summary.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_pca_scores.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_time_steps.csv")
    print("[ok] wrote:", f"{interim_dir}/audit_top_feature_variances.csv")
    print("[ok] wrote:", f"{logs_dir}/audit_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit GSE4370 and compute first-pass PCA geometry.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
