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


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - np.nanmean(x)) / std
    return 0.67448975 * (x - med) / mad


def minmax01(x: pd.Series) -> pd.Series:
    xmin = x.min()
    xmax = x.max()
    if pd.isna(xmin) or pd.isna(xmax) or xmax == xmin:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    master = pd.read_parquet(cfg["output"]["master_table"])
    sample_meta = pd.read_csv(cfg["output"]["sample_metadata"])

    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(interim_dir)
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    # Wide matrix: features x samples
    wide = master.pivot_table(
        index="feature_id",
        columns="sample_accession",
        values="expression_value",
        aggfunc="mean",
    )

    ordered_samples = (
        sample_meta.sort_values(["condition_order", "replicate"])["sample_accession"].tolist()
    )
    wide = wide[ordered_samples]

    # PCA on top variable features
    feature_var = wide.var(axis=1).sort_values(ascending=False)
    top_features = feature_var.head(1000).index.tolist()
    X = wide.loc[top_features].T.fillna(wide.loc[top_features].T.mean())

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)

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
    )

    # Condition centroids
    centroids = (
        pca_df.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
        .sort_values("condition_order")
    )

    baseline = centroids.loc[centroids["condition"] == "normal_growth"].iloc[0]
    acute = centroids.loc[centroids["condition"] == "during_stress_adaptation"].iloc[0]
    after_recovery = centroids.loc[centroids["condition"] == "after_stress_recovery"].iloc[0]

    bvec = np.array([baseline["PC1_centroid"], baseline["PC2_centroid"], baseline["PC3_centroid"]])
    avec = np.array([acute["PC1_centroid"], acute["PC2_centroid"], acute["PC3_centroid"]])
    rvec = np.array([after_recovery["PC1_centroid"], after_recovery["PC2_centroid"], after_recovery["PC3_centroid"]])

    def row_vec(row):
        return np.array([row["PC1"], row["PC2"], row["PC3"]])

    # Distances to biologically important centroids
    pca_df["dist_to_baseline"] = pca_df.apply(lambda r: np.linalg.norm(row_vec(r) - bvec), axis=1)
    pca_df["dist_to_acute_stress"] = pca_df.apply(lambda r: np.linalg.norm(row_vec(r) - avec), axis=1)
    pca_df["dist_to_after_recovery"] = pca_df.apply(lambda r: np.linalg.norm(row_vec(r) - rvec), axis=1)

    # Memory stage: explicit first-pass history coordinate from ordered condition
    pca_df["M_stage_raw"] = pca_df["condition_order"].astype(float)
    pca_df["M"] = minmax01(pca_df["M_stage_raw"])

    # First-pass H:
    # interpret higher H as being closer to baseline / recovered organized states
    # and farther from the acute-stress centroid
    h_raw = -0.5 * pca_df["dist_to_baseline"] - 0.5 * pca_df["dist_to_after_recovery"] + pca_df["dist_to_acute_stress"]
    pca_df["H_raw"] = h_raw
    pca_df["H"] = robust_z(pca_df["H_raw"])

    # First-pass S:
    # interpret higher S as lower stress burden, approximated by lower distance from the
    # baseline-recovery corridor and lower acute displacement.
    corridor_mid = 0.5 * (bvec + rvec)

    def dist_to_corridor_mid(row):
        return np.linalg.norm(row_vec(row) - corridor_mid)

    pca_df["dist_to_baseline_recovery_mid"] = pca_df.apply(dist_to_corridor_mid, axis=1)
    s_raw = -0.7 * pca_df["dist_to_acute_stress"] - 0.3 * pca_df["dist_to_baseline_recovery_mid"]
    pca_df["S_raw"] = s_raw
    pca_df["S"] = robust_z(pca_df["S_raw"])

    # First-pass latent R:
    # recoverability is higher when a sample is close to the recovered state and
    # lies in the return direction away from acute stress.
    stress_to_recovery = rvec - avec
    norm_sr = np.linalg.norm(stress_to_recovery)
    if norm_sr == 0:
        stress_to_recovery_unit = stress_to_recovery
    else:
        stress_to_recovery_unit = stress_to_recovery / norm_sr

    def return_projection(row):
        vec = row_vec(row) - avec
        return float(np.dot(vec, stress_to_recovery_unit))

    pca_df["return_projection_from_acute"] = pca_df.apply(return_projection, axis=1)
    r_raw = pca_df["return_projection_from_acute"] - pca_df["dist_to_after_recovery"]
    pca_df["R_raw"] = r_raw
    pca_df["R"] = robust_z(pca_df["R_raw"])

    # Useful burden proxy
    pca_df["stress_burden_raw"] = pca_df["dist_to_baseline"] + pca_df["dist_to_after_recovery"] - pca_df["dist_to_acute_stress"]
    pca_df["stress_burden"] = robust_z(pca_df["stress_burden_raw"])

    # Condition-level state summary
    condition_state = (
        pca_df.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            M_mean=("M", "mean"),
            R_mean=("R", "mean"),
            burden_mean=("stress_burden", "mean"),
            dist_to_baseline_mean=("dist_to_baseline", "mean"),
            dist_to_after_recovery_mean=("dist_to_after_recovery", "mean"),
            dist_to_acute_stress_mean=("dist_to_acute_stress", "mean"),
        )
        .sort_values("condition_order")
    )

    # Save outputs
    pca_df.to_csv(f"{processed_dir}/gse36599_state_table_samples.csv", index=False)
    condition_state.to_csv(f"{processed_dir}/gse36599_state_table_conditions.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_samples": int(len(pca_df)),
        "condition_order": condition_state["condition"].tolist(),
        "pca_explained_variance_ratio": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "PC3": float(pca.explained_variance_ratio_[2]),
        },
        "notes": {
            "H": "First-pass depth proxy from organized-state proximity versus acute-stress displacement",
            "S": "First-pass stability proxy from low acute burden and proximity to baseline-recovery corridor",
            "M": "Explicit ordered exposure-history stage",
            "R": "Latent recoverability from return-direction projection toward recovered state",
        },
    }

    with open(f"{logs_dir}/state_table_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_state_table_samples.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_state_table_conditions.csv")
    print("[ok] wrote:", f"{logs_dir}/state_table_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build first-pass H,S,M,R state table for GSE36599.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
