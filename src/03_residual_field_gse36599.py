from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def classify_sector(row: pd.Series) -> str:
    # provisional sector logic, intentionally simple and readable
    if row["condition"] == "during_stress_adaptation":
        if row["stress_burden"] > 1.0 or row["R"] < -1.0:
            return "acute_stress"
        return "burdened"

    if row["condition"] == "after_stress_adaptation":
        if row["residual_z"] > 0.5 and row["S"] > 0:
            return "adaptive"
        return "burdened"

    if row["condition"] == "during_stress_recovery":
        if row["R"] > 0:
            return "restorative"
        return "intermediate"

    if row["condition"] == "after_stress_recovery":
        if row["H"] > 0 and row["R"] > 0:
            return "coherent"
        return "restorative"

    if row["condition"] == "normal_growth":
        if row["H"] > 0:
            return "coherent"
        return "intermediate"

    return "intermediate"


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse36599_state_table_samples.csv")
    condition_state = pd.read_csv(f"{processed_dir}/gse36599_state_table_conditions.csv")

    # ----------------------------
    # 1. Backbone fit in PC1-PC2 space
    # ----------------------------
    x = samples["PC1"].values
    y = samples["PC2"].values

    # simple quadratic backbone
    coeffs = np.polyfit(x, y, deg=2)
    yhat = np.polyval(coeffs, x)

    samples["pc2_backbone_fit"] = yhat
    samples["residual_pc2"] = samples["PC2"] - samples["pc2_backbone_fit"]
    samples["residual_z"] = robust_z(samples["residual_pc2"])

    # ----------------------------
    # 2. Ordered centroid path distance in 3D
    # ----------------------------
    centroids = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
        .sort_values("condition_order")
    )

    centroid_points = centroids[["PC1_centroid", "PC2_centroid", "PC3_centroid"]].to_numpy()

    dists = []
    for _, row in samples.iterrows():
        p = np.array([row["PC1"], row["PC2"], row["PC3"]], dtype=float)
        seg_dists = []
        for i in range(len(centroid_points) - 1):
            a = centroid_points[i]
            b = centroid_points[i + 1]
            seg_dists.append(point_to_segment_distance(p, a, b))
        dists.append(min(seg_dists))

    samples["dist_to_centroid_path"] = dists
    samples["dist_to_centroid_path_z"] = robust_z(samples["dist_to_centroid_path"])

    # ----------------------------
    # 3. Provisional residual field
    # ----------------------------
    samples["residual_field_raw"] = (
        0.6 * samples["residual_z"]
        - 0.5 * samples["dist_to_centroid_path_z"]
        - 0.4 * samples["stress_burden"]
        + 0.5 * samples["R"]
    )
    samples["residual_field"] = robust_z(samples["residual_field_raw"])

    # sector assignment
    samples["sector"] = samples.apply(classify_sector, axis=1)

    # sample-level rank
    samples["residual_rank_desc"] = samples["residual_field"].rank(ascending=False, method="dense")
    samples["burden_rank_desc"] = samples["stress_burden"].rank(ascending=False, method="dense")
    samples["recoverability_rank_desc"] = samples["R"].rank(ascending=False, method="dense")

    # ----------------------------
    # 4. Condition summaries
    # ----------------------------
    cond_resid = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            residual_field_mean=("residual_field", "mean"),
            residual_pc2_mean=("residual_pc2", "mean"),
            residual_z_mean=("residual_z", "mean"),
            path_dist_mean=("dist_to_centroid_path", "mean"),
            path_dist_z_mean=("dist_to_centroid_path_z", "mean"),
            burden_mean=("stress_burden", "mean"),
            R_mean=("R", "mean"),
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
        )
        .sort_values("condition_order")
    )

    sector_counts = (
        samples.groupby(["condition", "condition_order", "sector"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["condition_order", "sector"])
    )

    top_positive = samples.sort_values("residual_field", ascending=False).head(5)
    top_negative = samples.sort_values("residual_field", ascending=True).head(5)

    # ----------------------------
    # 5. Save
    # ----------------------------
    samples.to_csv(f"{processed_dir}/gse36599_residual_field_samples.csv", index=False)
    cond_resid.to_csv(f"{processed_dir}/gse36599_residual_field_conditions.csv", index=False)
    sector_counts.to_csv(f"{processed_dir}/gse36599_sector_counts.csv", index=False)
    top_positive.to_csv(f"{processed_dir}/gse36599_top_positive_residuals.csv", index=False)
    top_negative.to_csv(f"{processed_dir}/gse36599_top_negative_residuals.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "backbone_model": "quadratic PC2 ~ f(PC1)",
        "n_samples": int(len(samples)),
        "residual_field_mean": float(samples["residual_field"].mean()),
        "residual_field_std": float(samples["residual_field"].std()),
        "sector_counts_total": samples["sector"].value_counts().to_dict(),
        "top_positive_samples": top_positive["sample_accession"].tolist(),
        "top_negative_samples": top_negative["sample_accession"].tolist(),
    }

    with open(f"{logs_dir}/residual_field_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_residual_field_samples.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_residual_field_conditions.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_sector_counts.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_top_positive_residuals.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_top_negative_residuals.csv")
    print("[ok] wrote:", f"{logs_dir}/residual_field_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Residual field construction for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
