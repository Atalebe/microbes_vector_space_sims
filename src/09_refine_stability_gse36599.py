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


def point_to_segment_distance_and_t(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return float(np.linalg.norm(p - a)), 0.0
    t = np.dot(p - a, ab) / denom
    t_clip = max(0.0, min(1.0, t))
    proj = a + t_clip * ab
    return float(np.linalg.norm(p - proj)), float(t_clip)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse36599_residual_field_samples.csv")

    # -----------------------------------------
    # 1. Condition centroids and local coherence
    # -----------------------------------------
    centroids = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
        .sort_values("condition_order")
    )

    centroid_map = {
        row["condition"]: np.array([row["PC1_centroid"], row["PC2_centroid"], row["PC3_centroid"]], dtype=float)
        for _, row in centroids.iterrows()
    }

    def row_vec(row: pd.Series) -> np.ndarray:
        return np.array([row["PC1"], row["PC2"], row["PC3"]], dtype=float)

    # distance to own condition centroid = local instability
    samples["dist_to_own_centroid"] = samples.apply(
        lambda r: float(np.linalg.norm(row_vec(r) - centroid_map[r["condition"]])),
        axis=1,
    )
    samples["dist_to_own_centroid_z"] = robust_z(samples["dist_to_own_centroid"])

    # -----------------------------------------
    # 2. Baseline-recovery corridor coherence
    # -----------------------------------------
    baseline = centroids.loc[centroids["condition"] == "normal_growth"].iloc[0]
    recovered = centroids.loc[centroids["condition"] == "after_stress_recovery"].iloc[0]
    acute = centroids.loc[centroids["condition"] == "during_stress_adaptation"].iloc[0]

    bvec = np.array([baseline["PC1_centroid"], baseline["PC2_centroid"], baseline["PC3_centroid"]], dtype=float)
    rvec = np.array([recovered["PC1_centroid"], recovered["PC2_centroid"], recovered["PC3_centroid"]], dtype=float)
    avec = np.array([acute["PC1_centroid"], acute["PC2_centroid"], acute["PC3_centroid"]], dtype=float)

    corridor_dist = []
    corridor_t = []
    for _, row in samples.iterrows():
        d, t = point_to_segment_distance_and_t(row_vec(row), bvec, rvec)
        corridor_dist.append(d)
        corridor_t.append(t)

    samples["dist_to_baseline_recovery_corridor"] = corridor_dist
    samples["corridor_position_t"] = corridor_t
    samples["dist_to_baseline_recovery_corridor_z"] = robust_z(samples["dist_to_baseline_recovery_corridor"])

    # -----------------------------------------
    # 3. Acute-stress domination
    # -----------------------------------------
    samples["dist_to_acute_centroid"] = samples.apply(
        lambda r: float(np.linalg.norm(row_vec(r) - avec)),
        axis=1,
    )
    samples["dist_to_acute_centroid_z"] = robust_z(samples["dist_to_acute_centroid"])

    # -----------------------------------------
    # 4. Refined S_v2
    # Higher S_v2 = tighter local coherence, closer to corridor, less acute distortion
    # -----------------------------------------
    s2_raw = (
        -0.50 * samples["dist_to_own_centroid_z"]
        -0.35 * samples["dist_to_baseline_recovery_corridor_z"]
        +0.40 * samples["dist_to_acute_centroid_z"]
    )
    samples["S_v2_raw"] = s2_raw
    samples["S_v2"] = robust_z(samples["S_v2_raw"])

    # -----------------------------------------
    # 5. Updated residual field using S_v2
    # -----------------------------------------
    samples["residual_field_v2_raw"] = (
        0.45 * samples["residual_z"]
        -0.35 * samples["dist_to_centroid_path_z"]
        -0.25 * samples["stress_burden"]
        +0.35 * samples["R"]
        +0.30 * samples["S_v2"]
    )
    samples["residual_field_v2"] = robust_z(samples["residual_field_v2_raw"])

    # -----------------------------------------
    # 6. Condition summaries
    # -----------------------------------------
    cond = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            S_old_mean=("S", "mean"),
            S_v2_mean=("S_v2", "mean"),
            residual_field_old_mean=("residual_field", "mean"),
            residual_field_v2_mean=("residual_field_v2", "mean"),
            own_centroid_dist_mean=("dist_to_own_centroid", "mean"),
            corridor_dist_mean=("dist_to_baseline_recovery_corridor", "mean"),
            acute_dist_mean=("dist_to_acute_centroid", "mean"),
        )
        .sort_values("condition_order")
    )

    # ranking diagnostics
    s_old_rank = (
        samples.groupby(["condition", "condition_order"], as_index=False)["S"]
        .mean()
        .sort_values("S", ascending=False)["condition"]
        .tolist()
    )
    s_v2_rank = (
        samples.groupby(["condition", "condition_order"], as_index=False)["S_v2"]
        .mean()
        .sort_values("S_v2", ascending=False)["condition"]
        .tolist()
    )

    samples.to_csv(f"{processed_dir}/gse36599_refined_stability_samples.csv", index=False)
    cond.to_csv(f"{processed_dir}/gse36599_refined_stability_conditions.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "S_old_rank_desc": s_old_rank,
        "S_v2_rank_desc": s_v2_rank,
        "notes": {
            "S_v2": "Refined stability from local centroid coherence, corridor coherence, and reduced acute-stress domination",
            "residual_field_v2": "Updated residual field using S_v2 instead of original S contribution",
        },
    }

    with open(f"{logs_dir}/refined_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_refined_stability_samples.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_refined_stability_conditions.csv")
    print("[ok] wrote:", f"{logs_dir}/refined_stability_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refine S coordinate for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
