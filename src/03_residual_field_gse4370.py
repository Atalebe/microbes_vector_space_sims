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


def classify_phase(row: pd.Series) -> str:
    t = row["time_min"]
    if t == 0:
        return "stationary_start"
    if t <= 30:
        return "early_recovery"
    if t <= 60:
        return "mid_recovery"
    return "late_recovery"


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse4370_state_table_samples.csv").sort_values("time_order").reset_index(drop=True)

    # Backbone in PC1-PC2
    x = samples["PC1"].values
    y = samples["PC2"].values
    coeffs = np.polyfit(x, y, deg=2)
    yhat = np.polyval(coeffs, x)

    samples["pc2_backbone_fit"] = yhat
    samples["residual_pc2"] = samples["PC2"] - samples["pc2_backbone_fit"]
    samples["residual_z"] = robust_z(samples["residual_pc2"])

    # Path distance to ordered sample trajectory in 3D
    pts = samples[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    dists = []
    for i, row in samples.iterrows():
        p = row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        seg_dists = []
        for j in range(len(pts) - 1):
            seg_dists.append(point_to_segment_distance(p, pts[j], pts[j + 1]))
        # each point lies on its own path, but this still measures off-path behavior numerically
        dists.append(min(seg_dists))
    samples["dist_to_time_path"] = dists
    samples["dist_to_time_path_z"] = robust_z(samples["dist_to_time_path"])

    # Residual field
    samples["residual_field_raw"] = (
        0.45 * samples["residual_z"]
        - 0.30 * samples["dist_to_time_path_z"]
        - 0.35 * samples["stress_burden"]
        + 0.45 * samples["R"]
        + 0.20 * samples["S"]
    )
    samples["residual_field"] = robust_z(samples["residual_field_raw"])

    samples["phase"] = samples.apply(classify_phase, axis=1)
    samples["residual_rank_desc"] = samples["residual_field"].rank(ascending=False, method="dense")
    samples["burden_rank_desc"] = samples["stress_burden"].rank(ascending=False, method="dense")
    samples["recoverability_rank_desc"] = samples["R"].rank(ascending=False, method="dense")

    phase_summary = (
        samples.groupby("phase", as_index=False)
        .agg(
            residual_field_mean=("residual_field", "mean"),
            burden_mean=("stress_burden", "mean"),
            R_mean=("R", "mean"),
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
        )
    )

    samples.to_csv(f"{processed_dir}/gse4370_residual_field_samples.csv", index=False)
    phase_summary.to_csv(f"{processed_dir}/gse4370_phase_summary.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "backbone_model": "quadratic PC2 ~ f(PC1)",
        "n_samples": int(len(samples)),
        "top_positive_samples": samples.sort_values("residual_field", ascending=False)["sample_accession"].head(3).tolist(),
        "top_negative_samples": samples.sort_values("residual_field", ascending=True)["sample_accession"].head(3).tolist(),
    }

    with open(f"{logs_dir}/residual_field_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_residual_field_samples.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_phase_summary.csv")
    print("[ok] wrote:", f"{logs_dir}/residual_field_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Residual field construction for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
