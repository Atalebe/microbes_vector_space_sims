#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    processed_root = Path(cfg.get("processed_root", "data/processed/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))

    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("processed_root", str(processed_root))
    cfg.setdefault("logs_root", str(logs_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return (x - med) / std
    return 0.67448975 * (x - med) / mad


def classify_sector(row: pd.Series) -> str:
    rf = row["residual_field"]
    burden = row["stress_burden"]
    r = row["R"]

    if rf > 0.7 and burden < 0 and r > 0:
        return "coherent"
    if rf < -0.7 and burden > 0.5 and r < 0:
        return "burdened"
    if r > 0 and burden <= 0:
        return "restorative"
    if burden > 0.5 and rf < 0:
        return "fragile"
    return "intermediate"


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(processed_root, logs_root)

    df = pd.read_csv(processed_root / "gse95575_state_table_samples.csv")

    x = df["PC1"].to_numpy(dtype=float)
    y = df["PC2"].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, deg=2)
    yhat = np.polyval(coeffs, x)

    df["pc2_backbone_fit"] = yhat
    df["residual_pc2"] = df["PC2"] - df["pc2_backbone_fit"]
    df["residual_z"] = robust_z(df["residual_pc2"])

    # centroid path by condition means
    cond = (
        df.groupby(["treatment", "time_min", "condition"], dropna=False)[["PC1", "PC2", "PC3"]]
        .mean()
        .reset_index()
    )
    path_points = cond[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)

    def min_dist_to_path(p: np.ndarray, pts: np.ndarray) -> float:
        return float(np.min(np.linalg.norm(pts - p[None, :], axis=1)))

    dpath = []
    for _, row in df.iterrows():
        p = row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        dpath.append(min_dist_to_path(p, path_points))
    df["dist_to_centroid_path"] = dpath
    df["dist_to_centroid_path_z"] = robust_z(df["dist_to_centroid_path"])

    df["residual_field_raw"] = (
        df["residual_z"] - 0.5 * df["dist_to_centroid_path_z"] + 0.35 * df["R"] - 0.35 * df["stress_burden"]
    )
    df["residual_field"] = robust_z(df["residual_field_raw"])
    df["sector"] = df.apply(classify_sector, axis=1)

    phase_summary = (
        df.groupby(["treatment", "time_min", "condition"], dropna=False)
        .agg(
            residual_field_mean=("residual_field", "mean"),
            residual_pc2_mean=("residual_pc2", "mean"),
            burden_mean=("stress_burden", "mean"),
            R_mean=("R", "mean"),
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            M_mean=("M", "mean"),
        )
        .reset_index()
        .sort_values(["treatment", "time_min"])
    )

    sector_counts = (
        df.groupby(["treatment", "condition", "sector"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["treatment", "condition", "sector"])
    )

    top_positive = df.sort_values("residual_field", ascending=False).head(10)
    top_negative = df.sort_values("residual_field", ascending=True).head(10)

    samples_out = processed_root / "gse95575_residual_field_samples.csv"
    phase_out = processed_root / "gse95575_phase_summary.csv"
    sectors_out = processed_root / "gse95575_sector_counts.csv"
    pos_out = processed_root / "gse95575_top_positive_residuals.csv"
    neg_out = processed_root / "gse95575_top_negative_residuals.csv"
    log_out = logs_root / "residual_field_summary.json"

    df.to_csv(samples_out, index=False)
    phase_summary.to_csv(phase_out, index=False)
    sector_counts.to_csv(sectors_out, index=False)
    top_positive.to_csv(pos_out, index=False)
    top_negative.to_csv(neg_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "backbone_model": "quadratic PC2 ~ f(PC1)",
        "n_samples": int(df["sample_accession"].nunique()),
        "sector_counts_total": {k: int(v) for k, v in df["sector"].value_counts().to_dict().items()},
        "top_positive_samples": top_positive["sample_accession"].astype(str).tolist()[:5],
        "top_negative_samples": top_negative["sample_accession"].astype(str).tolist()[:5],
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {samples_out}")
    print(f"[ok] wrote: {phase_out}")
    print(f"[ok] wrote: {sectors_out}")
    print(f"[ok] wrote: {pos_out}")
    print(f"[ok] wrote: {neg_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
