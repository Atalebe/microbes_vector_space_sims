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
    processed_root = Path(cfg.get("processed_root", "data/processed/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))

    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("interim_root", str(interim_root))
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


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    interim_root = Path(cfg["interim_root"])
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(processed_root, logs_root)

    master = pd.read_parquet(interim_root / "master_table.parquet")
    sample_meta = pd.read_csv(interim_root / "sample_metadata.csv")

    master["expression_value"] = pd.to_numeric(master["expression_value"], errors="coerce")

    wide = master.pivot_table(
        index="sample_accession",
        columns="feature_id",
        values="expression_value",
        aggfunc="mean",
    )
    wide_log = np.log1p(wide)
    wide_filled = wide_log.apply(lambda c: c.fillna(c.median()), axis=0)

    pca = PCA(n_components=min(3, wide_filled.shape[0], wide_filled.shape[1]))
    pcs = pca.fit_transform(wide_filled.values)
    pca_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]

    sample_scores = pd.DataFrame(pcs, columns=pca_cols)
    sample_scores.insert(0, "sample_accession", wide_filled.index)
    sample_scores = sample_scores.merge(
        sample_meta[
            [
                "sample_accession",
                "raw_sample_label",
                "geo_accession",
                "series_sample_title",
                "sample_description",
                "replicate",
                "treatment",
                "time_min",
                "condition",
            ]
        ],
        on="sample_accession",
        how="left",
    )

    if "PC3" not in sample_scores.columns:
        sample_scores["PC3"] = 0.0

    cond_centroids = (
        sample_scores.groupby(["treatment", "time_min", "condition"], dropna=False)[["PC1", "PC2", "PC3"]]
        .mean()
        .reset_index()
        .sort_values(["treatment", "time_min"])
    )

    # Control reference centroids
    control = cond_centroids[cond_centroids["treatment"] == "control"].copy().sort_values("time_min")
    if control.empty:
        raise ValueError("No control branch found in GSE95575.")

    control_t0 = control.loc[control["time_min"].idxmin(), ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    control_t60 = control.loc[control["time_min"].idxmax(), ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)

    # Treatment centroids and late disrupted sector proxy
    treated = cond_centroids[cond_centroids["treatment"].isin(["hgcl2", "pma"])].copy()
    disrupted_ref = (
        treated[treated["time_min"] == treated["time_min"].max()][["PC1", "PC2", "PC3"]]
        .mean()
        .to_numpy(dtype=float)
    )

    # Replicate tightness per condition
    rep_tightness = (
        sample_scores.groupby("condition")[["PC1", "PC2", "PC3"]]
        .std()
        .fillna(0.0)
        .sum(axis=1)
        .rename("replicate_spread")
        .reset_index()
    )
    sample_scores = sample_scores.merge(rep_tightness, on="condition", how="left")

    # Helper maps
    cond_to_centroid = {
        row["condition"]: row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        for _, row in cond_centroids.iterrows()
    }
    control_by_time = {
        float(row["time_min"]): row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        for _, row in control.iterrows()
    }

    def euclid(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    H_raw = []
    S_raw = []
    M_raw = []
    R_raw = []
    burden_raw = []
    dist_to_control_t0 = []
    dist_to_control_t60 = []
    dist_to_matched_control = []
    dist_to_disrupted_ref = []
    branch_progress = []
    memory_deviation_from_control = []

    control_vec = control_t60 - control_t0
    control_len2 = float(np.dot(control_vec, control_vec))
    if control_len2 == 0:
        control_len2 = 1.0

    for _, row in sample_scores.iterrows():
        p = row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        t = float(row["time_min"])
        treatment = row["treatment"]
        cond = row["condition"]

        d0 = euclid(p, control_t0)
        dlate = euclid(p, control_t60)
        dmatch = euclid(p, control_by_time[t])
        ddisrupt = euclid(p, disrupted_ref)

        # H: closer to late coherent control, farther from disrupted reference
        h = ddisrupt - dlate

        # S: small matched-control deviation and tight replicate cloud
        s = -(dmatch + float(row["replicate_spread"]))

        # M: treatment-specific trajectory position minus control-expected progress
        # For control, this reduces to normalized control progress.
        proj_control = float(np.dot(p - control_t0, control_vec) / control_len2)
        proj_control = float(np.clip(proj_control, 0.0, 1.0))

        treatment_branch = cond_centroids[cond_centroids["treatment"] == treatment].sort_values("time_min")
        t_start = treatment_branch.loc[treatment_branch["time_min"].idxmin(), ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        t_end = treatment_branch.loc[treatment_branch["time_min"].idxmax(), ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        t_vec = t_end - t_start
        t_len2 = float(np.dot(t_vec, t_vec))
        if t_len2 == 0:
            t_len2 = 1.0
        proj_treat = float(np.dot(p - t_start, t_vec) / t_len2)
        proj_treat = float(np.clip(proj_treat, 0.0, 1.0))

        expected_progress = t / 60.0
        expected_progress = float(np.clip(expected_progress, 0.0, 1.0))

        # Conditioning memory is treatment-progress plus deviation from control progress
        mem = proj_treat + 0.5 * (proj_treat - expected_progress) + 0.5 * (proj_treat - proj_control)

        # R: directional return toward late control endpoint
        r = float(np.dot(control_t60 - p, control_t60 - control_t0) / np.sqrt(control_len2))

        # Burden: matched-control deviation plus disrupted affinity
        b = dmatch + (euclid(p, control_t60) - ddisrupt)

        H_raw.append(h)
        S_raw.append(s)
        M_raw.append(mem)
        R_raw.append(r)
        burden_raw.append(b)
        dist_to_control_t0.append(d0)
        dist_to_control_t60.append(dlate)
        dist_to_matched_control.append(dmatch)
        dist_to_disrupted_ref.append(ddisrupt)
        branch_progress.append(proj_treat)
        memory_deviation_from_control.append(proj_treat - proj_control)

    sample_scores["dist_to_control_t0"] = dist_to_control_t0
    sample_scores["dist_to_control_t60"] = dist_to_control_t60
    sample_scores["dist_to_matched_control"] = dist_to_matched_control
    sample_scores["dist_to_disrupted_ref"] = dist_to_disrupted_ref
    sample_scores["branch_progress_raw"] = branch_progress
    sample_scores["memory_deviation_from_control_raw"] = memory_deviation_from_control

    sample_scores["H_raw"] = H_raw
    sample_scores["S_raw"] = S_raw
    sample_scores["M_raw"] = M_raw
    sample_scores["R_raw"] = R_raw
    sample_scores["stress_burden_raw"] = burden_raw

    sample_scores["H"] = robust_z(sample_scores["H_raw"])
    sample_scores["S"] = robust_z(sample_scores["S_raw"])
    sample_scores["M"] = robust_z(sample_scores["M_raw"])
    sample_scores["R"] = robust_z(sample_scores["R_raw"])
    sample_scores["stress_burden"] = robust_z(sample_scores["stress_burden_raw"])

    condition_summary = (
        sample_scores.groupby(["treatment", "time_min", "condition"], dropna=False)
        .agg(
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            M_mean=("M", "mean"),
            R_mean=("R", "mean"),
            burden_mean=("stress_burden", "mean"),
            dist_to_matched_control_mean=("dist_to_matched_control", "mean"),
            dist_to_control_t60_mean=("dist_to_control_t60", "mean"),
            dist_to_disrupted_ref_mean=("dist_to_disrupted_ref", "mean"),
            branch_progress_mean=("branch_progress_raw", "mean"),
            memory_deviation_from_control_mean=("memory_deviation_from_control_raw", "mean"),
        )
        .reset_index()
        .sort_values(["treatment", "time_min"])
    )

    sample_out = processed_root / "gse95575_state_table_samples.csv"
    cond_out = processed_root / "gse95575_state_table_conditions.csv"
    log_out = logs_root / "state_table_summary.json"

    sample_scores.to_csv(sample_out, index=False)
    condition_summary.to_csv(cond_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "n_samples": int(sample_scores["sample_accession"].nunique()),
        "conditions": condition_summary["condition"].astype(str).tolist(),
        "treatments": sorted(sample_scores["treatment"].dropna().astype(str).unique().tolist()),
        "timepoints_min": sorted(sample_scores["time_min"].dropna().astype(float).unique().tolist()),
        "pca_explained_variance_ratio": {
            f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)
        },
        "notes": {
            "H": "Depth proxy from approach toward late coherent control and separation from disrupted treatment sector.",
            "S": "Stability proxy from matched-control coherence and within-condition replicate compactness.",
            "M": "Conditioning-memory proxy from treatment-specific branch progress relative to control-expected progress.",
            "R": "Recoverability from directional return toward the late control endpoint.",
        },
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {sample_out}")
    print(f"[ok] wrote: {cond_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
