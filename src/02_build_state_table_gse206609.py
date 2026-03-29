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
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or pd.isna(mad):
        std = x.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / std
    return 0.67448975 * (x - med) / mad


def euclid_from_cols(df: pd.DataFrame, cols_a: list[str], cols_b: list[str]) -> np.ndarray:
    a = df[cols_a].to_numpy(dtype=float)
    b = df[cols_b].to_numpy(dtype=float)
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    pca_scores = pd.read_csv(f"{interim_dir}/audit_pca_scores.csv")
    centroids = pd.read_csv(f"{interim_dir}/audit_condition_centroids.csv")

    cent = centroids.copy()

    control_t = cent.loc[cent["condition"] == "none_treatment"].iloc[0]
    control_r = cent.loc[cent["condition"] == "none_recovery"].iloc[0]

    out = pca_scores.copy()

    rec_cent = (
        cent.loc[cent["phase"] == "recovery", ["treatment", "PC1_centroid", "PC2_centroid", "PC3_centroid"]]
        .rename(columns={
            "PC1_centroid": "PC1_rec_cent",
            "PC2_centroid": "PC2_rec_cent",
            "PC3_centroid": "PC3_rec_cent",
        })
    )
    treat_cent = (
        cent.loc[cent["phase"] == "treatment", ["treatment", "PC1_centroid", "PC2_centroid", "PC3_centroid"]]
        .rename(columns={
            "PC1_centroid": "PC1_treat_cent",
            "PC2_centroid": "PC2_treat_cent",
            "PC3_centroid": "PC3_treat_cent",
        })
    )

    out = out.merge(rec_cent, on="treatment", how="left")
    out = out.merge(treat_cent, on="treatment", how="left")

    for coord in ["PC1", "PC2", "PC3"]:
        out[f"{coord}_none_treat"] = float(control_t[f"{coord}_centroid"])
        out[f"{coord}_none_rec"] = float(control_r[f"{coord}_centroid"])

    out["dist_to_matched_recovery"] = euclid_from_cols(
        out, ["PC1", "PC2", "PC3"], ["PC1_rec_cent", "PC2_rec_cent", "PC3_rec_cent"]
    )
    out["dist_to_matched_treatment"] = euclid_from_cols(
        out, ["PC1", "PC2", "PC3"], ["PC1_treat_cent", "PC2_treat_cent", "PC3_treat_cent"]
    )
    out["dist_to_none_treatment"] = euclid_from_cols(
        out, ["PC1", "PC2", "PC3"], ["PC1_none_treat", "PC2_none_treat", "PC3_none_treat"]
    )
    out["dist_to_none_recovery"] = euclid_from_cols(
        out, ["PC1", "PC2", "PC3"], ["PC1_none_rec", "PC2_none_rec", "PC3_none_rec"]
    )

    out["M"] = out["phase"].map({"treatment": 0.0, "recovery": 1.0}).astype(float)

    H_raw = -out["dist_to_none_recovery"]
    out["H"] = robust_z(H_raw)

    S_raw = -(0.6 * out["dist_to_none_treatment"] + 0.4 * out["dist_to_matched_treatment"])
    out["S"] = robust_z(S_raw)

    return_projection = out["dist_to_matched_treatment"] - out["dist_to_matched_recovery"]
    out["R_raw"] = return_projection
    out["R"] = robust_z(return_projection)

    burden_raw = 0.5 * out["dist_to_none_treatment"] + 0.5 * out["dist_to_none_recovery"]
    out["stress_burden_raw"] = burden_raw
    out["stress_burden"] = robust_z(burden_raw)

    keep = [
        "sample_accession", "sample_title", "treatment", "phase", "replicate", "condition",
        "PC1", "PC2", "PC3",
        "dist_to_matched_recovery", "dist_to_matched_treatment",
        "dist_to_none_treatment", "dist_to_none_recovery",
        "H", "S", "M", "R_raw", "R", "stress_burden_raw", "stress_burden"
    ]
    out_keep = out[keep].copy()

    cond_summary = (
        out_keep.groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            M_mean=("M", "mean"),
            R_mean=("R", "mean"),
            burden_mean=("stress_burden", "mean"),
            dist_to_none_treatment_mean=("dist_to_none_treatment", "mean"),
            dist_to_none_recovery_mean=("dist_to_none_recovery", "mean"),
            dist_to_matched_treatment_mean=("dist_to_matched_treatment", "mean"),
            dist_to_matched_recovery_mean=("dist_to_matched_recovery", "mean"),
        )
        .sort_values(["phase", "treatment"])
    )

    out_keep.to_csv(f"{processed_dir}/gse206609_state_table_samples.csv", index=False)
    cond_summary.to_csv(f"{processed_dir}/gse206609_state_table_conditions.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_samples": int(len(out_keep)),
        "conditions": cond_summary["condition"].tolist(),
        "notes": {
            "H": "Depth proxy from approach toward control recovery structure",
            "S": "Stability proxy from reduced treatment/control burden",
            "M": "Binary phase memory coordinate: treatment=0, recovery=1",
            "R": "Return projection from matched treatment centroid toward matched recovery centroid"
        },
    }

    with open(f"{logs_dir}/state_table_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_state_table_samples.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_state_table_conditions.csv")
    print("[ok] wrote:", f"{logs_dir}/state_table_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
