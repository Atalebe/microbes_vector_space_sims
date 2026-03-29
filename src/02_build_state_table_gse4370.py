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

    master = pd.read_parquet(cfg["output"]["master_table"])
    sample_meta = pd.read_csv(cfg["output"]["sample_metadata"])

    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    wide = master.pivot_table(
        index="feature_id",
        columns="sample_accession",
        values="expression_value",
        aggfunc="mean",
    )

    ordered_samples = sample_meta.sort_values("time_order")["sample_accession"].tolist()
    wide = wide[ordered_samples]

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
        sample_meta[["sample_accession", "sample_title", "condition", "time_min", "time_order"]],
        on="sample_accession",
        how="left",
    ).sort_values("time_order").reset_index(drop=True)

    # anchors
    start = pca_df.iloc[0][["PC1", "PC2", "PC3"]].to_numpy(dtype=float)   # 0 min
    end = pca_df.iloc[-1][["PC1", "PC2", "PC3"]].to_numpy(dtype=float)    # 90 min
    early = pca_df.iloc[1][["PC1", "PC2", "PC3"]].to_numpy(dtype=float)   # 15 min

    def row_vec(row):
        return row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)

    # explicit time-memory
    pca_df["M_raw"] = pca_df["time_min"]
    pca_df["M"] = minmax01(pca_df["M_raw"])

    # depth H: organized progression away from start disruption toward later recovered side
    pca_df["dist_to_start"] = pca_df.apply(lambda r: float(np.linalg.norm(row_vec(r) - start)), axis=1)
    pca_df["dist_to_end"] = pca_df.apply(lambda r: float(np.linalg.norm(row_vec(r) - end)), axis=1)
    pca_df["dist_to_early"] = pca_df.apply(lambda r: float(np.linalg.norm(row_vec(r) - early)), axis=1)

    h_raw = -0.6 * pca_df["dist_to_end"] + 0.4 * pca_df["dist_to_early"] - 0.2 * pca_df["dist_to_start"]
    pca_df["H_raw"] = h_raw
    pca_df["H"] = robust_z(pca_df["H_raw"])

    # stability S: coherence with overall recovery corridor (start -> end)
    corridor_dist = []
    corridor_t = []
    for _, row in pca_df.iterrows():
        d, t = point_to_segment_distance_and_t(row_vec(row), start, end)
        corridor_dist.append(d)
        corridor_t.append(t)

    pca_df["dist_to_recovery_corridor"] = corridor_dist
    pca_df["corridor_position_t"] = corridor_t

    # smooth ordered position: compare corridor_t to normalized time
    pca_df["time_norm"] = minmax01(pca_df["time_min"])
    pca_df["corridor_time_mismatch"] = (pca_df["corridor_position_t"] - pca_df["time_norm"]).abs()

    s_raw = (
        -0.7 * robust_z(pca_df["dist_to_recovery_corridor"])
        -0.5 * robust_z(pca_df["corridor_time_mismatch"])
        +0.2 * robust_z(pca_df["dist_to_early"])
    )
    pca_df["S_raw"] = s_raw
    pca_df["S"] = robust_z(pca_df["S_raw"])

    # recoverability R: directional return from early recovery toward endpoint
    recovery_vec = end - early
    rec_norm = np.linalg.norm(recovery_vec)
    recovery_unit = recovery_vec / rec_norm if rec_norm > 0 else recovery_vec

    def recovery_projection(row):
        vec = row_vec(row) - early
        return float(np.dot(vec, recovery_unit))

    pca_df["return_projection_from_early"] = pca_df.apply(recovery_projection, axis=1)
    r_raw = pca_df["return_projection_from_early"] - pca_df["dist_to_end"]
    pca_df["R_raw"] = r_raw
    pca_df["R"] = robust_z(pca_df["R_raw"])

    # burden proxy: distance from end and early domination
    burden_raw = pca_df["dist_to_end"] - 0.5 * pca_df["dist_to_start"] - 0.3 * pca_df["return_projection_from_early"]
    pca_df["stress_burden_raw"] = burden_raw
    pca_df["stress_burden"] = robust_z(pca_df["stress_burden_raw"])

    pca_df.to_csv(f"{processed_dir}/gse4370_state_table_samples.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_samples": int(len(pca_df)),
        "ordered_timepoints_min": [float(x) for x in pca_df["time_min"].tolist()],
        "pca_explained_variance_ratio": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "PC3": float(pca.explained_variance_ratio_[2]),
        },
        "notes": {
            "H": "Depth proxy from approach toward late recovered structure and separation from early disrupted state",
            "S": "Stability proxy from coherence with the overall recovery corridor and smooth ordered placement",
            "M": "Normalized recovery time",
            "R": "Recoverability from directional return toward the late-recovery endpoint",
        },
    }

    with open(f"{logs_dir}/state_table_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_state_table_samples.csv")
    print("[ok] wrote:", f"{logs_dir}/state_table_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build first-pass H,S,M,R state table for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
