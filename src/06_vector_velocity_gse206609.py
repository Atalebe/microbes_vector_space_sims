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


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    pca = pd.read_csv(f"{interim_dir}/audit_pca_scores.csv")
    samples = pd.read_csv(f"{processed_dir}/gse206609_state_table_samples.csv")

    centroids = (
        pca.groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(
            PC1_centroid=("PC1", "mean"),
            PC2_centroid=("PC2", "mean"),
            PC3_centroid=("PC3", "mean"),
        )
    )

    treat = centroids.loc[centroids["phase"] == "treatment"].copy()
    rec = centroids.loc[centroids["phase"] == "recovery"].copy()

    steps = treat.merge(rec, on="treatment", suffixes=("_treat", "_rec"))
    steps["dPC1"] = steps["PC1_centroid_rec"] - steps["PC1_centroid_treat"]
    steps["dPC2"] = steps["PC2_centroid_rec"] - steps["PC2_centroid_treat"]
    steps["dPC3"] = steps["PC3_centroid_rec"] - steps["PC3_centroid_treat"]
    steps["step_length_3d"] = np.sqrt(steps["dPC1"] ** 2 + steps["dPC2"] ** 2 + steps["dPC3"] ** 2)

    # return efficiency using matched treatment vs recovery distances
    conds = pd.read_csv(f"{processed_dir}/gse206609_state_table_conditions.csv")
    trec = conds.loc[conds["phase"] == "treatment", ["treatment", "dist_to_none_recovery_mean"]].rename(
        columns={"dist_to_none_recovery_mean": "treat_dist_to_recovery_anchor"}
    )
    rrec = conds.loc[conds["phase"] == "recovery", ["treatment", "dist_to_none_recovery_mean"]].rename(
        columns={"dist_to_none_recovery_mean": "rec_dist_to_recovery_anchor"}
    )
    steps = steps.merge(trec, on="treatment", how="left").merge(rrec, on="treatment", how="left")
    steps["anchor_return_drop"] = steps["treat_dist_to_recovery_anchor"] - steps["rec_dist_to_recovery_anchor"]
    steps["anchor_return_efficiency"] = steps["anchor_return_drop"] / steps["step_length_3d"].replace(0, np.nan)

    steps.to_csv(f"{processed_dir}/gse206609_centroid_steps.csv", index=False)

    sample_shift = (
        samples.groupby(["treatment", "phase"], as_index=False)
        .agg(
            mean_R=("R", "mean"),
            mean_burden=("stress_burden", "mean"),
            mean_H=("H", "mean"),
            mean_S=("S", "mean"),
        )
    )
    sample_shift.to_csv(f"{processed_dir}/gse206609_sample_velocity_summary.csv", index=False)

    best = steps.sort_values("anchor_return_efficiency", ascending=False).iloc[0].to_dict()

    summary = {
        "dataset_name": cfg["dataset_name"],
        "best_return_drug": best["treatment"],
        "best_anchor_return_efficiency": float(best["anchor_return_efficiency"]),
        "largest_step_drug": steps.sort_values("step_length_3d", ascending=False).iloc[0]["treatment"],
    }

    with open(f"{logs_dir}/vector_velocity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_centroid_steps.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_sample_velocity_summary.csv")
    print("[ok] wrote:", f"{logs_dir}/vector_velocity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
