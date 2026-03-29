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
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]

    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse36599_residual_field_samples.csv")

    centroids = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            PC1=("PC1", "mean"),
            PC2=("PC2", "mean"),
            PC3=("PC3", "mean"),
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            M_mean=("M", "mean"),
            R_mean=("R", "mean"),
        )
        .sort_values("condition_order")
        .reset_index(drop=True)
    )

    step_records = []
    for i in range(len(centroids) - 1):
        a = centroids.loc[i, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        b = centroids.loc[i + 1, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        step = b - a
        step_records.append(
            {
                "from_condition": centroids.loc[i, "condition"],
                "to_condition": centroids.loc[i + 1, "condition"],
                "from_order": int(centroids.loc[i, "condition_order"]),
                "to_order": int(centroids.loc[i + 1, "condition_order"]),
                "dPC1": float(step[0]),
                "dPC2": float(step[1]),
                "dPC3": float(step[2]),
                "step_length_3d": float(np.linalg.norm(step)),
            }
        )

    steps = pd.DataFrame(step_records)

    start = centroids.loc[0, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    end = centroids.loc[len(centroids) - 1, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    direct_displacement = float(np.linalg.norm(end - start))
    path_length = float(steps["step_length_3d"].sum())
    path_efficiency = direct_displacement / path_length if path_length > 0 else np.nan

    # Recovery direction: from acute stress to after stress recovery
    acute = centroids.loc[centroids["condition"] == "during_stress_adaptation", ["PC1", "PC2", "PC3"]].iloc[0].to_numpy(dtype=float)
    recovered = centroids.loc[centroids["condition"] == "after_stress_recovery", ["PC1", "PC2", "PC3"]].iloc[0].to_numpy(dtype=float)
    recovery_vec = recovered - acute
    recovery_norm = np.linalg.norm(recovery_vec)
    recovery_unit = recovery_vec / recovery_norm if recovery_norm > 0 else recovery_vec

    sample_velocity_records = []
    for _, row in samples.iterrows():
        p = row[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        order = int(row["condition_order"])

        rec = {
            "sample_accession": row["sample_accession"],
            "sample_title": row["sample_title"],
            "condition": row["condition"],
            "condition_order": order,
            "replicate": row["replicate"],
        }

        if order > 0:
            prev_cent = centroids.loc[centroids["condition_order"] == order - 1, ["PC1", "PC2", "PC3"]].iloc[0].to_numpy(dtype=float)
            prev_vec = p - prev_cent
            rec["from_prev_stage_length"] = float(np.linalg.norm(prev_vec))
            rec["from_prev_stage_recovery_alignment"] = float(np.dot(prev_vec, recovery_unit))
        else:
            rec["from_prev_stage_length"] = np.nan
            rec["from_prev_stage_recovery_alignment"] = np.nan

        if order < centroids["condition_order"].max():
            next_cent = centroids.loc[centroids["condition_order"] == order + 1, ["PC1", "PC2", "PC3"]].iloc[0].to_numpy(dtype=float)
            next_vec = next_cent - p
            rec["to_next_stage_length"] = float(np.linalg.norm(next_vec))
            rec["to_next_stage_recovery_alignment"] = float(np.dot(next_vec, recovery_unit))
        else:
            rec["to_next_stage_length"] = np.nan
            rec["to_next_stage_recovery_alignment"] = np.nan

        rec["dist_to_acute"] = float(np.linalg.norm(p - acute))
        rec["dist_to_recovered"] = float(np.linalg.norm(p - recovered))
        sample_velocity_records.append(rec)

    sample_velocity = pd.DataFrame(sample_velocity_records)

    centroids.to_csv(f"{processed_dir}/gse36599_centroid_trajectory.csv", index=False)
    steps.to_csv(f"{processed_dir}/gse36599_centroid_steps.csv", index=False)
    sample_velocity.to_csv(f"{processed_dir}/gse36599_sample_velocity_summary.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "path_length_3d": path_length,
        "direct_displacement_3d": direct_displacement,
        "path_efficiency": path_efficiency,
        "largest_step": steps.sort_values("step_length_3d", ascending=False).iloc[0].to_dict(),
    }

    with open(f"{logs_dir}/vector_velocity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_centroid_trajectory.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_centroid_steps.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_sample_velocity_summary.csv")
    print("[ok] wrote:", f"{logs_dir}/vector_velocity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector velocity diagnostics for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
