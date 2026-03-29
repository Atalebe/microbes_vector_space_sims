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

    df = pd.read_csv(f"{processed_dir}/gse4370_residual_field_samples.csv").sort_values("time_order").reset_index(drop=True)

    step_records = []
    for i in range(len(df) - 1):
        a = df.loc[i, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        b = df.loc[i + 1, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
        step = b - a
        step_records.append(
            {
                "from_condition": df.loc[i, "condition"],
                "to_condition": df.loc[i + 1, "condition"],
                "from_time_min": float(df.loc[i, "time_min"]),
                "to_time_min": float(df.loc[i + 1, "time_min"]),
                "dPC1": float(step[0]),
                "dPC2": float(step[1]),
                "dPC3": float(step[2]),
                "step_length_3d": float(np.linalg.norm(step)),
            }
        )

    steps = pd.DataFrame(step_records)

    start = df.loc[0, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    end = df.loc[len(df) - 1, ["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    path_length = float(steps["step_length_3d"].sum())
    direct_displacement = float(np.linalg.norm(end - start))
    path_efficiency = direct_displacement / path_length if path_length > 0 else np.nan

    steps.to_csv(f"{processed_dir}/gse4370_centroid_steps.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "path_length_3d": path_length,
        "direct_displacement_3d": direct_displacement,
        "path_efficiency": path_efficiency,
        "largest_step": steps.sort_values("step_length_3d", ascending=False).iloc[0].to_dict(),
    }

    with open(f"{logs_dir}/vector_velocity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_centroid_steps.csv")
    print("[ok] wrote:", f"{logs_dir}/vector_velocity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vector velocity diagnostics for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
