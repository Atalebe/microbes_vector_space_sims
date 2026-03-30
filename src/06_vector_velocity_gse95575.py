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


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])
    ensure_dirs(processed_root, logs_root)

    s = pd.read_csv(processed_root / "gse95575_state_table_samples.csv")
    c = pd.read_csv(processed_root / "gse95575_state_table_conditions.csv")

    step_rows = []
    velocity_rows = []

    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min").reset_index(drop=True)

        path_len = 0.0
        largest_step = 0.0
        largest_pair = None

        for i in range(len(sub) - 1):
            a = sub.loc[i, ["H_mean", "S_mean", "M_mean", "R_mean"]].to_numpy(dtype=float)
            b = sub.loc[i + 1, ["H_mean", "S_mean", "M_mean", "R_mean"]].to_numpy(dtype=float)
            step = euclid(a, b)
            path_len += step
            if step > largest_step:
                largest_step = step
                largest_pair = (
                    str(sub.loc[i, "condition"]),
                    str(sub.loc[i + 1, "condition"]),
                )
            step_rows.append(
                {
                    "treatment": treatment,
                    "from_condition": sub.loc[i, "condition"],
                    "to_condition": sub.loc[i + 1, "condition"],
                    "from_time_min": float(sub.loc[i, "time_min"]),
                    "to_time_min": float(sub.loc[i + 1, "time_min"]),
                    "dH": float(sub.loc[i + 1, "H_mean"] - sub.loc[i, "H_mean"]),
                    "dS": float(sub.loc[i + 1, "S_mean"] - sub.loc[i, "S_mean"]),
                    "dM": float(sub.loc[i + 1, "M_mean"] - sub.loc[i, "M_mean"]),
                    "dR": float(sub.loc[i + 1, "R_mean"] - sub.loc[i, "R_mean"]),
                    "step_length_4d": step,
                }
            )

        start = sub.loc[0, ["H_mean", "S_mean", "M_mean", "R_mean"]].to_numpy(dtype=float)
        end = sub.loc[len(sub) - 1, ["H_mean", "S_mean", "M_mean", "R_mean"]].to_numpy(dtype=float)
        direct = euclid(start, end)
        efficiency = direct / path_len if path_len > 0 else np.nan

        velocity_rows.append(
            {
                "treatment": treatment,
                "path_length_4d": path_len,
                "direct_displacement_4d": direct,
                "path_efficiency": efficiency,
                "largest_step_length_4d": largest_step,
                "largest_step_pair": " -> ".join(largest_pair) if largest_pair else None,
                "net_dR": float(sub.loc[len(sub) - 1, "R_mean"] - sub.loc[0, "R_mean"]),
                "net_dM": float(sub.loc[len(sub) - 1, "M_mean"] - sub.loc[0, "M_mean"]),
                "net_burden_change": float(sub.loc[len(sub) - 1, "burden_mean"] - sub.loc[0, "burden_mean"]),
            }
        )

    centroid_steps = pd.DataFrame(step_rows)
    velocity = pd.DataFrame(velocity_rows).sort_values("path_efficiency", ascending=False)

    steps_out = processed_root / "gse95575_centroid_steps.csv"
    vel_out = processed_root / "gse95575_sample_velocity_summary.csv"
    log_out = logs_root / "vector_velocity_summary.json"

    centroid_steps.to_csv(steps_out, index=False)
    velocity.to_csv(vel_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "best_efficiency_treatment": str(velocity.iloc[0]["treatment"]),
        "worst_efficiency_treatment": str(velocity.iloc[-1]["treatment"]),
        "largest_step_treatment": str(
            centroid_steps.sort_values("step_length_4d", ascending=False).iloc[0]["treatment"]
        ),
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {steps_out}")
    print(f"[ok] wrote: {vel_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
