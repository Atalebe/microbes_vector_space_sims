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


def rank_condition_means(df: pd.DataFrame, value_col: str) -> list[str]:
    return (
        df.groupby(["condition", "condition_order"], as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)["condition"]
        .tolist()
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    df = pd.read_csv(f"{processed_dir}/gse36599_refined_stability_samples.csv")

    condition_summary = (
        df.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            S_old_mean=("S", "mean"),
            S_v2_mean=("S_v2", "mean"),
            residual_old_mean=("residual_field", "mean"),
            residual_v2_mean=("residual_field_v2", "mean"),
            burden_mean=("stress_burden", "mean"),
            R_mean=("R", "mean"),
            own_centroid_dist_mean=("dist_to_own_centroid", "mean"),
            corridor_dist_mean=("dist_to_baseline_recovery_corridor", "mean"),
        )
        .sort_values("condition_order")
    )
    condition_summary.to_csv(f"{processed_dir}/gse36599_refined_stability_validation.csv", index=False)

    # frontier sensitivity for S_v2
    acute_removed = df.loc[df["condition"] != "during_stress_adaptation"].copy()

    full_var_old = float(np.var(df["S"], ddof=1))
    full_var_new = float(np.var(df["S_v2"], ddof=1))
    trim_var_old = float(np.var(acute_removed["S"], ddof=1))
    trim_var_new = float(np.var(acute_removed["S_v2"], ddof=1))

    summary = {
        "dataset_name": cfg["dataset_name"],
        "S_old_rank_desc": rank_condition_means(df, "S"),
        "S_v2_rank_desc": rank_condition_means(df, "S_v2"),
        "residual_old_rank_desc": rank_condition_means(df, "residual_field"),
        "residual_v2_rank_desc": rank_condition_means(df, "residual_field_v2"),
        "S_old_variance": full_var_old,
        "S_v2_variance": full_var_new,
        "S_old_variance_without_acute": trim_var_old,
        "S_v2_variance_without_acute": trim_var_new,
        "S_old_ratio_without_acute": trim_var_old / full_var_old if full_var_old > 0 else np.nan,
        "S_v2_ratio_without_acute": trim_var_new / full_var_new if full_var_new > 0 else np.nan,
    }

    with open(f"{logs_dir}/refined_stability_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_refined_stability_validation.csv")
    print("[ok] wrote:", f"{logs_dir}/refined_stability_validation_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation of refined S_v2 for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
