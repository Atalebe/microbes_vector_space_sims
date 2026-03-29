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


def variance_contribution_curve(x: pd.Series) -> pd.DataFrame:
    vals = pd.to_numeric(x, errors="coerce").dropna().values
    if len(vals) == 0:
        return pd.DataFrame(columns=["k", "cum_sq_fraction"])
    order = np.argsort(np.abs(vals))[::-1]
    vals_ord = vals[order]
    total = np.sum(vals_ord ** 2)
    frac = np.cumsum(vals_ord ** 2) / total if total > 0 else np.zeros(len(vals_ord))
    return pd.DataFrame({"k": np.arange(1, len(vals_ord) + 1), "cum_sq_fraction": frac})


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    df = pd.read_csv(f"{processed_dir}/gse4370_residual_field_samples.csv").sort_values("time_order").reset_index(drop=True)

    # restrict to 45, 60, 75 min
    window = df.loc[df["time_min"].isin([45.0, 60.0, 75.0])].copy().reset_index(drop=True)

    packets = {
        "residual_field": "residual_field",
        "stress_burden": "stress_burden",
        "recoverability_R": "R",
        "H": "H",
        "S": "S",
    }

    curves = []
    packet_summary = []

    for packet_name, col in packets.items():
        curve = variance_contribution_curve(window[col])
        curve["packet"] = packet_name
        curves.append(curve)

        total_var = float(np.var(window[col], ddof=1)) if len(window) > 1 else np.nan
        top_idx = np.argmax(np.abs(window[col].values))
        top_sid = window.iloc[top_idx]["sample_accession"]

        trimmed = window.loc[window["sample_accession"] != top_sid, col]
        trimmed_var = float(np.var(trimmed, ddof=1)) if len(trimmed) > 1 else np.nan

        packet_summary.append(
            {
                "packet": packet_name,
                "window": "45_60_75",
                "total_variance": total_var,
                "top1_abs_contributor": top_sid,
                "variance_without_top1": trimmed_var,
                "variance_ratio_without_top1": trimmed_var / total_var if total_var and total_var > 0 else np.nan,
            }
        )

    curves_df = pd.concat(curves, ignore_index=True)
    packet_df = pd.DataFrame(packet_summary)

    # specifically remove 60 min and compare
    center_sid = window.loc[window["time_min"] == 60.0, "sample_accession"].iloc[0]
    no_60 = window.loc[window["sample_accession"] != center_sid].copy()

    center_records = []
    for packet_name, col in packets.items():
        full_var = float(np.var(window[col], ddof=1)) if len(window) > 1 else np.nan
        trim_var = float(np.var(no_60[col], ddof=1)) if len(no_60) > 1 else np.nan
        center_records.append(
            {
                "packet": packet_name,
                "full_variance": full_var,
                "variance_without_60min": trim_var,
                "variance_ratio_without_60min": trim_var / full_var if full_var and full_var > 0 else np.nan,
            }
        )
    center_df = pd.DataFrame(center_records)

    # save
    curves_df.to_csv(f"{processed_dir}/gse4370_variance_curves_45_75.csv", index=False)
    packet_df.to_csv(f"{processed_dir}/gse4370_variance_packet_summary_45_75.csv", index=False)
    center_df.to_csv(f"{processed_dir}/gse4370_variance_without_60min_45_75.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "window_timepoints": [45.0, 60.0, 75.0],
        "samples_in_window": window["sample_accession"].tolist(),
        "sample_removed_for_center_test": center_sid,
    }

    with open(f"{logs_dir}/variance_scaling_45_75_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_curves_45_75.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_packet_summary_45_75.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_without_60min_45_75.csv")
    print("[ok] wrote:", f"{logs_dir}/variance_scaling_45_75_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Variance scaling for the 45–75 min window in GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
