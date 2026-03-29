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
        curve = variance_contribution_curve(df[col])
        curve["packet"] = packet_name
        curves.append(curve)

        total_var = float(np.var(df[col], ddof=1))
        top_sid = df.iloc[np.argmax(np.abs(df[col].values))]["sample_accession"]
        trimmed = df.loc[df["sample_accession"] != top_sid, col]
        trimmed_var = float(np.var(trimmed, ddof=1)) if len(trimmed) > 1 else np.nan

        packet_summary.append(
            {
                "packet": packet_name,
                "total_variance": total_var,
                "top1_abs_contributor": top_sid,
                "variance_without_top1": trimmed_var,
                "variance_ratio_without_top1": trimmed_var / total_var if total_var > 0 else np.nan,
            }
        )

    curves_df = pd.concat(curves, ignore_index=True)
    packet_df = pd.DataFrame(packet_summary)

    # remove earliest stressed states: 0 and 15 min
    frontier = df.loc[df["time_min"].isin([0.0, 15.0])].copy()
    frontier_sids = frontier["sample_accession"].tolist()
    trimmed_df = df.loc[~df["sample_accession"].isin(frontier_sids)].copy()

    frontier_records = []
    for packet_name, col in packets.items():
        full_var = float(np.var(df[col], ddof=1))
        trim_var = float(np.var(trimmed_df[col], ddof=1)) if len(trimmed_df) > 1 else np.nan
        frontier_records.append(
            {
                "packet": packet_name,
                "full_variance": full_var,
                "variance_without_0_15min": trim_var,
                "variance_ratio_without_0_15min": trim_var / full_var if full_var > 0 else np.nan,
            }
        )
    frontier_df = pd.DataFrame(frontier_records)

    curves_df.to_csv(f"{processed_dir}/gse4370_variance_curves.csv", index=False)
    packet_df.to_csv(f"{processed_dir}/gse4370_variance_packet_summary.csv", index=False)
    frontier_df.to_csv(f"{processed_dir}/gse4370_variance_without_early_frontier.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "frontier_removed_samples": frontier_sids,
        "n_samples_full": int(len(df)),
        "n_samples_without_early_frontier": int(len(trimmed_df)),
    }

    with open(f"{logs_dir}/variance_scaling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_curves.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_packet_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_variance_without_early_frontier.csv")
    print("[ok] wrote:", f"{logs_dir}/variance_scaling_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Variance scaling diagnostics for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
