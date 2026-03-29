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


def variance_contribution_curve(x: pd.Series, descending: bool = True) -> pd.DataFrame:
    vals = pd.to_numeric(x, errors="coerce").dropna().values
    if len(vals) == 0:
        return pd.DataFrame(columns=["k", "cum_sq_fraction"])
    order = np.argsort(np.abs(vals))[::-1] if descending else np.argsort(np.abs(vals))
    vals_ord = vals[order]
    total = np.sum(vals_ord ** 2)
    if total == 0:
        frac = np.zeros(len(vals_ord))
    else:
        frac = np.cumsum(vals_ord ** 2) / total
    return pd.DataFrame({"k": np.arange(1, len(vals_ord) + 1), "cum_sq_fraction": frac})


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    df = pd.read_csv(f"{processed_dir}/gse36599_residual_field_samples.csv")

    packets = {
        "residual_field": "residual_field",
        "stress_burden": "stress_burden",
        "recoverability_R": "R",
        "H": "H",
        "S": "S",
    }

    curve_frames = []
    packet_summary = []

    for packet_name, col in packets.items():
        curve = variance_contribution_curve(df[col])
        curve["packet"] = packet_name
        curve_frames.append(curve)

        total_var = float(np.var(df[col], ddof=1))
        top1_sid = df.iloc[np.argmax(np.abs(df[col].values))]["sample_accession"]

        trimmed = df.loc[df["sample_accession"] != top1_sid, col]
        trimmed_var = float(np.var(trimmed, ddof=1)) if len(trimmed) > 1 else np.nan

        packet_summary.append(
            {
                "packet": packet_name,
                "total_variance": total_var,
                "top1_abs_contributor": top1_sid,
                "variance_without_top1": trimmed_var,
                "variance_ratio_without_top1": trimmed_var / total_var if total_var > 0 else np.nan,
            }
        )

    curves = pd.concat(curve_frames, ignore_index=True)
    summary_df = pd.DataFrame(packet_summary)

    # Frontier removal: remove acute stress samples
    no_frontier = df.loc[df["sector"] != "acute_stress"].copy()
    frontier_records = []
    for packet_name, col in packets.items():
        full_var = float(np.var(df[col], ddof=1))
        trimmed_var = float(np.var(no_frontier[col], ddof=1)) if len(no_frontier) > 1 else np.nan
        frontier_records.append(
            {
                "packet": packet_name,
                "full_variance": full_var,
                "variance_without_acute_stress": trimmed_var,
                "variance_ratio_without_acute_stress": trimmed_var / full_var if full_var > 0 else np.nan,
            }
        )
    frontier_df = pd.DataFrame(frontier_records)

    # Leave-top2 acute removal
    acute_sids = (
        df.loc[df["sector"] == "acute_stress"]
        .sort_values("stress_burden", ascending=False)["sample_accession"]
        .tolist()[:2]
    )
    no_top2_acute = df.loc[~df["sample_accession"].isin(acute_sids)].copy()
    top2_records = []
    for packet_name, col in packets.items():
        full_var = float(np.var(df[col], ddof=1))
        trimmed_var = float(np.var(no_top2_acute[col], ddof=1)) if len(no_top2_acute) > 1 else np.nan
        top2_records.append(
            {
                "packet": packet_name,
                "full_variance": full_var,
                "variance_without_top2_acute": trimmed_var,
                "variance_ratio_without_top2_acute": trimmed_var / full_var if full_var > 0 else np.nan,
            }
        )
    top2_df = pd.DataFrame(top2_records)

    curves.to_csv(f"{processed_dir}/gse36599_variance_curves.csv", index=False)
    summary_df.to_csv(f"{processed_dir}/gse36599_variance_packet_summary.csv", index=False)
    frontier_df.to_csv(f"{processed_dir}/gse36599_variance_without_frontier.csv", index=False)
    top2_df.to_csv(f"{processed_dir}/gse36599_variance_without_top2_acute.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "acute_samples_removed_for_frontier_test": acute_sids,
        "n_samples_full": int(len(df)),
        "n_samples_without_frontier": int(len(no_frontier)),
        "n_samples_without_top2_acute": int(len(no_top2_acute)),
    }

    with open(f"{logs_dir}/variance_scaling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_variance_curves.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_variance_packet_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_variance_without_frontier.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_variance_without_top2_acute.csv")
    print("[ok] wrote:", f"{logs_dir}/variance_scaling_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Variance scaling diagnostics for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
