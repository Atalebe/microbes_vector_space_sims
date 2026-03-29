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


def variance_packet(df: pd.DataFrame, col: str) -> dict:
    vals = df[col].dropna()
    total_var = float(vals.var(ddof=0))
    abs_centered = (vals - vals.mean()).abs()
    top_idx = abs_centered.idxmax()
    top_sample = df.loc[top_idx, "sample_accession"] if top_idx in df.index else None
    reduced = vals.drop(index=top_idx)
    reduced_var = float(reduced.var(ddof=0)) if len(reduced) > 1 else 0.0
    ratio = reduced_var / total_var if total_var > 0 else np.nan
    return {
        "packet": col,
        "total_variance": total_var,
        "top1_abs_contributor": top_sample,
        "variance_without_top1": reduced_var,
        "variance_ratio_without_top1": ratio,
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse206609_state_table_samples.csv")
    exp_samples = pd.read_csv(f"{processed_dir}/gse206609_explicit_R_recovery_samples.csv")

    merged = samples.merge(
        exp_samples[["sample_accession", "explicit_R_observed"]],
        on="sample_accession",
        how="left"
    )

    packets = []
    for col in ["R", "stress_burden", "H", "S"]:
        packets.append(variance_packet(merged, col))

    rec_only = merged.loc[merged["phase"] == "recovery"].copy()
    if "explicit_R_observed" in rec_only.columns:
        packets.append(variance_packet(rec_only, "explicit_R_observed"))

    packet_df = pd.DataFrame(packets)
    packet_df.to_csv(f"{processed_dir}/gse206609_variance_packet_summary.csv", index=False)

    # with and without MMC
    full_rows = []
    compare_cols = ["R", "stress_burden", "H", "S"]
    for col in compare_cols:
        full_var = float(merged[col].var(ddof=0))
        no_mmc_var = float(merged.loc[merged["treatment"] != "mmc", col].var(ddof=0))
        full_rows.append({
            "packet": col,
            "full_variance": full_var,
            "variance_without_mmc": no_mmc_var,
            "variance_ratio_without_mmc": no_mmc_var / full_var if full_var > 0 else np.nan
        })

    if "explicit_R_observed" in rec_only.columns:
        full_var = float(rec_only["explicit_R_observed"].dropna().var(ddof=0))
        no_mmc_var = float(rec_only.loc[rec_only["treatment"] != "mmc", "explicit_R_observed"].dropna().var(ddof=0))
        full_rows.append({
            "packet": "explicit_R_observed",
            "full_variance": full_var,
            "variance_without_mmc": no_mmc_var,
            "variance_ratio_without_mmc": no_mmc_var / full_var if full_var > 0 else np.nan
        })

    no_mmc = pd.DataFrame(full_rows)
    no_mmc.to_csv(f"{processed_dir}/gse206609_variance_without_mmc.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "frontier_removed_group": "mmc",
        "n_samples_full": int(len(merged)),
        "n_samples_without_mmc": int((merged["treatment"] != "mmc").sum()),
    }

    with open(f"{logs_dir}/variance_scaling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_variance_packet_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_variance_without_mmc.csv")
    print("[ok] wrote:", f"{logs_dir}/variance_scaling_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
