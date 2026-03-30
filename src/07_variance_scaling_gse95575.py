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


def packet_variance(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        v = float(np.nanvar(df[col].to_numpy(dtype=float), ddof=1))
        absmax_idx = df[col].abs().idxmax()
        rows.append(
            {
                "packet": col,
                "window": label,
                "total_variance": v,
                "top1_abs_contributor": str(df.loc[absmax_idx, "sample_accession"]),
                "variance_without_top1": float(
                    np.nanvar(df.drop(index=absmax_idx)[col].to_numpy(dtype=float), ddof=1)
                ) if len(df) > 2 else np.nan,
                "variance_ratio_without_top1": float(
                    np.nanvar(df.drop(index=absmax_idx)[col].to_numpy(dtype=float), ddof=1) / v
                ) if len(df) > 2 and v > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_drop(df: pd.DataFrame, cols: list[str], mask_keep: pd.Series, drop_name: str) -> pd.DataFrame:
    rows = []
    kept = df.loc[mask_keep].copy()
    for col in cols:
        full_var = float(np.nanvar(df[col].to_numpy(dtype=float), ddof=1))
        kept_var = float(np.nanvar(kept[col].to_numpy(dtype=float), ddof=1))
        rows.append(
            {
                "packet": col,
                "full_variance": full_var,
                f"variance_without_{drop_name}": kept_var,
                f"variance_ratio_without_{drop_name}": kept_var / full_var if full_var > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])
    ensure_dirs(processed_root, logs_root)

    df = pd.read_csv(processed_root / "gse95575_residual_field_samples.csv")

    packets = ["residual_field", "stress_burden", "R", "H", "S", "M"]

    packet_summary = packet_variance(df, packets, "full")
    without_pma = compare_drop(df, packets, df["treatment"] != "pma", "pma")
    without_hg = compare_drop(df, packets, df["treatment"] != "hgcl2", "hgcl2")

    packet_out = processed_root / "gse95575_variance_packet_summary.csv"
    pma_out = processed_root / "gse95575_variance_without_pma.csv"
    hg_out = processed_root / "gse95575_variance_without_hgcl2.csv"
    log_out = logs_root / "variance_scaling_summary.json"

    packet_summary.to_csv(packet_out, index=False)
    without_pma.to_csv(pma_out, index=False)
    without_hg.to_csv(hg_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "frontier_removed_groups": ["pma", "hgcl2"],
        "n_samples_full": int(len(df)),
        "n_samples_without_pma": int((df["treatment"] != "pma").sum()),
        "n_samples_without_hgcl2": int((df["treatment"] != "hgcl2").sum()),
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {packet_out}")
    print(f"[ok] wrote: {pma_out}")
    print(f"[ok] wrote: {hg_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
