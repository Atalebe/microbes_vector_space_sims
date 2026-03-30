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


def mono_flag(vals: np.ndarray) -> bool:
    return bool(np.all(np.diff(vals) >= -1e-12))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])
    ensure_dirs(processed_root, logs_root)

    c = pd.read_csv(processed_root / "gse95575_refined_memory_conditions.csv")

    rows = []
    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min")
        for coord in ["M_mean", "M_v2", "M_v3"]:
            vals = sub[coord].to_numpy(dtype=float)
            rows.append(
                {
                    "treatment": treatment,
                    "coordinate": coord,
                    "is_monotone_non_decreasing": mono_flag(vals),
                    "min_step_change": float(np.min(np.diff(vals))) if len(vals) > 1 else np.nan,
                    "max_step_change": float(np.max(np.diff(vals))) if len(vals) > 1 else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out_path = processed_root / "gse95575_refined_memory_validation.csv"
    log_out = logs_root / "refined_memory_validation_summary.json"

    out.to_csv(out_path, index=False)

    summary = {
        "dataset_name": dataset_name,
        "monotonicity": {
            f"{row['treatment']}::{row['coordinate']}": bool(row["is_monotone_non_decreasing"])
            for _, row in out.iterrows()
        }
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
