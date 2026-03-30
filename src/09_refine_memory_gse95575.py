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


def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return (x - med) / std
    return 0.67448975 * (x - med) / mad


def cumulative_progress(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    d = [0.0]
    for i in range(1, len(points)):
        d.append(d[-1] + float(np.linalg.norm(points[i] - points[i - 1])))
    d = np.array(d, dtype=float)
    if d[-1] > 0:
        d = d / d[-1]
    return d


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])
    ensure_dirs(processed_root, logs_root)

    s = pd.read_csv(processed_root / "gse95575_state_table_samples.csv")
    c = pd.read_csv(processed_root / "gse95575_state_table_conditions.csv")

    refined_condition_rows = []
    treatment_progress_maps = {}

    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min").copy()
        pts = sub[["H_mean", "S_mean", "R_mean"]].to_numpy(dtype=float)
        prog = cumulative_progress(pts)
        sub["M_v2_raw"] = prog
        treatment_progress_maps[treatment] = dict(zip(sub["condition"], sub["M_v2_raw"]))
        refined_condition_rows.append(sub)

    c2 = pd.concat(refined_condition_rows, ignore_index=True)

    # Relative memory vs matched control progress
    control_prog = (
        c2[c2["treatment"] == "control"][["time_min", "M_v2_raw"]]
        .drop_duplicates()
        .set_index("time_min")["M_v2_raw"]
        .to_dict()
    )
    c2["M_v3_raw"] = c2.apply(
        lambda row: float(row["M_v2_raw"] - control_prog.get(row["time_min"], 0.0)),
        axis=1,
    )

    c2["M_v2"] = robust_z(c2["M_v2_raw"])
    c2["M_v3"] = robust_z(c2["M_v3_raw"])

    s["M_v2_raw"] = s["condition"].map({k: v for t in treatment_progress_maps.values() for k, v in t.items()})
    s["M_v3_raw"] = s.apply(
        lambda row: float(row["M_v2_raw"] - control_prog.get(row["time_min"], 0.0)),
        axis=1,
    )
    s["M_v2"] = robust_z(s["M_v2_raw"])
    s["M_v3"] = robust_z(s["M_v3_raw"])

    validation = (
        c2.groupby("treatment")[["M_mean", "M_v2", "M_v3", "R_mean", "burden_mean"]]
        .mean()
        .reset_index()
    )

    samples_out = processed_root / "gse95575_refined_memory_samples.csv"
    cond_out = processed_root / "gse95575_refined_memory_conditions.csv"
    val_out = processed_root / "gse95575_memory_validation.csv"
    log_out = logs_root / "refined_memory_summary.json"

    s.to_csv(samples_out, index=False)
    c2.to_csv(cond_out, index=False)
    validation.to_csv(val_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "memory_versions": ["M_mean", "M_v2", "M_v3"],
        "notes": {
            "M_v2": "Monotone centroid-path progress within each treatment branch.",
            "M_v3": "Relative treatment memory progress after subtracting matched control progress.",
        },
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {samples_out}")
    print(f"[ok] wrote: {cond_out}")
    print(f"[ok] wrote: {val_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
