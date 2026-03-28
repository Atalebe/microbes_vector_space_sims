#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils import dump_json, load_config, read_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    raw_path = cfg["io"]["raw_input_path"]
    out_dir = Path(cfg["io"]["results_dir"]) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(raw_path)

    missing = pd.DataFrame({
        "column": df.columns,
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
        "pct_missing": [float(df[c].isna().mean() * 100.0) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    missing.to_csv(out_dir / "missingness_report.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "time_column": cfg.get("columns", {}).get("time"),
        "condition_column": cfg.get("columns", {}).get("condition"),
        "strain_column": cfg.get("columns", {}).get("strain"),
    }

    for key in ["time", "condition", "strain"]:
        col = cfg.get("columns", {}).get(key)
        if col and col in df.columns:
            summary[f"{key}_n_unique"] = int(df[col].nunique(dropna=True))

    dump_json(summary, out_dir / "dataset_summary.json")

    preview_cols = list(df.columns[: min(30, len(df.columns))])
    pd.DataFrame({"column": preview_cols}).to_csv(out_dir / "column_inventory.csv", index=False)

    print(f"[ok] audited {raw_path}")
    print(f"[ok] wrote {out_dir / 'missingness_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
