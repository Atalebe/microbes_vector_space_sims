#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils import load_config, read_table


TARGETS = ["R_latent", "R_explicit", "H", "S", "M"]


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(Path(cfg["io"]["results_dir"]) / "tables" / "residual_field.csv")
    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_tbl.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s) < 3:
            continue
        q95 = s.quantile(0.95)
        clipped = s[s <= q95]
        rows.append({
            "target": target,
            "n": int(len(s)),
            "var_full": float(s.var(ddof=1)),
            "var_without_top_5pct": float(clipped.var(ddof=1)) if len(clipped) >= 3 else None,
            "mean_full": float(s.mean()),
            "mean_without_top_5pct": float(clipped.mean()) if len(clipped) >= 3 else None,
        })

    pd.DataFrame(rows).to_csv(out_tbl / "variance_scaling_summary.csv", index=False)
    print("[ok] wrote variance scaling summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
