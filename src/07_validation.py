#!/usr/bin/env python3
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from utils import get_col, load_config, read_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(Path(cfg["io"]["results_dir"]) / "tables" / "residual_field.csv")
    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_tbl.mkdir(parents=True, exist_ok=True)

    results = []
    if "R_explicit" in df.columns:
        pair = df[["R_latent", "R_explicit"]].dropna()
        if len(pair) >= 3:
            rho, pval = spearmanr(pair["R_latent"], pair["R_explicit"])
            results.append({"test": "global_latent_explicit_alignment", "stat": float(rho), "pvalue": float(pval), "n": int(len(pair))})

    for key in ["condition", "strain"]:
        col = get_col(cfg, key)
        if col and col in df.columns:
            counts = df[col].value_counts(dropna=True)
            for level, n in counts.items():
                if n < 5:
                    continue
                sub = df.loc[df[col] == level]
                if "R_explicit" in sub.columns:
                    pair = sub[["R_latent", "R_explicit"]].dropna()
                    if len(pair) >= 3:
                        rho, pval = spearmanr(pair["R_latent"], pair["R_explicit"])
                        results.append({"test": f"subset_alignment_{key}", "level": level, "stat": float(rho), "pvalue": float(pval), "n": int(len(pair))})

    pd.DataFrame(results).to_csv(out_tbl / "validation_summary.csv", index=False)
    print("[ok] wrote validation summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
