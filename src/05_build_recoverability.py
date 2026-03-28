#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from utils import load_config, read_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    residual = read_table(Path(cfg["io"]["results_dir"]) / "tables" / "residual_field.csv")
    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_tbl.mkdir(parents=True, exist_ok=True)

    records = []
    if "R_explicit" in residual.columns:
        pair = residual[["R_latent", "R_explicit"]].dropna()
        if len(pair) >= 3:
            rho, pval = spearmanr(pair["R_latent"], pair["R_explicit"])
            records.append({
                "metric": "spearman_rho_latent_vs_explicit_R",
                "value": float(rho),
            })
            records.append({
                "metric": "spearman_pvalue_latent_vs_explicit_R",
                "value": float(pval),
            })
            pair.to_csv(out_tbl / "recoverability_alignment_pairs.csv", index=False)
        else:
            records.append({"metric": "spearman_rho_latent_vs_explicit_R", "value": None})
    else:
        records.append({"metric": "explicit_R_available", "value": 0})

    pd.DataFrame(records).to_csv(out_tbl / "recoverability_summary.csv", index=False)
    print("[ok] wrote recoverability summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
