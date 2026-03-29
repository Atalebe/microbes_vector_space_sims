from __future__ import annotations

import gzip
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


def minmax01(x: pd.Series) -> pd.Series:
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


def load_deseq(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    suppl_dir = cfg["paths"]["suppl_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse206609_state_table_samples.csv")
    conds = pd.read_csv(f"{processed_dir}/gse206609_state_table_conditions.csv")

    # condition-level treatment vs recovery relaxation
    conds["drug"] = conds["treatment"]

    control_t = conds.loc[conds["condition"] == "none_treatment"].iloc[0]
    control_r = conds.loc[conds["condition"] == "none_recovery"].iloc[0]

    treatment_rows = conds.loc[conds["phase"] == "treatment"].copy()
    recovery_rows = conds.loc[conds["phase"] == "recovery"].copy()

    merged = treatment_rows.merge(
        recovery_rows,
        on="drug",
        suffixes=("_treat", "_rec"),
        how="inner"
    )
    merged = merged.loc[merged["drug"] != "none"].copy()

    # geometry-based relaxation
    merged["burden_drop"] = (
        merged["dist_to_none_treatment_mean_treat"] - merged["dist_to_none_recovery_mean_rec"]
    )

    merged["relative_relaxation"] = merged["burden_drop"] / merged["dist_to_none_treatment_mean_treat"].replace(0, np.nan)

    # DESeq-based recovery burden
    deseq_map = {
        "levo": f"{suppl_dir}/GSE206609_Recovery_LEVO_deseq.txt.gz",
        "moxi": f"{suppl_dir}/GSE206609_Recovery_MOXI_deseq.txt.gz",
        "mmc":  f"{suppl_dir}/GSE206609_Recovery_MMC_deseq.txt.gz",
    }

    deseq_rows = []
    for drug, path in deseq_map.items():
        d = load_deseq(path)
        d.columns = [c.strip() for c in d.columns]

        sig = d.loc[d["padj"].notna() & (d["padj"] < 0.05)].copy()
        sig["abs_lfc"] = sig["log2FoldChange"].abs()

        deseq_rows.append(
            {
                "drug": drug,
                "n_sig_recovery_genes": int(len(sig)),
                "mean_abs_lfc_sig": float(sig["abs_lfc"].mean()) if len(sig) else 0.0,
                "sum_abs_lfc_sig": float(sig["abs_lfc"].sum()) if len(sig) else 0.0,
            }
        )

    deseq_summary = pd.DataFrame(deseq_rows)

    merged = merged.merge(deseq_summary, on="drug", how="left")

    # explicit recoverability score
    merged["geometry_recovery_score"] = minmax01(merged["relative_relaxation"].fillna(0.0))
    merged["transcriptional_residual_burden"] = minmax01(merged["sum_abs_lfc_sig"].fillna(0.0))
    merged["explicit_R_observed"] = (
        0.65 * merged["geometry_recovery_score"] +
        0.35 * (1.0 - merged["transcriptional_residual_burden"])
    )

    # match back against latent R
    latent_rec = recovery_rows[["drug", "R_mean", "burden_mean", "H_mean", "S_mean"]].copy()
    latent_rec = latent_rec.merge(
        merged[[
            "drug", "burden_drop", "relative_relaxation",
            "n_sig_recovery_genes", "mean_abs_lfc_sig", "sum_abs_lfc_sig",
            "geometry_recovery_score", "explicit_R_observed"
        ]],
        on="drug",
        how="left"
    )

    corr = latent_rec[["R_mean", "explicit_R_observed"]].corr(method="spearman").iloc[0, 1]

    # sample-level explicit R inherited from drug-level score for recovery samples
    recovery_samples = samples.loc[samples["phase"] == "recovery"].copy()
    recovery_samples["drug"] = recovery_samples["treatment"]
    recovery_samples = recovery_samples.merge(
        latent_rec[["drug", "explicit_R_observed"]],
        on="drug",
        how="left"
    )

    latent_rec.to_csv(f"{processed_dir}/gse206609_explicit_R_condition_summary.csv", index=False)
    recovery_samples.to_csv(f"{processed_dir}/gse206609_explicit_R_recovery_samples.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "drugs_evaluated": latent_rec["drug"].tolist(),
        "latent_vs_explicit_R_spearman": None if pd.isna(corr) else float(corr),
        "note": "Explicit R here is built from treatment-to-recovery relaxation and DESeq residual burden in recovery.",
    }

    with open(f"{logs_dir}/explicit_R_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_explicit_R_condition_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_explicit_R_recovery_samples.csv")
    print("[ok] wrote:", f"{logs_dir}/explicit_R_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
