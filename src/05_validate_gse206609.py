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


def zscore(x: pd.Series) -> pd.Series:
    std = x.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def fit_poly_residuals(df: pd.DataFrame, degree: int, xcol: str = "PC1", ycol: str = "PC2") -> pd.Series:
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    coefs = np.polyfit(x, y, deg=degree)
    yhat = np.polyval(coefs, x)
    resid = y - yhat
    return pd.Series(zscore(pd.Series(resid)), index=df.index)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    interim_dir = cfg["paths"]["interim_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse206609_state_table_samples.csv")
    explicit_r = pd.read_csv(f"{processed_dir}/gse206609_explicit_R_condition_summary.csv")

    backbone_rows = []
    for degree, label in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        resid_z = fit_poly_residuals(samples, degree=degree)
        tmp = samples.copy()
        tmp["residual_fit_z"] = resid_z
        cond = (
            tmp.groupby(["condition", "phase", "treatment"], as_index=False)
            .agg(residual_mean=("residual_fit_z", "mean"),
                 residual_std=("residual_fit_z", "std"))
        )
        cond["backbone_model"] = label
        backbone_rows.append(cond)

    backbone = pd.concat(backbone_rows, ignore_index=True)
    backbone.to_csv(f"{processed_dir}/gse206609_backbone_sensitivity.csv", index=False)

    loo_rows = []
    for acc in samples["sample_accession"].unique():
        tmp = samples.loc[samples["sample_accession"] != acc].copy()
        cond = (
            tmp.groupby(["condition", "phase", "treatment"], as_index=False)
            .agg(
                H_mean=("H", "mean"),
                S_mean=("S", "mean"),
                R_mean=("R", "mean"),
                burden_mean=("stress_burden", "mean"),
            )
        )
        cond["left_out_sample"] = acc
        loo_rows.append(cond)

    loo = pd.concat(loo_rows, ignore_index=True)
    loo.to_csv(f"{processed_dir}/gse206609_leave_one_out_sensitivity.csv", index=False)

    no_anchor = samples.loc[samples["treatment"] != "none"].copy()
    no_anchor_cond = (
        no_anchor.groupby(["condition", "phase", "treatment"], as_index=False)
        .agg(
            H_mean=("H", "mean"),
            S_mean=("S", "mean"),
            R_mean=("R", "mean"),
            burden_mean=("stress_burden", "mean"),
        )
    )
    no_anchor_cond.to_csv(f"{processed_dir}/gse206609_without_control_anchor_summary.csv", index=False)

    # drug ranking summary excluding control
    drug_rank = explicit_r.loc[explicit_r["drug"] != "none"].copy()
    drug_rank = drug_rank.sort_values("explicit_R_observed", ascending=False)
    drug_rank["explicit_R_rank_desc"] = np.arange(1, len(drug_rank) + 1)
    drug_rank.to_csv(f"{processed_dir}/gse206609_explicit_R_drug_ranking.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "backbone_models_tested": ["linear", "quadratic", "cubic"],
        "explicit_R_rank_desc": drug_rank["drug"].tolist(),
        "note": "Validation includes backbone sensitivity, leave-one-replicate-out, and no-control-anchor summary.",
    }

    with open(f"{logs_dir}/validation_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_backbone_sensitivity.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_leave_one_out_sensitivity.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_without_control_anchor_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse206609_explicit_R_drug_ranking.csv")
    print("[ok] wrote:", f"{logs_dir}/validation_sensitivity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
