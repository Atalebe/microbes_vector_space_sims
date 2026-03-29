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


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - np.nanmean(x)) / std
    return 0.67448975 * (x - med) / mad


def fit_backbone_and_residuals(df: pd.DataFrame, degree: int) -> pd.DataFrame:
    out = df.copy()
    coeffs = np.polyfit(out["PC1"].values, out["PC2"].values, deg=degree)
    out["pc2_fit"] = np.polyval(coeffs, out["PC1"].values)
    out["residual_pc2_fit"] = out["PC2"] - out["pc2_fit"]
    out["residual_pc2_fit_z"] = robust_z(out["residual_pc2_fit"])
    return out


def rank_condition_means(df: pd.DataFrame, value_col: str) -> list[str]:
    s = (
        df.groupby(["condition", "condition_order"], as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
    )
    return s["condition"].tolist()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    samples = pd.read_csv(f"{processed_dir}/gse36599_residual_field_samples.csv")

    # -----------------------------
    # 1. Backbone-model sensitivity
    # -----------------------------
    backbone_results = []
    backbone_tables = []

    for degree, label in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        tmp = fit_backbone_and_residuals(samples, degree)
        cond = (
            tmp.groupby(["condition", "condition_order"], as_index=False)
            .agg(
                residual_mean=("residual_pc2_fit_z", "mean"),
                residual_std=("residual_pc2_fit_z", "std"),
            )
            .sort_values("condition_order")
        )
        cond["backbone_model"] = label
        backbone_tables.append(cond)

        backbone_results.append(
            {
                "backbone_model": label,
                "condition_ranking_desc": rank_condition_means(tmp, "residual_pc2_fit_z"),
                "acute_mean": float(
                    cond.loc[cond["condition"] == "during_stress_adaptation", "residual_mean"].iloc[0]
                ),
                "baseline_mean": float(
                    cond.loc[cond["condition"] == "normal_growth", "residual_mean"].iloc[0]
                ),
                "after_recovery_mean": float(
                    cond.loc[cond["condition"] == "after_stress_recovery", "residual_mean"].iloc[0]
                ),
            }
        )

    backbone_df = pd.concat(backbone_tables, ignore_index=True)
    backbone_df.to_csv(f"{processed_dir}/gse36599_backbone_sensitivity.csv", index=False)

    # -----------------------------
    # 2. Leave-one-sample-out
    # -----------------------------
    loo_records = []
    sample_ids = samples["sample_accession"].tolist()

    for sid in sample_ids:
        subset = samples.loc[samples["sample_accession"] != sid].copy()
        tmp = fit_backbone_and_residuals(subset, 2)

        cond_means = (
            tmp.groupby(["condition", "condition_order"], as_index=False)
            .agg(residual_mean=("residual_pc2_fit_z", "mean"))
            .sort_values("condition_order")
        )

        loo_records.append(
            {
                "left_out_sample": sid,
                "acute_mean": float(
                    cond_means.loc[cond_means["condition"] == "during_stress_adaptation", "residual_mean"].iloc[0]
                ),
                "baseline_mean": float(
                    cond_means.loc[cond_means["condition"] == "normal_growth", "residual_mean"].iloc[0]
                ),
                "after_recovery_mean": float(
                    cond_means.loc[cond_means["condition"] == "after_stress_recovery", "residual_mean"].iloc[0]
                ),
                "condition_ranking_desc": " > ".join(rank_condition_means(tmp, "residual_pc2_fit_z")),
            }
        )

    loo_df = pd.DataFrame(loo_records)
    loo_df.to_csv(f"{processed_dir}/gse36599_leave_one_out_sensitivity.csv", index=False)

    # -----------------------------
    # 3. Remove strongest acute outlier
    # -----------------------------
    acute_only = samples.loc[samples["condition"] == "during_stress_adaptation"].copy()
    worst_sid = acute_only.sort_values("residual_field").iloc[0]["sample_accession"]

    outlier_removed = samples.loc[samples["sample_accession"] != worst_sid].copy()
    outlier_tmp = fit_backbone_and_residuals(outlier_removed, 2)

    outlier_cond = (
        outlier_tmp.groupby(["condition", "condition_order"], as_index=False)
        .agg(
            residual_mean=("residual_pc2_fit_z", "mean"),
            burden_mean=("stress_burden", "mean"),
            R_mean=("R", "mean"),
        )
        .sort_values("condition_order")
    )
    outlier_cond.to_csv(f"{processed_dir}/gse36599_outlier_removed_condition_summary.csv", index=False)

    # -----------------------------
    # 4. Condition-order validation
    # -----------------------------
    burden_rank = rank_condition_means(samples, "stress_burden")
    recover_rank = rank_condition_means(samples, "R")
    H_rank = rank_condition_means(samples, "H")
    S_rank = rank_condition_means(samples, "S")

    summary = {
        "dataset_name": cfg["dataset_name"],
        "worst_acute_outlier_removed": worst_sid,
        "backbone_models_tested": [r["backbone_model"] for r in backbone_results],
        "backbone_results": backbone_results,
        "burden_rank_desc": burden_rank,
        "recoverability_rank_desc": recover_rank,
        "H_rank_desc": H_rank,
        "S_rank_desc": S_rank,
    }

    with open(f"{logs_dir}/validation_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse36599_backbone_sensitivity.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_leave_one_out_sensitivity.csv")
    print("[ok] wrote:", f"{processed_dir}/gse36599_outlier_removed_condition_summary.csv")
    print("[ok] wrote:", f"{logs_dir}/validation_sensitivity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation and sensitivity checks for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
