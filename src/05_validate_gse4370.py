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


def fit_backbone(df: pd.DataFrame, degree: int) -> pd.DataFrame:
    out = df.copy()
    coeffs = np.polyfit(out["PC1"].values, out["PC2"].values, deg=degree)
    out["pc2_fit"] = np.polyval(coeffs, out["PC1"].values)
    out["residual_fit"] = out["PC2"] - out["pc2_fit"]
    out["residual_fit_z"] = robust_z(out["residual_fit"])
    return out


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    df = pd.read_csv(f"{processed_dir}/gse4370_residual_field_samples.csv").sort_values("time_order").reset_index(drop=True)

    model_tables = []
    model_results = []

    for degree, label in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        tmp = fit_backbone(df, degree)
        tmp["backbone_model"] = label
        model_tables.append(tmp[["sample_accession", "condition", "time_min", "backbone_model", "residual_fit_z"]])

        rank_desc = (
            tmp.sort_values("residual_fit_z", ascending=False)["condition"].tolist()
        )
        model_results.append(
            {
                "backbone_model": label,
                "rank_desc": rank_desc,
                "top_condition": rank_desc[0],
                "bottom_condition": rank_desc[-1],
            }
        )

    model_df = pd.concat(model_tables, ignore_index=True)
    model_df.to_csv(f"{processed_dir}/gse4370_backbone_sensitivity.csv", index=False)

    # leave-one-timepoint-out
    loo_records = []
    for sid in df["sample_accession"]:
        sub = df.loc[df["sample_accession"] != sid].copy()
        tmp = fit_backbone(sub, 2)
        rank_desc = tmp.sort_values("residual_fit_z", ascending=False)["condition"].tolist()
        loo_records.append(
            {
                "left_out_sample": sid,
                "top_condition": rank_desc[0],
                "bottom_condition": rank_desc[-1],
                "rank_desc": " > ".join(rank_desc),
            }
        )

    loo_df = pd.DataFrame(loo_records)
    loo_df.to_csv(f"{processed_dir}/gse4370_leave_one_out_sensitivity.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "backbone_models_tested": [r["backbone_model"] for r in model_results],
        "model_results": model_results,
        "R_rank_desc": df.sort_values("R", ascending=False)["condition"].tolist(),
        "burden_rank_desc": df.sort_values("stress_burden", ascending=False)["condition"].tolist(),
    }

    with open(f"{logs_dir}/validation_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_backbone_sensitivity.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_leave_one_out_sensitivity.csv")
    print("[ok] wrote:", f"{logs_dir}/validation_sensitivity_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validation and sensitivity for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
