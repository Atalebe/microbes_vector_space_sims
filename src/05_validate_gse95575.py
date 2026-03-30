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
    interim_root = Path(cfg.get("interim_root", "data/interim/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))

    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("processed_root", str(processed_root))
    cfg.setdefault("interim_root", str(interim_root))
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


def residual_fit_scores(df: pd.DataFrame, degree: int) -> pd.DataFrame:
    x = df["PC1"].to_numpy(dtype=float)
    y = df["PC2"].to_numpy(dtype=float)
    coeffs = np.polyfit(x, y, deg=degree)
    yhat = np.polyval(coeffs, x)
    out = df[["sample_accession", "condition", "treatment", "time_min", "replicate"]].copy()
    out["backbone_model"] = {1: "linear", 2: "quadratic", 3: "cubic"}[degree]
    out["residual_fit"] = y - yhat
    out["residual_fit_z"] = robust_z(out["residual_fit"])
    return out


def monotonicity_summary(cond_df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for treatment, sub in cond_df.groupby("treatment"):
        sub = sub.sort_values("time_min")
        vals = sub[col].to_numpy(dtype=float)
        diffs = np.diff(vals)
        rows.append(
            {
                "treatment": treatment,
                "coordinate": col,
                "n_steps": len(diffs),
                "n_non_decreasing": int(np.sum(diffs >= 0)),
                "n_strict_decreases": int(np.sum(diffs < 0)),
                "min_step_change": float(np.min(diffs)) if len(diffs) else np.nan,
                "max_step_change": float(np.max(diffs)) if len(diffs) else np.nan,
                "is_monotone_non_decreasing": bool(np.all(diffs >= 0)),
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(processed_root, logs_root)

    sample_df = pd.read_csv(processed_root / "gse95575_state_table_samples.csv")
    cond_df = pd.read_csv(processed_root / "gse95575_state_table_conditions.csv")

    backbone_parts = []
    for degree in [1, 2, 3]:
        scores = residual_fit_scores(sample_df, degree)
        cond_rank = (
            scores.groupby("condition")["residual_fit_z"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        scores["condition_rank_desc"] = ",".join(cond_rank)
        backbone_parts.append(scores)
    backbone = pd.concat(backbone_parts, ignore_index=True)

    loo_rows = []
    reps = sorted(sample_df["sample_accession"].astype(str).unique().tolist())
    for removed in reps:
        sub = sample_df[sample_df["sample_accession"] != removed].copy()
        x = sub["PC1"].to_numpy(dtype=float)
        y = sub["PC2"].to_numpy(dtype=float)
        coeffs = np.polyfit(x, y, deg=2)
        yhat = np.polyval(coeffs, x)
        sub["residual_fit_z"] = robust_z(pd.Series(y - yhat))
        cond_means = (
            sub.groupby("condition")["residual_fit_z"]
            .mean()
            .sort_values(ascending=False)
        )
        loo_rows.append(
            {
                "removed_sample": removed,
                "top_condition": cond_means.index[0],
                "bottom_condition": cond_means.index[-1],
                "pma_t30_mean": float(cond_means.get("pma_t30", np.nan)),
                "hgcl2_t30_mean": float(cond_means.get("hgcl2_t30", np.nan)),
                "control_t60_mean": float(cond_means.get("control_t60", np.nan)),
            }
        )
    loo = pd.DataFrame(loo_rows)

    mono_M = monotonicity_summary(cond_df, "M_mean")
    mono_R = monotonicity_summary(cond_df, "R_mean")
    memory_monotonicity = pd.concat([mono_M, mono_R], ignore_index=True)

    drug_condition_ranking = (
        cond_df.groupby("treatment")[["H_mean", "S_mean", "M_mean", "R_mean", "burden_mean"]]
        .mean()
        .reset_index()
        .sort_values("R_mean", ascending=False)
    )
    drug_condition_ranking["R_rank_desc"] = range(1, len(drug_condition_ranking) + 1)

    backbone_out = processed_root / "gse95575_backbone_sensitivity.csv"
    loo_out = processed_root / "gse95575_leave_one_out_sensitivity.csv"
    mono_out = processed_root / "gse95575_memory_monotonicity.csv"
    rank_out = processed_root / "gse95575_treatment_ranking_validation.csv"
    log_out = logs_root / "validation_sensitivity_summary.json"

    backbone.to_csv(backbone_out, index=False)
    loo.to_csv(loo_out, index=False)
    memory_monotonicity.to_csv(mono_out, index=False)
    drug_condition_ranking.to_csv(rank_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "backbone_models_tested": ["linear", "quadratic", "cubic"],
        "worst_loo_pma_t30_mean": float(loo["pma_t30_mean"].min()),
        "best_loo_pma_t30_mean": float(loo["pma_t30_mean"].max()),
        "M_monotone_by_treatment": {
            row["treatment"]: bool(row["is_monotone_non_decreasing"])
            for _, row in mono_M.iterrows()
        },
        "R_monotone_by_treatment": {
            row["treatment"]: bool(row["is_monotone_non_decreasing"])
            for _, row in mono_R.iterrows()
        },
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {backbone_out}")
    print(f"[ok] wrote: {loo_out}")
    print(f"[ok] wrote: {mono_out}")
    print(f"[ok] wrote: {rank_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
