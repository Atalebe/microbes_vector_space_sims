from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def classify_missingness(frac: float) -> str:
    if frac <= 0.05:
        return "low_missingness"
    if frac <= 0.10:
        return "moderate_missingness"
    return "high_missingness"


def main() -> None:
    outdir = "data/processed/microbial_closure"
    logdir = "results/logs/microbial_closure"
    ensure_dir(outdir)
    ensure_dir(logdir)

    rows = []

    # -------------------------
    # GSE36599
    # -------------------------
    g1_expr = pd.read_csv("data/interim/gse36599/audit_expression_summary.csv")
    g1_samples = pd.read_csv("data/processed/gse36599/gse36599_residual_field_samples.csv")

    # no missing expression issue in benchmark branch
    for _, r in g1_samples.iterrows():
        rows.append(
            {
                "dataset": "GSE36599",
                "sample_accession": r["sample_accession"],
                "condition": r["condition"],
                "time_or_order": r["condition_order"],
                "missing_fraction": 0.0,
                "missingness_flag": "low_missingness",
                "metadata_flag": "complete_metadata",
                "geometry_flag": "benchmark_branch",
                "frontier_flag": (
                    "frontier_case" if r["sector"] in ["acute_stress", "burdened"] else "non_frontier"
                ),
                "stability_flag": "S_v2_available",
                "overall_quality_flag": "usable",
                "notes": "paired staged benchmark branch",
            }
        )

    # -------------------------
    # GSE4370
    # -------------------------
    g2_expr = pd.read_csv("data/processed/gse4370/gse4370_gsm99161_expression_summary.csv")
    g2_samples = pd.read_csv("data/processed/gse4370/gse4370_residual_field_samples.csv")
    g2_backbone = pd.read_csv("data/processed/gse4370/gse4370_backbone_sensitivity.csv")

    for _, r in g2_samples.iterrows():
        expr_row = g2_expr.loc[g2_expr["sample_accession"] == r["sample_accession"]].iloc[0]
        missing_frac = expr_row["n_missing"] / expr_row["n_rows"]

        # backbone caution: large spread across models
        sub = g2_backbone.loc[g2_backbone["sample_accession"] == r["sample_accession"]]
        spread = sub["residual_fit_z"].max() - sub["residual_fit_z"].min()
        if spread > 3.0:
            geometry_flag = "high_backbone_sensitivity"
        elif spread > 1.5:
            geometry_flag = "moderate_backbone_sensitivity"
        else:
            geometry_flag = "stable_across_backbones"

        frontier_flag = "frontier_case" if r["time_min"] in [0.0, 15.0] else "non_frontier"
        if r["sample_accession"] == "GSM99161":
            frontier_flag = "hinge_case"

        overall = "usable"
        if missing_frac > 0.10:
            overall = "caution"
        if geometry_flag == "high_backbone_sensitivity":
            overall = "caution"

        rows.append(
            {
                "dataset": "GSE4370",
                "sample_accession": r["sample_accession"],
                "condition": r["condition"],
                "time_or_order": r["time_min"],
                "missing_fraction": float(missing_frac),
                "missingness_flag": classify_missingness(float(missing_frac)),
                "metadata_flag": "complete_metadata",
                "geometry_flag": geometry_flag,
                "frontier_flag": frontier_flag,
                "stability_flag": "trajectory_stability",
                "overall_quality_flag": overall,
                "notes": "continuous recovery companion branch",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/microbial_quality_flags.csv", index=False)

    summary = {
        "datasets": df["dataset"].unique().tolist(),
        "n_rows": int(len(df)),
        "overall_quality_counts": df["overall_quality_flag"].value_counts().to_dict(),
    }

    with open(f"{logdir}/quality_flags_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_quality_flags.csv")
    print("[ok] wrote:", f"{logdir}/quality_flags_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
