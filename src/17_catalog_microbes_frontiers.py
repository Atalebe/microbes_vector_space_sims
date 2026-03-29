from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    outdir = "data/processed/microbial_closure"
    logdir = "results/logs/microbial_closure"
    ensure_dir(outdir)
    ensure_dir(logdir)

    rows = []

    # GSE36599 frontier logic
    g1 = pd.read_csv("data/processed/gse36599/gse36599_refined_stability_samples.csv")
    for _, r in g1.iterrows():
        mode = None
        rationale = None

        if r["condition"] == "during_stress_adaptation":
            mode = "acute_instability_frontier"
            rationale = "high burden and low recoverability under acute stress"
        elif r["condition"] == "after_stress_adaptation" and r["residual_field_v2"] < 0:
            mode = "burdened_transition_case"
            rationale = "post-adaptation but still negative in residual field"
        elif r["condition"] == "during_stress_recovery" and r["R"] > 0:
            mode = "restorative_bridge_case"
            rationale = "actively returning toward recovered side"
        elif r["condition"] == "after_stress_recovery" and r["H"] > 0:
            mode = "coherent_recovered_case"
            rationale = "recovered-side sample with positive depth"
        elif r["condition"] == "normal_growth":
            mode = "baseline_coherent_case"
            rationale = "reference coherent state"

        if mode is not None:
            rows.append(
                {
                    "dataset": "GSE36599",
                    "sample_accession": r["sample_accession"],
                    "condition": r["condition"],
                    "frontier_mode": mode,
                    "H": r["H"],
                    "S": r["S_v2"],
                    "M": r["M"],
                    "R": r["R"],
                    "stress_burden": r["stress_burden"],
                    "rationale": rationale,
                }
            )

    # GSE4370 frontier logic
    g2 = pd.read_csv("data/processed/gse4370/gse4370_residual_field_samples.csv")
    for _, r in g2.iterrows():
        mode = None
        rationale = None

        if r["time_min"] in [0.0, 15.0]:
            mode = "early_recovery_frontier"
            rationale = "high burden and negative recoverability on disrupted side"
        elif r["time_min"] == 60.0:
            mode = "hinge_transition_case"
            rationale = "bridging state between mid and late recovery with local pivot"
        elif r["time_min"] in [75.0, 90.0]:
            mode = "late_recovery_coherent_case"
            rationale = "late recovery state with high recoverability and low burden"
        elif r["time_min"] in [45.0]:
            mode = "mid_recovery_bridge_case"
            rationale = "transitional corridor sample"
        elif r["time_min"] in [30.0]:
            mode = "early_to_mid_transition_case"
            rationale = "leaving early burdened side"

        if mode is not None:
            rows.append(
                {
                    "dataset": "GSE4370",
                    "sample_accession": r["sample_accession"],
                    "condition": r["condition"],
                    "frontier_mode": mode,
                    "H": r["H"],
                    "S": r["S"],
                    "M": r["M"],
                    "R": r["R"],
                    "stress_burden": r["stress_burden"],
                    "rationale": rationale,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/microbial_frontier_catalog.csv", index=False)

    summary = {
        "datasets": sorted(df["dataset"].unique().tolist()),
        "frontier_mode_counts": df["frontier_mode"].value_counts().to_dict(),
    }

    with open(f"{logdir}/frontier_catalog_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_frontier_catalog.csv")
    print("[ok] wrote:", f"{logdir}/frontier_catalog_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
