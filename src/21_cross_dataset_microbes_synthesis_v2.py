from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    outdir = "data/processed/cross_dataset_microbes"
    logdir = "results/logs/cross_dataset_microbes"
    ensure_dir(outdir)
    ensure_dir(logdir)

    g36599 = pd.read_csv("data/processed/gse36599/gse36599_refined_stability_validation.csv")
    g4370 = pd.read_csv("data/processed/gse4370/gse4370_phase_summary.csv")
    g206609 = pd.read_csv("data/processed/gse206609/gse206609_explicit_R_final_ranking.csv")

    rows = []

    rows.append({
        "dataset": "GSE36599",
        "branch_role": "staged benchmark stress-recovery branch",
        "organism": "Saccharomyces cerevisiae",
        "core_result": "staged stress-adaptation-recovery geometry with restorative bridge and burdened frontier",
        "explicit_R_status": "latent_to_bridge",
        "top_recovery_mode": "after_stress_recovery",
        "main_frontier": "acute stress adaptation"
    })

    rows.append({
        "dataset": "GSE4370",
        "branch_role": "continuous temporal recovery backbone",
        "organism": "Escherichia coli",
        "core_result": "continuous return from 0 to 90 min with hinge at 60 min and late coherent sector",
        "explicit_R_status": "temporal_bridge",
        "top_recovery_mode": "late recovery 75 to 90 min",
        "main_frontier": "0 to 15 min early recovery burden"
    })

    top_drug = g206609.sort_values("final_explicit_R_rank_desc").iloc[0]["drug"]
    worst_drug = g206609.sort_values("final_explicit_R_rank_desc").iloc[-1]["drug"]

    rows.append({
        "dataset": "GSE206609",
        "branch_role": "explicit treatment-recovery recoverability branch",
        "organism": "Escherichia coli",
        "core_result": "drug-resolved explicit recoverability with treatment-recovery and DE-burden coupling",
        "explicit_R_status": "explicit_observed",
        "top_recovery_mode": top_drug,
        "main_frontier": worst_drug
    })

    comp = pd.DataFrame(rows)
    comp.to_csv(f"{outdir}/microbes_dataset_comparison_v2.csv", index=False)

    frontier_rows = [
        {
            "dataset": "GSE36599",
            "frontier_group": "acute stress adaptation",
            "frontier_role": "benchmark burdened instability wing"
        },
        {
            "dataset": "GSE4370",
            "frontier_group": "0 to 15 min",
            "frontier_role": "early recovery burden wing"
        },
        {
            "dataset": "GSE206609",
            "frontier_group": worst_drug,
            "frontier_role": "drug-resolved explicit high-burden frontier"
        },
    ]
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(f"{outdir}/microbes_frontier_variance_comparison_v2.csv", index=False)

    summary = {
        "datasets_compared": comp["dataset"].tolist(),
        "explicit_R_upgrade_dataset": "GSE206609",
        "top_explicit_recovery_drug": top_drug,
        "worst_explicit_recovery_drug": worst_drug
    }

    with open(f"{logdir}/cross_dataset_summary_v2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbes_dataset_comparison_v2.csv")
    print("[ok] wrote:", f"{outdir}/microbes_frontier_variance_comparison_v2.csv")
    print("[ok] wrote:", f"{logdir}/cross_dataset_summary_v2.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
