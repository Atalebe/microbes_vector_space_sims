#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    out_dir = Path("data/processed/cross_dataset_microbes")
    log_dir = Path("results/logs/cross_dataset_microbes")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = [
        {
            "dataset": "GSE36599",
            "branch_role": "staged benchmark stress-recovery branch",
            "organism": "Saccharomyces cerevisiae",
            "core_result": "staged stress-adaptation-recovery geometry with restorative bridge and burdened frontier",
            "memory_status": "ordinal stage memory",
            "explicit_R_status": "latent_to_bridge",
            "top_recovery_mode": "after_stress_recovery",
            "main_frontier": "acute stress adaptation",
        },
        {
            "dataset": "GSE4370",
            "branch_role": "continuous temporal recovery backbone",
            "organism": "Escherichia coli",
            "core_result": "continuous return from 0 to 90 min with hinge at 60 min and late coherent sector",
            "memory_status": "normalized time memory",
            "explicit_R_status": "temporal_bridge",
            "top_recovery_mode": "late recovery 75 to 90 min",
            "main_frontier": "0 to 15 min early recovery burden",
        },
        {
            "dataset": "GSE206609",
            "branch_role": "explicit treatment-recovery recoverability branch",
            "organism": "Escherichia coli",
            "core_result": "drug-resolved explicit recoverability with treatment-recovery and DE-burden coupling",
            "memory_status": "binary phase memory",
            "explicit_R_status": "explicit_observed",
            "top_recovery_mode": "levo",
            "main_frontier": "mmc",
        },
        {
            "dataset": "GSE95575",
            "branch_role": "conditioning-memory upgrade branch",
            "organism": "Escherichia coli",
            "core_result": "treatment-resolved conditioning geometry with PMA early burden and refined monotone memory coordinate",
            "memory_status": "refined monotone treatment-path memory",
            "explicit_R_status": "latent_with_treatment_return",
            "top_recovery_mode": "pma_t60 partial restoration",
            "main_frontier": "pma_t30",
        },
    ]

    frontier_rows = [
        {"dataset": "GSE36599", "frontier_group": "acute stress adaptation", "frontier_role": "benchmark burdened instability wing"},
        {"dataset": "GSE4370", "frontier_group": "0 to 15 min", "frontier_role": "early recovery burden wing"},
        {"dataset": "GSE206609", "frontier_group": "mmc", "frontier_role": "drug-resolved explicit high-burden frontier"},
        {"dataset": "GSE95575", "frontier_group": "pma_t30", "frontier_role": "conditioning branch early burdened frontier"},
    ]

    df1 = pd.DataFrame(dataset_rows)
    df2 = pd.DataFrame(frontier_rows)

    out1 = out_dir / "microbes_dataset_comparison_v3.csv"
    out2 = out_dir / "microbes_frontier_variance_comparison_v3.csv"
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)

    summary = {
        "datasets_compared": df1["dataset"].tolist(),
        "explicit_R_upgrade_dataset": "GSE206609",
        "conditioning_memory_upgrade_dataset": "GSE95575",
        "dominant_conditioning_frontier": "pma_t30",
    }

    log_path = log_dir / "cross_dataset_summary_v3.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out1}")
    print(f"[ok] wrote: {out2}")
    print(f"[ok] wrote: {log_path}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
