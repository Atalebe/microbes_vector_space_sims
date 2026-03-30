#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    out_dir = Path("data/processed/microbial_closure")
    log_dir = Path("results/logs/microbial_closure")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "dataset": "GSE36599",
            "organism": "Saccharomyces cerevisiae",
            "branch_role": "staged benchmark stress-recovery branch",
            "n_samples": 10,
            "memory_strength": "ordinal stage memory only",
            "explicit_R_status": "latent_to_bridge",
            "rank": 1,
            "notes": "Best benchmark branch, strongest staged geometry, refined stability, and clear stress-to-recovery structure.",
        },
        {
            "dataset": "GSE4370",
            "organism": "Escherichia coli",
            "branch_role": "continuous temporal recovery backbone",
            "n_samples": 7,
            "memory_strength": "normalized time memory",
            "explicit_R_status": "temporal_bridge",
            "rank": 2,
            "notes": "Best temporal backbone branch, continuous return geometry, strong hinge at 60 min, and cross-dataset projection anchor.",
        },
        {
            "dataset": "GSE206609",
            "organism": "Escherichia coli",
            "branch_role": "explicit treatment-recovery recoverability branch",
            "n_samples": 24,
            "memory_strength": "binary phase memory",
            "explicit_R_status": "explicit_observed",
            "rank": 3,
            "notes": "Best explicit-R branch, drug-resolved recoverability ranking, and strongest observed recovery contrast.",
        },
        {
            "dataset": "GSE95575",
            "organism": "Escherichia coli",
            "branch_role": "conditioning-memory upgrade branch",
            "n_samples": 30,
            "memory_strength": "refined monotone treatment-path memory",
            "explicit_R_status": "latent_with_treatment_return",
            "rank": 4,
            "notes": "Added to strengthen microbial M; PMA defines dominant early burdened frontier and refined M_v2 passes monotonicity hardening.",
        },
    ]

    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    out_path = out_dir / "microbial_dataset_suitability_ranking_v2.csv"
    df.to_csv(out_path, index=False)

    summary = {
        "datasets_ranked": df["dataset"].tolist(),
        "top_ranked_dataset": df.iloc[0]["dataset"],
        "conditioning_memory_upgrade_dataset": "GSE95575",
    }

    log_path = log_dir / "dataset_suitability_v2_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {log_path}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
