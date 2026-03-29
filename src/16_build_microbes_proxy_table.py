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

    rows = [
        {
            "dataset": "GSE36599",
            "coordinate": "H",
            "proxy_name": "organized_state_proximity_vs_acute_displacement",
            "operational_definition": "Depth proxy from organized-state proximity versus acute-stress displacement in PCA geometry.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE36599",
            "coordinate": "S",
            "proxy_name": "S_v2_local_and_corridor_coherence",
            "operational_definition": "Refined stability from local centroid coherence, baseline-recovery corridor coherence, and reduced acute-stress domination.",
            "status": "adopted_refined",
        },
        {
            "dataset": "GSE36599",
            "coordinate": "M",
            "proxy_name": "ordered_exposure_stage",
            "operational_definition": "Explicit ordered exposure-history stage from normal growth through recovery.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE36599",
            "coordinate": "R",
            "proxy_name": "latent_return_projection",
            "operational_definition": "Latent recoverability from return-direction projection toward recovered state.",
            "status": "latent_only",
        },
        {
            "dataset": "GSE4370",
            "coordinate": "H",
            "proxy_name": "approach_to_late_recovered_structure",
            "operational_definition": "Depth proxy from approach toward late recovered structure and separation from early disrupted state.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE4370",
            "coordinate": "S",
            "proxy_name": "trajectory_corridor_coherence",
            "operational_definition": "Stability proxy from coherence with overall recovery corridor and smooth ordered placement.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE4370",
            "coordinate": "M",
            "proxy_name": "normalized_recovery_time",
            "operational_definition": "Explicit time-memory coordinate from normalized recovery time.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE4370",
            "coordinate": "R",
            "proxy_name": "directional_return_to_late_endpoint",
            "operational_definition": "Recoverability from directional return toward the late-recovery endpoint.",
            "status": "latent_only",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/microbial_adopted_proxy_table.csv", index=False)

    summary = {
        "datasets": sorted(df["dataset"].unique().tolist()),
        "coordinates": ["H", "S", "M", "R"],
    }

    with open(f"{logdir}/proxy_table_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_adopted_proxy_table.csv")
    print("[ok] wrote:", f"{logdir}/proxy_table_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
