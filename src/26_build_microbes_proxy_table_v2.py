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
        {
            "dataset": "GSE206609",
            "coordinate": "H",
            "proxy_name": "control_recovery_proximity",
            "operational_definition": "Depth proxy from approach toward control recovery structure and separation from drug-treated treatment states.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE206609",
            "coordinate": "S",
            "proxy_name": "reduced_treatment_control_burden",
            "operational_definition": "Stability proxy from reduced treatment versus control burden in matched treatment-recovery geometry.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE206609",
            "coordinate": "M",
            "proxy_name": "binary_phase_memory",
            "operational_definition": "Binary phase memory coordinate with treatment equals 0 and recovery equals 1.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE206609",
            "coordinate": "R",
            "proxy_name": "matched_treatment_to_recovery_return_projection",
            "operational_definition": "Explicit recoverability from matched treatment-to-recovery relaxation and differential-expression burden release.",
            "status": "explicit_observed",
        },
        {
            "dataset": "GSE95575",
            "coordinate": "H",
            "proxy_name": "late_control_proximity_vs_disrupted_sector",
            "operational_definition": "Depth proxy from approach toward late coherent control and separation from disrupted treatment sector.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE95575",
            "coordinate": "S",
            "proxy_name": "matched_control_coherence_plus_replicate_compactness",
            "operational_definition": "Stability proxy from matched-control coherence together with within-condition replicate compactness.",
            "status": "adopted_first_pass",
        },
        {
            "dataset": "GSE95575",
            "coordinate": "M",
            "proxy_name": "M_v2_monotone_centroid_path_progress",
            "operational_definition": "Refined conditioning-memory proxy from monotone centroid-path progress within each treatment branch.",
            "status": "adopted_refined",
        },
        {
            "dataset": "GSE95575",
            "coordinate": "R",
            "proxy_name": "directional_return_to_late_control_endpoint",
            "operational_definition": "Recoverability from directional return toward the late control endpoint.",
            "status": "adopted_first_pass",
        },
    ]

    out = pd.DataFrame(rows).sort_values(["dataset", "coordinate"]).reset_index(drop=True)

    out_path = out_dir / "microbial_adopted_proxy_table_v2.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "datasets": sorted(out["dataset"].unique().tolist()),
        "coordinates": sorted(out["coordinate"].unique().tolist()),
        "gse206609_R_status": "explicit_observed",
        "gse95575_memory_status": "adopted_refined",
    }

    log_path = log_dir / "proxy_table_v2_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {log_path}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
