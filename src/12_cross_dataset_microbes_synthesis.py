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

    # GSE36599 summaries
    g1_velocity = json.load(open("results/logs/gse36599/vector_velocity_summary.json"))
    g1_variance = pd.read_csv("data/processed/gse36599/gse36599_variance_without_frontier.csv")
    g1_resid = pd.read_csv("data/processed/gse36599/gse36599_residual_field_conditions.csv")
    g1_state = pd.read_csv("data/processed/gse36599/gse36599_state_table_conditions.csv")

    # GSE4370 summaries
    g2_velocity = json.load(open("results/logs/gse4370/vector_velocity_summary.json"))
    g2_variance = pd.read_csv("data/processed/gse4370/gse4370_variance_without_early_frontier.csv")
    g2_phase = pd.read_csv("data/processed/gse4370/gse4370_phase_summary.csv")
    g2_state = pd.read_csv("data/processed/gse4370/gse4370_state_table_samples.csv")

    comparison_rows = [
        {
            "dataset": "GSE36599",
            "organism": "Saccharomyces cerevisiae",
            "design": "staged stress adaptation and recovery",
            "n_samples": 10,
            "path_length_3d": g1_velocity["path_length_3d"],
            "direct_displacement_3d": g1_velocity["direct_displacement_3d"],
            "path_efficiency": g1_velocity["path_efficiency"],
        },
        {
            "dataset": "GSE4370",
            "organism": "Escherichia coli",
            "design": "continuous recovery time-course",
            "n_samples": 7,
            "path_length_3d": g2_velocity["path_length_3d"],
            "direct_displacement_3d": g2_velocity["direct_displacement_3d"],
            "path_efficiency": g2_velocity["path_efficiency"],
        },
    ]
    comparison = pd.DataFrame(comparison_rows)

    # frontier sensitivity comparison for common packets
    common_packets = ["residual_field", "stress_burden", "recoverability_R", "H", "S"]
    frontier_comp = []
    for pkt in common_packets:
        a = g1_variance.loc[g1_variance["packet"] == pkt, "variance_ratio_without_acute_stress"].iloc[0]
        b = g2_variance.loc[g2_variance["packet"] == pkt, "variance_ratio_without_0_15min"].iloc[0]
        frontier_comp.append(
            {
                "packet": pkt,
                "GSE36599_ratio_after_frontier_removal": a,
                "GSE4370_ratio_after_frontier_removal": b,
            }
        )
    frontier_comp = pd.DataFrame(frontier_comp)

    comparison.to_csv(f"{outdir}/microbes_dataset_comparison.csv", index=False)
    frontier_comp.to_csv(f"{outdir}/microbes_frontier_variance_comparison.csv", index=False)

    summary = {
        "datasets_compared": ["GSE36599", "GSE4370"],
        "main_contrast": {
            "GSE36599": "more circuitous staged return",
            "GSE4370": "more directed continuous return",
        },
    }

    with open(f"{logdir}/cross_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbes_dataset_comparison.csv")
    print("[ok] wrote:", f"{outdir}/microbes_frontier_variance_comparison.csv")
    print("[ok] wrote:", f"{logdir}/cross_dataset_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
