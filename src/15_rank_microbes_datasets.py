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
            "organism": "Saccharomyces cerevisiae",
            "design": "staged stress adaptation and recovery",
            "time_structure_score_5": 4,
            "stress_recovery_clarity_score_5": 5,
            "replicates_score_5": 5,
            "explicit_recovery_observable_score_5": 2,
            "R_suitability_score_5": 4,
            "overall_suitability_score_25": 20,
            "rank": 1,
            "notes": "Best benchmark branch, strong stage clarity and replicate structure, weaker direct biological recovery metric.",
        },
        {
            "dataset": "GSE4370",
            "organism": "Escherichia coli",
            "design": "continuous recovery time-course",
            "time_structure_score_5": 5,
            "stress_recovery_clarity_score_5": 4,
            "replicates_score_5": 1,
            "explicit_recovery_observable_score_5": 2,
            "R_suitability_score_5": 4,
            "overall_suitability_score_25": 16,
            "rank": 2,
            "notes": "Excellent continuous recovery geometry and timing, weaker replicate support and no direct survival or regrowth metric.",
        },
    ]

    df = pd.DataFrame(rows).sort_values(["rank", "dataset"])
    df.to_csv(f"{outdir}/microbial_dataset_suitability_ranking.csv", index=False)

    summary = {
        "datasets_ranked": df["dataset"].tolist(),
        "top_ranked_dataset": df.iloc[0]["dataset"],
    }

    with open(f"{logdir}/dataset_suitability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_dataset_suitability_ranking.csv")
    print("[ok] wrote:", f"{logdir}/dataset_suitability_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
