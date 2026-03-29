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

    ranking = pd.read_csv("data/processed/gse206609/gse206609_explicit_R_final_ranking.csv")
    proj = pd.read_csv("data/processed/cross_dataset_microbes/gse206609_projected_onto_gse4370_backbone.csv")

    final = ranking.merge(proj[["treatment", "inferred_gse4370_time_min", "projection_distance"]], left_on="drug", right_on="treatment", how="left")
    final = final.drop(columns=["treatment"])
    final.to_csv(f"{outdir}/microbes_explicit_R_and_temporal_projection_summary.csv", index=False)

    summary = {
        "rank_desc": final["drug"].tolist(),
        "projected_minutes": final[["drug", "inferred_gse4370_time_min"]].to_dict(orient="records")
    }

    with open(f"{logdir}/microbes_explicit_R_and_temporal_projection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbes_explicit_R_and_temporal_projection_summary.csv")
    print("[ok] wrote:", f"{logdir}/microbes_explicit_R_and_temporal_projection_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
