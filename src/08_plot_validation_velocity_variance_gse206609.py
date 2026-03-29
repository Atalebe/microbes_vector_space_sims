from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    figdir = "results/figures/gse206609"
    ensure_dir(figdir)

    rank = pd.read_csv(f"{processed_dir}/gse206609_explicit_R_drug_ranking.csv")
    steps = pd.read_csv(f"{processed_dir}/gse206609_centroid_steps.csv")
    varsum = pd.read_csv(f"{processed_dir}/gse206609_variance_without_mmc.csv")

    # explicit R ranking
    plt.figure(figsize=(7, 5))
    plt.scatter(rank["explicit_R_rank_desc"], rank["explicit_R_observed"])
    for _, r in rank.iterrows():
        plt.text(r["explicit_R_rank_desc"], r["explicit_R_observed"], r["drug"], fontsize=9)
    plt.xlabel("Explicit R rank")
    plt.ylabel("Explicit observed R")
    plt.title("GSE206609 explicit recoverability ranking")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_explicit_R_ranking.png", dpi=300)
    plt.close()

    # centroid return efficiency
    plt.figure(figsize=(7, 5))
    plt.scatter(steps["step_length_3d"], steps["anchor_return_efficiency"])
    for _, r in steps.iterrows():
        plt.text(r["step_length_3d"], r["anchor_return_efficiency"], r["treatment"], fontsize=9)
    plt.xlabel("Treatment to recovery step length")
    plt.ylabel("Anchor return efficiency")
    plt.title("GSE206609 recovery return efficiency by drug")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_return_efficiency.png", dpi=300)
    plt.close()

    # variance ratio without MMC
    plt.figure(figsize=(8, 5))
    plt.scatter(varsum["packet"], varsum["variance_ratio_without_mmc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Variance ratio without MMC")
    plt.title("GSE206609 variance scaling without MMC")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_variance_without_mmc.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", figdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
