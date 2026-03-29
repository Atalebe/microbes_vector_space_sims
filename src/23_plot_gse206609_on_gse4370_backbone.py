from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    figdir = "results/figures/cross_dataset_microbes"
    ensure_dir(figdir)

    backbone = pd.read_csv("data/processed/cross_dataset_microbes/gse4370_temporal_backbone_centroids.csv")
    proj = pd.read_csv("data/processed/cross_dataset_microbes/gse206609_projected_onto_gse4370_backbone.csv")

    # PC1-PC2 view
    plt.figure(figsize=(8, 6))
    plt.plot(backbone["PC1"], backbone["PC2"], marker="o")
    for _, r in backbone.iterrows():
        plt.text(r["PC1"], r["PC2"], f'{int(r["time_min"])}', fontsize=8)

    plt.scatter(proj["proj_PC1"], proj["proj_PC2"], marker="s")
    for _, r in proj.iterrows():
        label = f'{r["treatment"]} -> {r["inferred_gse4370_time_min"]:.1f} min'
        plt.text(r["proj_PC1"], r["proj_PC2"], label, fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GSE206609 recovery drugs projected onto GSE4370 temporal backbone")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_on_gse4370_pc12.png", dpi=300)
    plt.close()

    # inferred time plot
    plt.figure(figsize=(7, 5))
    plt.scatter(proj["treatment"], proj["inferred_gse4370_time_min"])
    plt.ylabel("Inferred position on GSE4370 timeline, min")
    plt.title("Drug recovery states mapped onto GSE4370 0 to 90 min timeline")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_on_gse4370_time_projection.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", figdir)


if __name__ == "__main__":
    main()
