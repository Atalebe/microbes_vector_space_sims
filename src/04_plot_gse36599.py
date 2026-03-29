from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    fig_dir = "results/figures/gse36599"
    ensure_dir(fig_dir)

    samples = pd.read_csv(f"{processed_dir}/gse36599_residual_field_samples.csv")
    conds = pd.read_csv(f"{processed_dir}/gse36599_residual_field_conditions.csv")

    # ---------------------------------
    # Figure 1: PCA samples + centroids
    # ---------------------------------
    plt.figure(figsize=(8, 6))
    for condition, dfc in samples.groupby("condition"):
        plt.scatter(dfc["PC1"], dfc["PC2"], label=condition)
        for _, r in dfc.iterrows():
            plt.text(r["PC1"], r["PC2"], str(int(r["replicate"])), fontsize=8)

    centroids = (
        samples.groupby(["condition", "condition_order"], as_index=False)
        .agg(PC1=("PC1", "mean"), PC2=("PC2", "mean"))
        .sort_values("condition_order")
    )

    plt.plot(centroids["PC1"], centroids["PC2"], marker="o")
    for _, r in centroids.iterrows():
        plt.text(r["PC1"], r["PC2"], r["condition"], fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GSE36599 PCA geometry with ordered condition path")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_pca_geometry.png", dpi=300)
    plt.close()

    # ---------------------------------
    # Figure 2: Backbone + residuals
    # ---------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(samples["PC1"], samples["PC2"])

    xs = np.linspace(samples["PC1"].min(), samples["PC1"].max(), 300)
    coeffs = np.polyfit(samples["PC1"], samples["PC2"], deg=2)
    ys = np.polyval(coeffs, xs)
    plt.plot(xs, ys)

    for _, r in samples.iterrows():
        plt.plot([r["PC1"], r["PC1"]], [r["PC2"], r["pc2_backbone_fit"]], alpha=0.6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Residual field relative to quadratic backbone")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_backbone_residuals.png", dpi=300)
    plt.close()

    # ---------------------------------
    # Figure 3: Condition-level H,S,M,R
    # ---------------------------------
    state = pd.read_csv(f"{processed_dir}/gse36599_state_table_conditions.csv").sort_values("condition_order")

    x = np.arange(len(state))
    plt.figure(figsize=(9, 6))
    plt.plot(x, state["H_mean"], marker="o", label="H")
    plt.plot(x, state["S_mean"], marker="o", label="S")
    plt.plot(x, state["M_mean"], marker="o", label="M")
    plt.plot(x, state["R_mean"], marker="o", label="R")
    plt.xticks(x, state["condition"], rotation=30, ha="right")
    plt.ylabel("Condition mean")
    plt.title("First-pass condition-level H, S, M, R")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_state_means.png", dpi=300)
    plt.close()

    # ---------------------------------
    # Figure 4: Residual field vs burden
    # ---------------------------------
    plt.figure(figsize=(8, 6))
    for sector, dfs in samples.groupby("sector"):
        plt.scatter(dfs["stress_burden"], dfs["residual_field"], label=sector)
    plt.xlabel("Stress burden")
    plt.ylabel("Residual field")
    plt.title("Residual field versus stress burden")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_residual_vs_burden.png", dpi=300)
    plt.close()

    # ---------------------------------
    # Figure 5: Condition residual summary
    # ---------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(conds["condition_order"], conds["residual_field_mean"], marker="o", label="residual field")
    plt.plot(conds["condition_order"], conds["burden_mean"], marker="o", label="burden")
    plt.plot(conds["condition_order"], conds["R_mean"], marker="o", label="R")
    plt.xticks(conds["condition_order"], conds["condition"], rotation=30, ha="right")
    plt.ylabel("Condition mean")
    plt.title("Condition-level residual field, burden, and recoverability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_condition_residual_summary.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate first visualization set for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)
