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

    cond = pd.read_csv(f"{processed_dir}/gse36599_refined_stability_conditions.csv")
    val = pd.read_csv(f"{processed_dir}/gse36599_refined_stability_validation.csv")
    samp = pd.read_csv(f"{processed_dir}/gse36599_refined_stability_samples.csv")

    # 1. Old S vs new S by condition
    x = np.arange(len(cond))
    plt.figure(figsize=(8, 6))
    plt.plot(x, cond["S_old_mean"], marker="o", label="S old")
    plt.plot(x, cond["S_v2_mean"], marker="o", label="S v2")
    plt.xticks(x, cond["condition"], rotation=30, ha="right")
    plt.ylabel("Condition mean")
    plt.title("Old versus refined stability coordinate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_old_vs_refined_S.png", dpi=300)
    plt.close()

    # 2. Residual field old vs v2
    plt.figure(figsize=(8, 6))
    plt.plot(x, cond["residual_field_old_mean"], marker="o", label="residual old")
    plt.plot(x, cond["residual_field_v2_mean"], marker="o", label="residual v2")
    plt.xticks(x, cond["condition"], rotation=30, ha="right")
    plt.ylabel("Condition mean")
    plt.title("Residual field before and after refined stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_old_vs_refined_residual_field.png", dpi=300)
    plt.close()

    # 3. Sample-level S old vs S v2
    plt.figure(figsize=(8, 6))
    for condition, d in samp.groupby("condition"):
        plt.scatter(d["S"], d["S_v2"], label=condition)
    plt.xlabel("S old")
    plt.ylabel("S v2")
    plt.title("Sample-level comparison of old and refined stability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_sample_old_vs_refined_S.png", dpi=300)
    plt.close()

    # 4. Stability components
    plt.figure(figsize=(8, 6))
    plt.plot(x, val["own_centroid_dist_mean"], marker="o", label="own centroid dist")
    plt.plot(x, val["corridor_dist_mean"], marker="o", label="corridor dist")
    plt.xticks(x, val["condition"], rotation=30, ha="right")
    plt.ylabel("Mean distance")
    plt.title("Refined stability ingredients by condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_refined_stability_components.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots for refined stability analysis.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
