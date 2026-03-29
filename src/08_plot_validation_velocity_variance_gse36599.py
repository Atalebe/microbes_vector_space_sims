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

    backbone = pd.read_csv(f"{processed_dir}/gse36599_backbone_sensitivity.csv")
    loo = pd.read_csv(f"{processed_dir}/gse36599_leave_one_out_sensitivity.csv")
    steps = pd.read_csv(f"{processed_dir}/gse36599_centroid_steps.csv")
    var_summary = pd.read_csv(f"{processed_dir}/gse36599_variance_packet_summary.csv")
    var_frontier = pd.read_csv(f"{processed_dir}/gse36599_variance_without_frontier.csv")
    var_top2 = pd.read_csv(f"{processed_dir}/gse36599_variance_without_top2_acute.csv")

    # 1. Backbone sensitivity by condition
    plt.figure(figsize=(8, 6))
    for model, d in backbone.groupby("backbone_model"):
        plt.plot(d["condition_order"], d["residual_mean"], marker="o", label=model)
    labels = (
        backbone.sort_values("condition_order")[["condition_order", "condition"]]
        .drop_duplicates()
        .sort_values("condition_order")
    )
    plt.xticks(labels["condition_order"], labels["condition"], rotation=30, ha="right")
    plt.ylabel("Condition mean residual z")
    plt.title("Backbone sensitivity across model forms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_backbone_sensitivity.png", dpi=300)
    plt.close()

    # 2. Leave-one-out acute mean
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(loo)), loo["acute_mean"], marker="o")
    plt.xticks(np.arange(len(loo)), loo["left_out_sample"], rotation=45, ha="right")
    plt.ylabel("Acute-stress mean residual")
    plt.title("Leave-one-out sensitivity of acute-stress residual mean")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_leave_one_out_acute_mean.png", dpi=300)
    plt.close()

    # 3. Step lengths
    plt.figure(figsize=(8, 6))
    step_labels = steps["from_condition"] + " -> " + steps["to_condition"]
    plt.plot(np.arange(len(steps)), steps["step_length_3d"], marker="o")
    plt.xticks(np.arange(len(steps)), step_labels, rotation=30, ha="right")
    plt.ylabel("3D step length")
    plt.title("Centroid trajectory step lengths")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_centroid_step_lengths.png", dpi=300)
    plt.close()

    # 4. Variance packet summary
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(var_summary)), var_summary["total_variance"])
    plt.xticks(np.arange(len(var_summary)), var_summary["packet"], rotation=30, ha="right")
    plt.ylabel("Variance")
    plt.title("Variance by packet")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_variance_packets.png", dpi=300)
    plt.close()

    # 5. Frontier sensitivity ratios
    x = np.arange(len(var_frontier))
    width = 0.35
    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, var_frontier["variance_ratio_without_acute_stress"], width=width, label="without acute stress")
    plt.bar(x + width / 2, var_top2["variance_ratio_without_top2_acute"], width=width, label="without top2 acute")
    plt.xticks(x, var_frontier["packet"], rotation=30, ha="right")
    plt.ylabel("Remaining variance ratio")
    plt.title("Variance sensitivity to frontier removal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse36599_variance_frontier_sensitivity.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended validation, velocity, and variance plots for GSE36599.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
