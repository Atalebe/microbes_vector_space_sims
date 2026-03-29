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
    fig_dir = "results/figures/gse4370"
    ensure_dir(fig_dir)

    backbone = pd.read_csv(f"{processed_dir}/gse4370_backbone_sensitivity.csv")
    steps = pd.read_csv(f"{processed_dir}/gse4370_centroid_steps.csv")
    var_summary = pd.read_csv(f"{processed_dir}/gse4370_variance_packet_summary.csv")
    var_frontier = pd.read_csv(f"{processed_dir}/gse4370_variance_without_early_frontier.csv")

    # 1. Backbone sensitivity
    plt.figure(figsize=(8, 6))
    for model, d in backbone.groupby("backbone_model"):
        d = d.sort_values("time_min")
        plt.plot(d["time_min"], d["residual_fit_z"], marker="o", label=model)
    plt.xlabel("Recovery time (min)")
    plt.ylabel("Backbone residual z")
    plt.title("Backbone sensitivity across model forms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_backbone_sensitivity.png", dpi=300)
    plt.close()

    # 2. Step lengths
    plt.figure(figsize=(8, 6))
    x = np.arange(len(steps))
    labels = [f'{int(a)}→{int(b)}' for a, b in zip(steps["from_time_min"], steps["to_time_min"])]
    plt.plot(x, steps["step_length_3d"], marker="o")
    plt.xticks(x, labels)
    plt.ylabel("3D step length")
    plt.title("GSE4370 centroid step lengths")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_centroid_step_lengths.png", dpi=300)
    plt.close()

    # 3. Variance packets
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(var_summary)), var_summary["total_variance"])
    plt.xticks(np.arange(len(var_summary)), var_summary["packet"], rotation=30, ha="right")
    plt.ylabel("Variance")
    plt.title("GSE4370 variance by packet")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_variance_packets.png", dpi=300)
    plt.close()

    # 4. Early frontier sensitivity
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(var_frontier)), var_frontier["variance_ratio_without_0_15min"])
    plt.xticks(np.arange(len(var_frontier)), var_frontier["packet"], rotation=30, ha="right")
    plt.ylabel("Remaining variance ratio")
    plt.title("Variance sensitivity after removing 0 and 15 min frontier")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_frontier_sensitivity.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extended validation, velocity, and variance plots for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
