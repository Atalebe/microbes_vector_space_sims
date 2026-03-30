#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    processed_root = Path(cfg.get("processed_root", "data/processed/gse95575"))
    figures_root = Path(cfg.get("figures_root", "results/figures/gse95575"))
    cfg.setdefault("processed_root", str(processed_root))
    cfg.setdefault("figures_root", str(figures_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_root = Path(cfg["processed_root"])
    figures_root = Path(cfg["figures_root"])
    ensure_dirs(figures_root)

    backbone = pd.read_csv(processed_root / "gse95575_backbone_sensitivity.csv")
    mono = pd.read_csv(processed_root / "gse95575_memory_monotonicity.csv")
    steps = pd.read_csv(processed_root / "gse95575_centroid_steps.csv")
    vel = pd.read_csv(processed_root / "gse95575_sample_velocity_summary.csv")
    packets = pd.read_csv(processed_root / "gse95575_variance_packet_summary.csv")
    wpma = pd.read_csv(processed_root / "gse95575_variance_without_pma.csv")
    whg = pd.read_csv(processed_root / "gse95575_variance_without_hgcl2.csv")

    plt.figure(figsize=(8, 6))
    for model, sub in backbone.groupby("backbone_model"):
        means = sub.groupby("condition")["residual_fit_z"].mean().sort_values(ascending=False)
        plt.plot(range(len(means)), means.values, marker="o", label=model)
    plt.xlabel("Condition rank")
    plt.ylabel("Mean residual z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_backbone_sensitivity.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for coord, sub in mono.groupby("coordinate"):
        vals = sub.set_index("treatment")["n_non_decreasing"]
        plt.plot(range(len(vals)), vals.values, marker="o", label=coord)
    plt.xlabel("Treatment index")
    plt.ylabel("Non-decreasing steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_memory_monotonicity.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for treatment, sub in steps.groupby("treatment"):
        plt.plot(range(len(sub)), sub["step_length_4d"].values, marker="o", label=treatment)
    plt.xlabel("Step index")
    plt.ylabel("Step length 4D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_centroid_step_lengths.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(vel["path_length_4d"], vel["net_dR"])
    for _, row in vel.iterrows():
        plt.text(row["path_length_4d"], row["net_dR"], row["treatment"], fontsize=8)
    plt.xlabel("Path length 4D")
    plt.ylabel("Net dR")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_velocity_return.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(packets["packet"], packets["total_variance"], marker="o", label="full")
    plt.plot(wpma["packet"], wpma["variance_without_pma"], marker="o", label="without_pma")
    plt.plot(whg["packet"], whg["variance_without_hgcl2"], marker="o", label="without_hgcl2")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_variance_frontier_sensitivity.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(packets["packet"], packets["total_variance"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total variance")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_variance_packets.png", dpi=300)
    plt.close()

    print(f"[ok] wrote figures to {figures_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
