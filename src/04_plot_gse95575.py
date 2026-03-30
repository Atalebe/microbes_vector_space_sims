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

    s = pd.read_csv(processed_root / "gse95575_state_table_samples.csv")
    c = pd.read_csv(processed_root / "gse95575_state_table_conditions.csv")
    r = pd.read_csv(processed_root / "gse95575_residual_field_samples.csv")
    p = pd.read_csv(processed_root / "gse95575_phase_summary.csv")

    # PCA geometry
    plt.figure(figsize=(8, 6))
    for treatment, sub in s.groupby("treatment"):
        plt.scatter(sub["PC1"], sub["PC2"], label=treatment)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_pca_geometry.png", dpi=300)
    plt.close()

    # H-S condition means
    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        plt.plot(sub["H_mean"], sub["S_mean"], marker="o", label=treatment)
        for _, row in sub.iterrows():
            plt.text(row["H_mean"], row["S_mean"], f"{int(row['time_min'])}", fontsize=8)
    plt.xlabel("H mean")
    plt.ylabel("S mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_condition_HS.png", dpi=300)
    plt.close()

    # M-R condition means
    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        plt.plot(sub["M_mean"], sub["R_mean"], marker="o", label=treatment)
        for _, row in sub.iterrows():
            plt.text(row["M_mean"], row["R_mean"], f"{int(row['time_min'])}", fontsize=8)
    plt.xlabel("M mean")
    plt.ylabel("R mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_condition_MR.png", dpi=300)
    plt.close()

    # burden vs residual
    plt.figure(figsize=(8, 6))
    for treatment, sub in r.groupby("treatment"):
        plt.scatter(sub["stress_burden"], sub["residual_field"], label=treatment)
    plt.xlabel("Stress burden")
    plt.ylabel("Residual field")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_burden_residual_recovery.png", dpi=300)
    plt.close()

    # state trajectories
    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min")
        plt.plot(sub["time_min"], sub["M_mean"], marker="o", label=f"{treatment}: M")
    plt.xlabel("Time (min)")
    plt.ylabel("M mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_memory_trajectories.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min")
        plt.plot(sub["time_min"], sub["R_mean"], marker="o", label=f"{treatment}: R")
    plt.xlabel("Time (min)")
    plt.ylabel("R mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_recovery_trajectories.png", dpi=300)
    plt.close()

    print(f"[ok] wrote figures to {figures_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
