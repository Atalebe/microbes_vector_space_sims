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

    c = pd.read_csv(processed_root / "gse95575_refined_memory_conditions.csv")
    s = pd.read_csv(processed_root / "gse95575_refined_memory_samples.csv")

    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min")
        plt.plot(sub["time_min"], sub["M_mean"], marker="o", label=f"{treatment} M_old")
        plt.plot(sub["time_min"], sub["M_v2"], marker="o", label=f"{treatment} M_v2")
    plt.xlabel("Time (min)")
    plt.ylabel("Memory coordinate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_old_vs_refined_M.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for treatment, sub in c.groupby("treatment"):
        sub = sub.sort_values("time_min")
        plt.plot(sub["M_v3"], sub["R_mean"], marker="o", label=treatment)
    plt.xlabel("M_v3")
    plt.ylabel("R mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_refined_memory_vs_R.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for treatment, sub in s.groupby("treatment"):
        plt.scatter(sub["M_v2"], sub["stress_burden"], label=treatment)
    plt.xlabel("M_v2")
    plt.ylabel("Stress burden")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_refined_memory_vs_burden.png", dpi=300)
    plt.close()

    print(f"[ok] wrote figures to {figures_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
