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


def choose_label_col(df: pd.DataFrame) -> str:
    for cand in ["gene_symbol", "GENE_SYMBOL", "gene_name", "symbol", "feature_id"]:
        if cand in df.columns:
            return cand
    return "feature_id"


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_root = Path(cfg["processed_root"])
    figures_root = Path(cfg["figures_root"])
    ensure_dirs(figures_root)

    corr = pd.read_csv(processed_root / "gse95575_9C_PMA_t30_correlations.csv")
    spike = pd.read_csv(processed_root / "gse95575_9C_PMA_t30_top_spike_features.csv")
    up = pd.read_csv(processed_root / "gse95575_9C_PMA_t30_top_up_vs_peers.csv")
    down = pd.read_csv(processed_root / "gse95575_9C_PMA_t30_top_down_vs_peers.csv")

    label_col = choose_label_col(spike)

    plt.figure(figsize=(8, 6))
    plt.bar(range(min(10, len(corr))), corr.head(10)["spearman_corr_to_target"])
    plt.xticks(range(min(10, len(corr))), corr.head(10)["other_sample"], rotation=45, ha="right")
    plt.ylabel("Spearman correlation")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_9C_PMA_t30_correlations.png", dpi=300)
    plt.close()

    top = spike.head(20).copy()
    plt.figure(figsize=(9, 7))
    plt.barh(top[label_col].astype(str), top["residual_spike_score"])
    plt.gca().invert_yaxis()
    plt.xlabel("Residual spike score")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_9C_PMA_t30_spike_features.png", dpi=300)
    plt.close()

    topu = up.head(15).copy()
    plt.figure(figsize=(9, 7))
    plt.barh(topu[label_col].astype(str), topu["delta_vs_peer_pma_t30"])
    plt.gca().invert_yaxis()
    plt.xlabel("Delta vs peer PMA t30")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_9C_PMA_t30_up_vs_peers.png", dpi=300)
    plt.close()

    topd = down.head(15).copy().sort_values("delta_vs_peer_pma_t30", ascending=False)
    plt.figure(figsize=(9, 7))
    plt.barh(topd[label_col].astype(str), topd["delta_vs_peer_pma_t30"])
    plt.xlabel("Delta vs peer PMA t30")
    plt.tight_layout()
    plt.savefig(figures_root / "gse95575_9C_PMA_t30_down_vs_peers.png", dpi=300)
    plt.close()

    print(f"[ok] wrote figures to {figures_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
