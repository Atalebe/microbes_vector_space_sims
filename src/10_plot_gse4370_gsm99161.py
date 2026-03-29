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
    fig_dir = "results/figures/gse4370"
    ensure_dir(fig_dir)

    corr = pd.read_csv(f"{processed_dir}/gse4370_gsm99161_correlations.csv")
    d1 = pd.read_csv(f"{processed_dir}/gse4370_gsm99161_top_delta_45_60.csv")
    d2 = pd.read_csv(f"{processed_dir}/gse4370_gsm99161_top_delta_60_75.csv")
    turn = pd.read_csv(f"{processed_dir}/gse4370_gsm99161_top_turning_features.csv")

    # 1. Correlations to 60 min
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(corr)), corr["spearman_corr_to_GSM99161"])
    plt.xticks(range(len(corr)), corr["other_sample"], rotation=30, ha="right")
    plt.ylabel("Spearman correlation to GSM99161")
    plt.title("Sample-level similarity to 60 min recovery")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_gsm99161_correlations.png", dpi=300)
    plt.close()

    # 2. Top turning features around 60 min
    top = turn.head(10).copy()
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(top)), top["delta_45_60"], marker="o", label="45→60")
    plt.plot(range(len(top)), top["delta_60_75"], marker="o", label="60→75")
    labels = top["GENE_SYMBOL"].fillna(top["feature_id"].astype(str))
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.ylabel("Log-ratio change")
    plt.title("Largest turning features around 60 min")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_gsm99161_turning_features.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plots for GSM99161 diagnostic inspection.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
