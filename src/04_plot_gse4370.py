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
    interim_dir = cfg["paths"]["interim_dir"]
    fig_dir = "results/figures/gse4370"
    ensure_dir(fig_dir)

    pca_df = pd.read_csv(f"{interim_dir}/audit_pca_scores.csv").sort_values("time_order")
    steps = pd.read_csv(f"{interim_dir}/audit_time_steps.csv")
    state = pd.read_csv(f"{processed_dir}/gse4370_state_table_samples.csv").sort_values("time_order")
    resid = pd.read_csv(f"{processed_dir}/gse4370_residual_field_samples.csv").sort_values("time_order")

    # 1. PCA trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(pca_df["PC1"], pca_df["PC2"], marker="o")
    for _, r in pca_df.iterrows():
        plt.text(r["PC1"], r["PC2"], f'{int(r["time_min"])}', fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GSE4370 recovery trajectory in PCA space")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_pca_trajectory.png", dpi=300)
    plt.close()

    # 2. Step lengths
    plt.figure(figsize=(8, 6))
    x = np.arange(len(steps))
    labels = [f'{int(a)}→{int(b)}' for a, b in zip(steps["from_time_min"], steps["to_time_min"])]
    plt.plot(x, steps["step_length_3d"], marker="o")
    plt.xticks(x, labels)
    plt.ylabel("3D step length")
    plt.title("Recovery step lengths across time")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_step_lengths.png", dpi=300)
    plt.close()

    # 3. H, S, M, R over time
    plt.figure(figsize=(9, 6))
    plt.plot(state["time_min"], state["H"], marker="o", label="H")
    plt.plot(state["time_min"], state["S"], marker="o", label="S")
    plt.plot(state["time_min"], state["M"], marker="o", label="M")
    plt.plot(state["time_min"], state["R"], marker="o", label="R")
    plt.xlabel("Recovery time (min)")
    plt.ylabel("State value")
    plt.title("GSE4370 first-pass H, S, M, R over recovery time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_state_trajectories.png", dpi=300)
    plt.close()

    # 4. Burden and residual field
    plt.figure(figsize=(9, 6))
    plt.plot(resid["time_min"], resid["stress_burden"], marker="o", label="stress burden")
    plt.plot(resid["time_min"], resid["residual_field"], marker="o", label="residual field")
    plt.plot(resid["time_min"], resid["R"], marker="o", label="R")
    plt.xlabel("Recovery time (min)")
    plt.ylabel("Value")
    plt.title("Burden, residual field, and recoverability over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/gse4370_burden_residual_recovery.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", fig_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate first visualization set for GSE4370.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
