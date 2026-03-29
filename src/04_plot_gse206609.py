from __future__ import annotations

import json
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
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    figdir = "results/figures/gse206609"
    ensure_dir(figdir)

    pca = pd.read_csv(f"{interim_dir}/audit_pca_scores.csv")
    cond = pd.read_csv(f"{processed_dir}/gse206609_state_table_conditions.csv")
    expR = pd.read_csv(f"{processed_dir}/gse206609_explicit_R_condition_summary.csv")

    # PCA scatter
    plt.figure(figsize=(8, 6))
    for phase, marker in [("treatment", "o"), ("recovery", "s")]:
        sub = pca.loc[pca["phase"] == phase]
        for treatment in sorted(sub["treatment"].unique()):
            s2 = sub.loc[sub["treatment"] == treatment]
            plt.scatter(s2["PC1"], s2["PC2"], marker=marker, label=f"{treatment}_{phase}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GSE206609 PCA, treatment vs recovery")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_pca_pc12.png", dpi=300)
    plt.close()

    # H-S by condition
    plt.figure(figsize=(8, 6))
    plt.scatter(cond["H_mean"], cond["S_mean"])
    for _, r in cond.iterrows():
        plt.text(r["H_mean"], r["S_mean"], r["condition"], fontsize=8)
    plt.xlabel("H mean")
    plt.ylabel("S mean")
    plt.title("GSE206609 condition means in H-S space")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_condition_HS.png", dpi=300)
    plt.close()

    # explicit R by drug
    plt.figure(figsize=(7, 5))
    plt.scatter(expR["R_mean"], expR["explicit_R_observed"])
    for _, r in expR.iterrows():
        plt.text(r["R_mean"], r["explicit_R_observed"], r["drug"], fontsize=9)
    plt.xlabel("Latent R mean")
    plt.ylabel("Explicit observed R")
    plt.title("GSE206609 latent vs explicit recoverability by drug")
    plt.tight_layout()
    plt.savefig(f"{figdir}/gse206609_latent_vs_explicit_R.png", dpi=300)
    plt.close()

    print("[ok] wrote figures to", figdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
