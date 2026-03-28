#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from utils import load_config, read_table, subset_required, write_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(Path(cfg["io"]["processed_dir"]) / "state_table.parquet")
    df = subset_required(df, ["H", "S", "M"])

    out_fig = Path(cfg["io"]["results_dir"]) / "figures"
    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tbl.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(df["H"], df["S"], s=12, alpha=0.7)
    plt.xlabel("H")
    plt.ylabel("S")
    plt.title("Microbial state space: H vs S")
    plt.tight_layout()
    plt.savefig(out_fig / "geometry_HS.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["S"], df["M"], s=12, alpha=0.7)
    plt.xlabel("S")
    plt.ylabel("M")
    plt.title("Microbial state space: S vs M")
    plt.tight_layout()
    plt.savefig(out_fig / "geometry_SM.png", dpi=200)
    plt.close()

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(df[["H", "S", "M"]])
    pca_df = pd.DataFrame({
        "pc1": pcs[:, 0],
        "pc2": pcs[:, 1],
        "pc3": pcs[:, 2],
        "explained_variance_ratio": list(pca.explained_variance_ratio_) + [None] * max(0, len(df) - 3),
    })
    write_table(pca_df[["pc1", "pc2", "pc3"]], out_tbl / "geometry_pca_scores.csv")
    pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "loading_H": pca.components_[:, 0],
        "loading_S": pca.components_[:, 1],
        "loading_M": pca.components_[:, 2],
    }).to_csv(out_tbl / "geometry_pca_loadings.csv", index=False)

    print("[ok] wrote geometry figures and PCA summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
