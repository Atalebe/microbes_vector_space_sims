from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/gse188790")
    ap.add_argument("--fig-dir", default="results/figures/gse188790")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    base = Path(args.processed_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    state_path = base / "gse188790_state_table.csv"
    state_summary_path = base / "gse188790_state_table_summary.json"
    age_summary_path = base / "gse188790_age_class_summary.csv"
    contrast_path = base / "gse188790_gene_age_contrast.csv"
    top_aged_path = base / "gse188790_top_aged_up_genes.csv"
    top_unaged_path = base / "gse188790_top_unaged_up_genes.csv"

    if not state_path.exists():
        raise SystemExit("Missing gse188790_state_table.csv. Run state-table step first.")
    if not contrast_path.exists():
        raise SystemExit("Missing gse188790_gene_age_contrast.csv. Run residual-field step first.")

    state = pd.read_csv(state_path)
    contrast = pd.read_csv(contrast_path)
    age_summary = pd.read_csv(age_summary_path) if age_summary_path.exists() else pd.DataFrame()
    top_aged = pd.read_csv(top_aged_path) if top_aged_path.exists() else pd.DataFrame()
    top_unaged = pd.read_csv(top_unaged_path) if top_unaged_path.exists() else pd.DataFrame()

    pc_var = {"pc1": np.nan, "pc2": np.nan}
    if state_summary_path.exists():
        obj = json.loads(state_summary_path.read_text())
        pc_var = obj.get("pc_variance_fraction", pc_var)

    state = state.copy()
    state["age_class"] = state["age_class"].astype(str)

    # Figure 1: PCA
    fig, ax = plt.subplots(figsize=(7, 6))
    for age_class in sorted(state["age_class"].unique()):
        sub = state.loc[state["age_class"].eq(age_class)]
        ax.scatter(sub["pc1"], sub["pc2"], s=80, label=age_class)
        for _, row in sub.iterrows():
            ax.text(row["pc1"], row["pc2"], row["sample_name"], fontsize=9)

    ax.set_xlabel(f"PC1 ({100 * float(pc_var.get('pc1', np.nan)):.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * float(pc_var.get('pc2', np.nan)):.1f}% var)")
    ax.set_title("GSE188790 PCA, aged versus unaged")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse188790_pca_age_classes.png", dpi=300)
    plt.close(fig)

    # Figure 2: state space H vs M
    fig, ax = plt.subplots(figsize=(7, 6))
    for age_class in sorted(state["age_class"].unique()):
        sub = state.loc[state["age_class"].eq(age_class)]
        ax.scatter(sub["H"], sub["M"], s=80, label=age_class)
        for _, row in sub.iterrows():
            ax.text(row["H"], row["M"], row["sample_name"], fontsize=9)

    ax.set_xlabel("H, transcriptomic depth")
    ax.set_ylabel("M, age ordering")
    ax.set_title("GSE188790 state space, H versus M")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse188790_state_space_H_vs_M.png", dpi=300)
    plt.close(fig)

    # Figure 3: state space S vs M
    fig, ax = plt.subplots(figsize=(7, 6))
    for age_class in sorted(state["age_class"].unique()):
        sub = state.loc[state["age_class"].eq(age_class)]
        ax.scatter(sub["S"], sub["M"], s=80, label=age_class)
        for _, row in sub.iterrows():
            ax.text(row["S"], row["M"], row["sample_name"], fontsize=9)

    ax.set_xlabel("S, structural coherence")
    ax.set_ylabel("M, age ordering")
    ax.set_title("GSE188790 state space, S versus M")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse188790_state_space_S_vs_M.png", dpi=300)
    plt.close(fig)

    # Figure 4: library size by sample
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(state))
    ax.bar(x, state["library_size"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(state["sample_name"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Library size")
    ax.set_title("GSE188790 library size by sample")
    fig.tight_layout()
    fig.savefig(fig_dir / "gse188790_library_size_by_sample.png", dpi=300)
    plt.close(fig)

    # Figure 5: detected genes by sample
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(state))
    ax.bar(x, state["detected_genes"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(state["sample_name"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Detected genes")
    ax.set_title("GSE188790 detected genes by sample")
    fig.tight_layout()
    fig.savefig(fig_dir / "gse188790_detected_genes_by_sample.png", dpi=300)
    plt.close(fig)

    # Figure 6: age-class summary in vector space
    if not age_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary_long = age_summary.melt(
            id_vars=["age_class"],
            value_vars=["mean_H", "mean_S", "mean_M", "mean_R"],
            var_name="coordinate",
            value_name="value",
        )
        age_levels = summary_long["age_class"].unique().tolist()
        coords = ["mean_H", "mean_S", "mean_M", "mean_R"]
        xpos = np.arange(len(coords))
        width = 0.35

        for i, age in enumerate(sorted(age_levels)):
            sub = summary_long.loc[summary_long["age_class"].eq(age)].set_index("coordinate").reindex(coords)
            ax.bar(xpos + (i - 0.5) * width, sub["value"].values, width=width, label=age)

        ax.set_xticks(xpos)
        ax.set_xticklabels(["H", "S", "M", "R"])
        ax.set_ylabel("Mean coordinate")
        ax.set_title("GSE188790 age-class mean state coordinates")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "gse188790_age_class_mean_state_coordinates.png", dpi=300)
        plt.close(fig)

    # Figure 7: top aged-up genes
    if not top_aged.empty:
        top_n = min(int(args.top_n), len(top_aged))
        sub = top_aged.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * top_n)))
        ax.barh(sub["gene_id"], sub["delta_log2cpm_aged_minus_unaged"])
        ax.set_xlabel("Delta log2CPM, aged minus unaged")
        ax.set_title(f"GSE188790 top {top_n} aged-up genes")
        fig.tight_layout()
        fig.savefig(fig_dir / "gse188790_top_aged_up_genes.png", dpi=300)
        plt.close(fig)

    # Figure 8: top unaged-up genes
    if not top_unaged.empty:
        top_n = min(int(args.top_n), len(top_unaged))
        sub = top_unaged.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * top_n)))
        ax.barh(sub["gene_id"], np.abs(sub["delta_log2cpm_aged_minus_unaged"]))
        ax.set_xlabel("Absolute delta log2CPM, unaged-favored")
        ax.set_title(f"GSE188790 top {top_n} unaged-up genes")
        fig.tight_layout()
        fig.savefig(fig_dir / "gse188790_top_unaged_up_genes.png", dpi=300)
        plt.close(fig)

    manifest = {
        "dataset": "GSE188790",
        "figure_dir": str(fig_dir),
        "figures_written": sorted([p.name for p in fig_dir.glob("gse188790_*.png")]),
    }
    (base / "gse188790_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
