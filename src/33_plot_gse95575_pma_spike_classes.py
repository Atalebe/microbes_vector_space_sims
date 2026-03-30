#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    processed_dir = Path("data/processed/gse95575")
    figures_dir = Path("results/figures/gse95575")
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(processed_dir / "gse95575_pma_spike_summary_table.csv")

    counts = summary["functional_class"].value_counts()

    plt.figure(figsize=(9, 6))
    plt.barh(counts.index.astype(str), counts.values)
    plt.gca().invert_yaxis()
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "gse95575_9C_PMA_t30_functional_classes.png", dpi=300)
    plt.close()

    print(f"[ok] wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
