#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def choose_display_symbol(row: pd.Series) -> str:
    sym = str(row.get("gene_symbol", "")).strip()
    if sym and sym.lower() != "nan":
        return sym
    return str(row.get("feature_id", "")).strip()


def choose_description(row: pd.Series) -> str:
    desc = str(row.get("gene_description", "")).strip()
    if desc and desc.lower() != "nan":
        return desc
    feature = str(row.get("feature", "")).strip()
    if feature and feature.lower() != "nan":
        return feature
    return ""


def main() -> None:
    processed_dir = Path("data/processed/gse95575")
    log_dir = Path("results/logs/gse95575")
    processed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    spike = pd.read_csv(processed_dir / "gse95575_9C_PMA_t30_top_spike_features_annotated.csv")
    down = pd.read_csv(processed_dir / "gse95575_9C_PMA_t30_top_down_vs_peers_annotated.csv")

    spike_top = spike.head(15).copy()
    spike_top["display_symbol"] = spike_top.apply(choose_display_symbol, axis=1)
    spike_top["description_short"] = spike_top.apply(choose_description, axis=1)
    spike_top["spike_direction"] = "up"

    down_top = down.head(15).copy()
    down_top["display_symbol"] = down_top.apply(choose_display_symbol, axis=1)
    down_top["description_short"] = down_top.apply(choose_description, axis=1)
    down_top["spike_direction"] = "down"

    cols = [
        "feature_id",
        "display_symbol",
        "description_short",
        "functional_class",
        "delta_vs_peer_pma_t30",
        "delta_vs_control_t30",
        "spike_direction",
    ]

    summary = pd.concat([spike_top[cols], down_top[cols]], ignore_index=True)

    summary = summary.rename(
        columns={
            "display_symbol": "gene_symbol",
            "description_short": "gene_description",
        }
    )

    out_path = processed_dir / "gse95575_pma_spike_summary_table.csv"
    summary.to_csv(out_path, index=False)

    summary_json = {
        "dataset_name": "gse95575",
        "target_sample": "9C_PMA_t30",
        "n_up_rows": int((summary["spike_direction"] == "up").sum()),
        "n_down_rows": int((summary["spike_direction"] == "down").sum()),
        "functional_class_counts": summary["functional_class"].value_counts().to_dict(),
    }

    log_path = log_dir / "gse95575_pma_spike_summary_table.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {log_path}")
    print("[summary]", json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
