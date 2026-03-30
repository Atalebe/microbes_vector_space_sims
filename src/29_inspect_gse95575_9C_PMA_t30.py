#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


TARGET_SAMPLE = "9C_PMA_t30"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    interim_root = Path(cfg.get("interim_root", "data/interim/gse95575"))
    processed_root = Path(cfg.get("processed_root", "data/processed/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))
    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("interim_root", str(interim_root))
    cfg.setdefault("processed_root", str(processed_root))
    cfg.setdefault("logs_root", str(logs_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank(), method="pearson"))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    interim_root = Path(cfg["interim_root"])
    processed_root = Path(cfg["processed_root"])
    logs_root = Path(cfg["logs_root"])
    ensure_dirs(processed_root, logs_root)

    master = pd.read_parquet(interim_root / "master_table.parquet")
    feature_meta = pd.read_csv(interim_root / "feature_metadata.csv")
    sample_meta = pd.read_csv(interim_root / "sample_metadata.csv")
    state = pd.read_csv(processed_root / "gse95575_residual_field_samples.csv")

    wide = master.pivot_table(index="feature_id", columns="sample_accession", values="expression_value", aggfunc="mean")
    wide = wide.apply(pd.to_numeric, errors="coerce")

    if TARGET_SAMPLE not in wide.columns:
        raise ValueError(f"{TARGET_SAMPLE} not found in wide matrix.")

    expr_target = wide[TARGET_SAMPLE]

    corr_rows = []
    for col in wide.columns:
        corr_rows.append(
            {
                "other_sample": col,
                "spearman_corr_to_target": spearman_corr(expr_target, wide[col]),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_corr_to_target", ascending=False)

    pma_t30 = sample_meta.loc[sample_meta["condition"] == "pma_t30", "sample_accession"].astype(str).tolist()
    control_t30 = sample_meta.loc[sample_meta["condition"] == "control_t30", "sample_accession"].astype(str).tolist()
    pma_t10 = sample_meta.loc[sample_meta["condition"] == "pma_t10", "sample_accession"].astype(str).tolist()
    pma_t60 = sample_meta.loc[sample_meta["condition"] == "pma_t60", "sample_accession"].astype(str).tolist()

    peer_pma_t30 = [x for x in pma_t30 if x != TARGET_SAMPLE]

    out = pd.DataFrame({"feature_id": wide.index.astype(str)})
    out["expr_target"] = expr_target.values

    if peer_pma_t30:
        out["mean_peer_pma_t30"] = wide[peer_pma_t30].mean(axis=1).values
        out["delta_vs_peer_pma_t30"] = out["expr_target"] - out["mean_peer_pma_t30"]
    else:
        out["mean_peer_pma_t30"] = np.nan
        out["delta_vs_peer_pma_t30"] = np.nan

    if control_t30:
        out["mean_control_t30"] = wide[control_t30].mean(axis=1).values
        out["delta_vs_control_t30"] = out["expr_target"] - out["mean_control_t30"]
    else:
        out["mean_control_t30"] = np.nan
        out["delta_vs_control_t30"] = np.nan

    if pma_t10:
        out["mean_pma_t10"] = wide[pma_t10].mean(axis=1).values
        out["delta_vs_pma_t10"] = out["expr_target"] - out["mean_pma_t10"]
    else:
        out["mean_pma_t10"] = np.nan
        out["delta_vs_pma_t10"] = np.nan

    if pma_t60:
        out["mean_pma_t60"] = wide[pma_t60].mean(axis=1).values
        out["delta_vs_pma_t60"] = out["expr_target"] - out["mean_pma_t60"]
    else:
        out["mean_pma_t60"] = np.nan
        out["delta_vs_pma_t60"] = np.nan

    # residual-spike score: outlier to local peers + displaced from control_t30
    out["residual_spike_score"] = (
        out["delta_vs_peer_pma_t30"].abs().fillna(0.0)
        + 0.5 * out["delta_vs_control_t30"].abs().fillna(0.0)
    )

    # add feature metadata
    key_col = "feature_id" if "feature_id" in feature_meta.columns else feature_meta.columns[0]
    feature_meta = feature_meta.copy()
    feature_meta[key_col] = feature_meta[key_col].astype(str)
    out["feature_id"] = out["feature_id"].astype(str)
    out = out.merge(feature_meta, left_on="feature_id", right_on=key_col, how="left")

    top_spike = out.sort_values("residual_spike_score", ascending=False).head(100)
    top_up_peer = out.sort_values("delta_vs_peer_pma_t30", ascending=False).head(100)
    top_down_peer = out.sort_values("delta_vs_peer_pma_t30", ascending=True).head(100)

    target_state = state[state["sample_accession"] == TARGET_SAMPLE].iloc[0].to_dict()

    expr_summary = (
        master.groupby("sample_accession")["expression_value"]
        .agg(["count", "min", "median", "mean", "std", "max"])
        .reset_index()
        .rename(columns={"count": "n_rows"})
        .merge(sample_meta[["sample_accession", "condition", "time_min", "treatment"]], on="sample_accession", how="left")
    )

    expr_out = processed_root / "gse95575_9C_PMA_t30_expression_summary.csv"
    corr_out = processed_root / "gse95575_9C_PMA_t30_correlations.csv"
    spike_out = processed_root / "gse95575_9C_PMA_t30_top_spike_features.csv"
    up_out = processed_root / "gse95575_9C_PMA_t30_top_up_vs_peers.csv"
    down_out = processed_root / "gse95575_9C_PMA_t30_top_down_vs_peers.csv"
    log_out = logs_root / "gse95575_9C_PMA_t30_inspection_summary.json"

    expr_summary.to_csv(expr_out, index=False)
    corr_df.to_csv(corr_out, index=False)
    top_spike.to_csv(spike_out, index=False)
    top_up_peer.to_csv(up_out, index=False)
    top_down_peer.to_csv(down_out, index=False)

    summary = {
        "dataset_name": "gse95575",
        "target_sample": TARGET_SAMPLE,
        "target_condition": target_state.get("condition"),
        "target_treatment": target_state.get("treatment"),
        "target_time_min": float(target_state.get("time_min")),
        "target_residual_field": float(target_state.get("residual_field")),
        "top_correlated_samples_to_target": corr_df.head(6).to_dict(orient="records"),
    }

    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {expr_out}")
    print(f"[ok] wrote: {corr_out}")
    print(f"[ok] wrote: {spike_out}")
    print(f"[ok] wrote: {up_out}")
    print(f"[ok] wrote: {down_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
