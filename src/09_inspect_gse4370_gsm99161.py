from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.decomposition import PCA


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)

    master = pd.read_parquet(cfg["output"]["master_table"])
    sample_meta = pd.read_csv(cfg["output"]["sample_metadata"])
    feature_meta = pd.read_csv(cfg["output"]["feature_metadata"])

    # force merge key consistency
    master["feature_id"] = master["feature_id"].astype(str)
    feature_meta["feature_id"] = feature_meta["feature_id"].astype(str)

    target = "GSM99161"
    neighbors = ["GSM99160", "GSM99161", "GSM99162"]

    # 1. Expression summary and missingness
    expr_summary = (
        master.groupby("sample_accession", as_index=False)
        .agg(
            n_rows=("expression_value", "size"),
            n_missing=("expression_value", lambda x: int(x.isna().sum())),
            expr_min=("expression_value", "min"),
            expr_median=("expression_value", "median"),
            expr_mean=("expression_value", "mean"),
            expr_std=("expression_value", "std"),
            expr_max=("expression_value", "max"),
        )
        .merge(
            sample_meta[["sample_accession", "condition", "time_min"]],
            on="sample_accession",
            how="left",
        )
        .sort_values("time_min")
    )

    # 2. Wide matrix and correlations
    wide = master.pivot_table(
        index="feature_id",
        columns="sample_accession",
        values="expression_value",
        aggfunc="mean",
    )

    corr = wide.corr(method="spearman")
    corr_target = corr[target].sort_values(ascending=False).rename("spearman_corr_to_GSM99161").reset_index()
    corr_target = corr_target.rename(columns={"sample_accession": "other_sample"})

    # 3. Absolute changes around 60 min
    sub = wide[neighbors].copy()
    sub["delta_45_60"] = sub["GSM99161"] - sub["GSM99160"]
    sub["delta_60_75"] = sub["GSM99162"] - sub["GSM99161"]
    sub["abs_delta_45_60"] = sub["delta_45_60"].abs()
    sub["abs_delta_60_75"] = sub["delta_60_75"].abs()
    sub["abs_turning_sum"] = sub["abs_delta_45_60"] + sub["abs_delta_60_75"]
    turning = sub.reset_index()

    if "feature_id" not in turning.columns:
        turning = turning.rename(columns={turning.columns[0]: "feature_id"})
    turning["feature_id"] = turning["feature_id"].astype(str)

    turning = turning.merge(feature_meta, on="feature_id", how="left")

    top_45_60 = turning.sort_values("abs_delta_45_60", ascending=False).head(25)
    top_60_75 = turning.sort_values("abs_delta_60_75", ascending=False).head(25)
    top_turning = turning.sort_values("abs_turning_sum", ascending=False).head(25)

    # 4. PCA loadings on top variable features
    ordered_samples = sample_meta.sort_values("time_order")["sample_accession"].tolist()
    wide_ord = wide[ordered_samples]
    feature_var = wide_ord.var(axis=1).sort_values(ascending=False)
    top_features = feature_var.head(1000).index.tolist()

    X = wide_ord.loc[top_features].T.fillna(wide_ord.loc[top_features].T.mean())
    pca = PCA(n_components=3)
    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=top_features,
        columns=["PC1_loading", "PC2_loading", "PC3_loading"],
    ).reset_index().rename(columns={"index": "feature_id"})
    loadings["feature_id"] = loadings["feature_id"].astype(str)
    loadings = loadings.merge(feature_meta, on="feature_id", how="left")

    top_pc2_pos = loadings.sort_values("PC2_loading", ascending=False).head(25)
    top_pc2_neg = loadings.sort_values("PC2_loading", ascending=True).head(25)

    # 5. Save
    expr_summary.to_csv(f"{processed_dir}/gse4370_gsm99161_expression_summary.csv", index=False)
    corr_target.to_csv(f"{processed_dir}/gse4370_gsm99161_correlations.csv", index=False)
    top_45_60.to_csv(f"{processed_dir}/gse4370_gsm99161_top_delta_45_60.csv", index=False)
    top_60_75.to_csv(f"{processed_dir}/gse4370_gsm99161_top_delta_60_75.csv", index=False)
    top_turning.to_csv(f"{processed_dir}/gse4370_gsm99161_top_turning_features.csv", index=False)
    top_pc2_pos.to_csv(f"{processed_dir}/gse4370_top_pc2_positive_loadings.csv", index=False)
    top_pc2_neg.to_csv(f"{processed_dir}/gse4370_top_pc2_negative_loadings.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "target_sample": target,
        "target_condition": sample_meta.loc[sample_meta["sample_accession"] == target, "condition"].iloc[0],
        "target_time_min": float(sample_meta.loc[sample_meta["sample_accession"] == target, "time_min"].iloc[0]),
        "top_correlated_samples_to_target": corr_target.head(4).to_dict(orient="records"),
    }

    with open(f"{logs_dir}/gsm99161_inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse4370_gsm99161_expression_summary.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_gsm99161_correlations.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_gsm99161_top_delta_45_60.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_gsm99161_top_delta_60_75.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_gsm99161_top_turning_features.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_top_pc2_positive_loadings.csv")
    print("[ok] wrote:", f"{processed_dir}/gse4370_top_pc2_negative_loadings.csv")
    print("[ok] wrote:", f"{logs_dir}/gsm99161_inspection_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect GSM99161 as a possible secondary peak or noisy sample.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
