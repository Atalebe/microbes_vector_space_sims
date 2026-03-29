from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_series_matrix(matrix_file: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series matrix file into:
    1. sample metadata table
    2. expression matrix in long form
    """
    metadata_rows: List[List[str]] = []
    table_lines: List[str] = []
    in_table = False

    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue

            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_") or line.startswith("!Series_"):
                metadata_rows.append(next(csv.reader([line], delimiter="\t")))

    sample_meta = build_sample_metadata(metadata_rows)
    expr_long = build_expression_long(table_lines, sample_meta)

    return sample_meta, expr_long


def build_sample_metadata(metadata_rows: List[List[str]]) -> pd.DataFrame:
    """
    Convert !Sample_* rows into a sample metadata table with one row per sample.
    """
    sample_fields: Dict[str, List[str]] = {}

    for row in metadata_rows:
        key = row[0]
        vals = row[1:]
        if key.startswith("!Sample_"):
            sample_fields[key] = vals

    # GEO series matrix often stores all sample columns aligned by position
    sample_ids = sample_fields.get("!Sample_geo_accession", [])
    n = len(sample_ids)

    records = []
    for i in range(n):
        rec = {}
        for key, vals in sample_fields.items():
            field = key.replace("!Sample_", "")
            rec[field] = vals[i] if i < len(vals) else None
        records.append(rec)

    df = pd.DataFrame(records)

    # normalize expected columns
    rename_map = {
        "geo_accession": "sample_accession",
        "title": "sample_title",
        "source_name_ch1": "source_name",
        "organism_ch1": "organism",
        "description": "description",
        "molecule_ch1": "molecule",
        "platform_id": "platform_id",
        "taxid_ch1": "taxid",
    }
    df = df.rename(columns=rename_map)

    # collect repeated characteristics rows if present
    characteristic_cols = [c for c in df.columns if c.startswith("characteristics_ch1")]
    if characteristic_cols:
        for c in characteristic_cols:
            df[c] = df[c].astype(str)

    return df


def build_expression_long(table_lines: List[str], sample_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the expression table section into long format.
    """
    if not table_lines:
        raise ValueError("No expression table found in series matrix file.")

    header = next(csv.reader([table_lines[0]], delimiter="\t"))
    rows = [next(csv.reader([line], delimiter="\t")) for line in table_lines[1:]]

    expr = pd.DataFrame(rows, columns=header)

    # First column is usually probe id
    probe_col = expr.columns[0]
    expr = expr.rename(columns={probe_col: "feature_id"})

    sample_cols = [c for c in expr.columns if c in set(sample_meta["sample_accession"])]
    if not sample_cols:
        raise ValueError("Could not match expression columns to sample accessions.")

    long_df = expr.melt(
        id_vars=["feature_id"],
        value_vars=sample_cols,
        var_name="sample_accession",
        value_name="expression_value",
    )

    long_df["expression_value"] = pd.to_numeric(long_df["expression_value"], errors="coerce")

    long_df = long_df.merge(sample_meta, on="sample_accession", how="left")

    return long_df


def derive_condition_fields(sample_meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add condition labels and ordering from config and sample titles.
    """
    title_map = cfg["sample_title_map"]
    condition_order = cfg["condition_order"]

    out = sample_meta.copy()

    out["condition"] = out["sample_title"].map(title_map)

    # fallback using source_name if needed
    missing = out["condition"].isna()
    if missing.any():
        out.loc[missing, "condition"] = out.loc[missing, "source_name"].map(title_map)

    out["condition_order"] = out["condition"].map(condition_order)

    out["replicate"] = (
        out["sample_title"]
        .astype(str)
        .str.extract(r"(\d+)$", expand=False)
        .astype("Int64")
    )

    out["is_baseline"] = out["condition"] == cfg["dataset"]["baseline_label"]
    out["stressor"] = cfg["dataset"]["stressor"]
    out["stressor_concentration"] = cfg["dataset"]["stressor_concentration"]

    return out


def build_variable_dictionary() -> pd.DataFrame:
    records = [
        {"column_name": "feature_id", "description": "Microarray feature or probe identifier"},
        {"column_name": "sample_accession", "description": "GEO sample accession"},
        {"column_name": "sample_title", "description": "Sample title from GEO"},
        {"column_name": "condition", "description": "Normalized physiological condition label"},
        {"column_name": "condition_order", "description": "Ordered condition index from baseline through recovery"},
        {"column_name": "replicate", "description": "Replicate number parsed from sample title"},
        {"column_name": "expression_value", "description": "RMA-normalized expression value from GEO series matrix"},
        {"column_name": "is_baseline", "description": "Boolean flag for baseline normal-growth state"},
        {"column_name": "stressor", "description": "Named stressor from experiment"},
        {"column_name": "stressor_concentration", "description": "Stress concentration from protocol"},
        {"column_name": "organism", "description": "Organism from GEO metadata"},
        {"column_name": "platform_id", "description": "GEO platform accession"},
        {"column_name": "source_name", "description": "GEO source-name field"},
        {"column_name": "description", "description": "GEO sample description"},
    ]
    return pd.DataFrame(records)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    for _, p in cfg["paths"].items():
        if isinstance(p, str) and ("/" in p or p.endswith(".gz")):
            parent = Path(p).parent
            parent.mkdir(parents=True, exist_ok=True)

    sample_meta, expr_long = parse_series_matrix(cfg["paths"]["matrix_file"])
    sample_meta = derive_condition_fields(sample_meta, cfg)

    master = expr_long.drop(columns=[c for c in expr_long.columns if c in sample_meta.columns and c != "sample_accession"])
    master = master.merge(sample_meta, on="sample_accession", how="left")

    variable_dict = build_variable_dictionary()

    # outputs
    master.to_parquet(cfg["output"]["master_table"], index=False)
    sample_meta.to_csv(cfg["output"]["sample_metadata"], index=False)
    variable_dict.to_csv(cfg["output"]["variable_dictionary"], index=False)

    run_meta = {
        "dataset_name": cfg["dataset_name"],
        "series_accession": cfg["dataset"]["series_accession"],
        "platform_id": cfg["dataset"]["platform_id"],
        "organism": cfg["dataset"]["organism"],
        "n_samples": int(sample_meta["sample_accession"].nunique()),
        "n_conditions": int(sample_meta["condition"].nunique()),
        "conditions": sorted(sample_meta["condition"].dropna().unique().tolist(), key=lambda x: cfg["condition_order"][x]),
        "n_features": int(master["feature_id"].nunique()),
        "n_rows_long_table": int(len(master)),
    }

    with open(cfg["output"]["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("[ok] wrote:", cfg["output"]["master_table"])
    print("[ok] wrote:", cfg["output"]["sample_metadata"])
    print("[ok] wrote:", cfg["output"]["variable_dictionary"])
    print("[ok] wrote:", cfg["output"]["run_metadata"])
    print("[summary]", json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest GSE36599 GEO series matrix into a long-form master table.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
