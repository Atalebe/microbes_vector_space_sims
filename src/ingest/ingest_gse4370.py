from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_series_matrix(matrix_file: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    Keep repeated GEO fields instead of overwriting them.
    Repeated !Sample_description rows become sample_description__1, __2, etc.
    """
    sample_fields = {}
    repeat_count = {}

    for row in metadata_rows:
        key = row[0]
        vals = row[1:]
        if not key.startswith("!Sample_"):
            continue

        field = key.replace("!Sample_", "")
        repeat_count[field] = repeat_count.get(field, 0) + 1

        if repeat_count[field] == 1:
            colname = field
        else:
            colname = f"{field}__{repeat_count[field]}"

        sample_fields[colname] = vals

    # base sample count from geo accession
    geo_cols = [c for c in sample_fields if c.startswith("geo_accession")]
    if not geo_cols:
        raise ValueError("No !Sample_geo_accession field found.")
    sample_ids = sample_fields[geo_cols[0]]
    n = len(sample_ids)

    records = []
    for i in range(n):
        rec = {}
        for key, vals in sample_fields.items():
            rec[key] = vals[i] if i < len(vals) else None
        records.append(rec)

    df = pd.DataFrame(records)

    rename_map = {
        "geo_accession": "sample_accession",
        "title": "sample_title",
        "platform_id": "platform_id",
        "data_processing": "data_processing",
        "source_name_ch1": "source_name_ch1",
        "source_name_ch2": "source_name_ch2",
        "organism_ch1": "organism_ch1",
        "organism_ch2": "organism_ch2",
        "characteristics_ch1": "characteristics_ch1",
        "characteristics_ch2": "characteristics_ch2",
        "description": "sample_description",
    }
    df = df.rename(columns=rename_map)

    return df


def build_expression_long(table_lines: List[str], sample_meta: pd.DataFrame) -> pd.DataFrame:
    if not table_lines:
        raise ValueError("No expression table found in series matrix file.")

    header = next(csv.reader([table_lines[0]], delimiter="\t"))
    rows = [next(csv.reader([line], delimiter="\t")) for line in table_lines[1:]]

    expr = pd.DataFrame(rows, columns=header)
    expr = expr.rename(columns={expr.columns[0]: "feature_id"})

    sample_cols = [c for c in expr.columns if c in set(sample_meta["sample_accession"])]
    if not sample_cols:
        raise ValueError("Could not match sample columns in expression matrix.")

    long_df = expr.melt(
        id_vars=["feature_id"],
        value_vars=sample_cols,
        var_name="sample_accession",
        value_name="expression_value",
    )
    long_df["expression_value"] = pd.to_numeric(long_df["expression_value"], errors="coerce")
    long_df = long_df.dropna(subset=["feature_id"])

    return long_df


def parse_platform_table_from_soft(soft_file: str | Path) -> pd.DataFrame:
    lines = []
    in_platform = False

    with gzip.open(soft_file, "rt", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_platform = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_platform:
                lines.append(line)

    if not lines:
        return pd.DataFrame(columns=["feature_id", "ORF", "GENE_SYMBOL", "GENE_DESC"])

    header = next(csv.reader([lines[0]], delimiter="\t"))
    rows = [next(csv.reader([line], delimiter="\t")) for line in lines[1:]]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"ID": "feature_id"})
    return df


def extract_time_minutes_from_text(text: str) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s.lower() == "none":
        return None

    # handles "0 min", "15 min", "45min", "90 min"
    m = re.search(r"(\d+)\s*min\b", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # fallback for T1..T7 titles if needed
    t = re.search(r"\bT([1-7])\b", s, flags=re.IGNORECASE)
    if t:
        idx = int(t.group(1))
        mapping = {1: 0, 2: 15, 3: 30, 4: 45, 5: 60, 6: 75, 7: 90}
        return float(mapping[idx])

    return None


def derive_time_fields(sample_meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = sample_meta.copy()

    # gather all plausible text sources in priority order
    candidate_cols = [
        "sample_description",
        "sample_description__2",
        "sample_description__3",
        "source_name_ch2",
        "characteristics_ch2",
        "sample_title",
    ]

    time_vals = []
    chosen_source = []

    for _, row in out.iterrows():
        found = None
        found_col = None
        for col in candidate_cols:
            if col in out.columns:
                val = extract_time_minutes_from_text(row[col])
                if val is not None:
                    found = val
                    found_col = col
                    break
        time_vals.append(found)
        chosen_source.append(found_col)

    out["time_min"] = time_vals
    out["time_source_field"] = chosen_source

    def condition_label(x):
        if pd.isna(x):
            return None
        return f"recovery_{int(x)}min"

    out["condition"] = out["time_min"].apply(condition_label)
    out["time_order"] = out["time_min"]
    out["is_baseline"] = out["condition"] == cfg["dataset"]["baseline_label"]
    out["reference_channel"] = cfg["dataset"]["reference_channel"]
    out["value_definition"] = cfg["dataset"]["value_definition"]

    return out


def build_variable_dictionary() -> pd.DataFrame:
    records = [
        {"column_name": "feature_id", "description": "Microarray feature identifier"},
        {"column_name": "sample_accession", "description": "GEO sample accession"},
        {"column_name": "sample_title", "description": "GEO sample title"},
        {"column_name": "sample_description", "description": "Primary GEO sample description field"},
        {"column_name": "source_name_ch1", "description": "Channel 1 source"},
        {"column_name": "source_name_ch2", "description": "Channel 2 source"},
        {"column_name": "characteristics_ch1", "description": "Channel 1 characteristics"},
        {"column_name": "characteristics_ch2", "description": "Channel 2 characteristics"},
        {"column_name": "condition", "description": "Normalized recovery-time condition label"},
        {"column_name": "time_min", "description": "Recovery time in minutes"},
        {"column_name": "time_order", "description": "Ordered recovery-time index"},
        {"column_name": "time_source_field", "description": "Metadata field used to infer time_min"},
        {"column_name": "expression_value", "description": "Normalized log2(Cy5/Cy3) expression ratio"},
        {"column_name": "is_baseline", "description": "Flag for 0 minute recovery baseline"},
        {"column_name": "reference_channel", "description": "Reference channel used in array comparison"},
        {"column_name": "value_definition", "description": "Meaning of expression_value"},
    ]
    return pd.DataFrame(records)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    for _, p in cfg["paths"].items():
        if isinstance(p, str) and "/" in p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    sample_meta, expr_long = parse_series_matrix(cfg["paths"]["matrix_file"])
    sample_meta = derive_time_fields(sample_meta, cfg)

    feature_meta = parse_platform_table_from_soft(cfg["paths"]["soft_file"])
    variable_dict = build_variable_dictionary()

    master = expr_long.merge(sample_meta, on="sample_accession", how="left")
    if not feature_meta.empty:
        master = master.merge(feature_meta, on="feature_id", how="left")

    master.to_parquet(cfg["output"]["master_table"], index=False)
    sample_meta.to_csv(cfg["output"]["sample_metadata"], index=False)
    feature_meta.to_csv(cfg["output"]["feature_metadata"], index=False)
    variable_dict.to_csv(cfg["output"]["variable_dictionary"], index=False)

    timepoints = (
        sample_meta[["condition", "time_min"]]
        .dropna()
        .drop_duplicates()
        .sort_values("time_min")
        .to_dict(orient="records")
    )

    run_meta = {
        "dataset_name": cfg["dataset_name"],
        "series_accession": cfg["dataset"]["series_accession"],
        "platform_id": cfg["dataset"]["platform_id"],
        "organism": cfg["dataset"]["organism"],
        "n_samples": int(sample_meta["sample_accession"].nunique()),
        "n_timepoints": int(sample_meta["condition"].dropna().nunique()),
        "timepoints": timepoints,
        "n_features": int(master["feature_id"].nunique()),
        "n_rows_long_table": int(len(master)),
    }

    with open(cfg["output"]["run_metadata"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("[ok] wrote:", cfg["output"]["master_table"])
    print("[ok] wrote:", cfg["output"]["sample_metadata"])
    print("[ok] wrote:", cfg["output"]["feature_metadata"])
    print("[ok] wrote:", cfg["output"]["variable_dictionary"])
    print("[ok] wrote:", cfg["output"]["run_metadata"])
    print("[summary]", json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest GSE4370 GEO series matrix into a long-form master table.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
