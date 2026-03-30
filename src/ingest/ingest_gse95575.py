#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


SAMPLE_COL_RE = re.compile(r"^\d+[A-Z]_(MG|Hg|PMA)_t\d+$")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_root = Path(cfg.get("raw_root", "data/raw/gse95575"))
    interim_root = Path(cfg.get("interim_root", "data/interim/gse95575"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse95575"))

    cfg.setdefault("dataset_name", "gse95575")
    cfg.setdefault("series_matrix", str(raw_root / "GSE95575_series_matrix.txt.gz"))
    cfg.setdefault(
        "counts_matrix",
        str(raw_root / "suppl" / "GSE95575_GEO_Hg_RNA-Seq_raw_counts_matrix.csv.gz"),
    )
    cfg.setdefault("interim_root", str(interim_root))
    cfg.setdefault("logs_root", str(logs_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def parse_series_header(series_matrix_path: str) -> Dict[str, List[str]]:
    header_map: Dict[str, List[str]] = {}
    patterns = [
        "!Sample_title",
        "!Sample_geo_accession",
        "!Sample_description",
        "!Sample_characteristics_ch1",
        "!Series_supplementary_file",
    ]

    with gzip.open(series_matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            for pat in patterns:
                if line.startswith(pat):
                    vals = re.findall(r'"([^"]*)"', line)
                    header_map[pat] = vals
    return header_map


def build_series_sample_metadata(header_map: Dict[str, List[str]]) -> pd.DataFrame:
    accessions = header_map.get("!Sample_geo_accession", [])
    titles = header_map.get("!Sample_title", [""] * len(accessions))
    descs = header_map.get("!Sample_description", [""] * len(accessions))
    chars = header_map.get("!Sample_characteristics_ch1", [""] * len(accessions))

    rows = []
    for acc, title, desc, char in zip(accessions, titles, descs, chars):
        title_l = str(title).lower()
        char_l = str(char).lower()

        if "hg" in title_l or "hgcl2" in title_l or "hg" in char_l:
            treatment = "hgcl2"
        elif "pma" in title_l or "pma" in char_l:
            treatment = "pma"
        else:
            treatment = "control"

        m = re.search(r"t(\d+)", title_l)
        if m is None:
            m = re.search(r"t(\d+)", char_l)
        time_min = float(m.group(1)) if m else None

        rep_match = re.match(r"\d+([A-Z])_", str(title).strip())
        replicate = rep_match.group(1) if rep_match else None

        rows.append(
            {
                "sample_accession": acc,
                "series_sample_title": title,
                "sample_description": desc,
                "series_characteristic": char,
                "treatment": treatment,
                "time_min": time_min,
                "replicate": replicate,
            }
        )

    meta = pd.DataFrame(rows)
    return meta


def read_counts_matrix(counts_matrix_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(counts_matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Counts matrix not found: {counts_matrix_path}")

    # We now know from inspection that the real header is row 4, zero-index skiprows=3
    df = pd.read_csv(
        path,
        compression="gzip",
        sep=",",
        skiprows=3,
        dtype=str,
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "feature",
        "start",
        "end",
        "strand",
        "gene_ID",
        "gene_name",
        "product_description",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"GSE95575 counts matrix missing required columns: {missing}")

    sample_cols = [c for c in df.columns if SAMPLE_COL_RE.match(c)]
    if len(sample_cols) != 30:
        raise ValueError(
            f"Expected 30 raw-matrix sample columns, found {len(sample_cols)}: {sample_cols[:10]}"
        )

    # Drop junk rows if any
    df["gene_ID"] = df["gene_ID"].astype(str).str.strip()
    bad_mask = (
        df["gene_ID"].isna()
        | (df["gene_ID"] == "")
        | (df["gene_ID"].str.lower() == "nan")
        | (df["gene_ID"].str.lower() == "gene_id")
    )
    df = df.loc[~bad_mask].copy()

    feature_meta = df[
        ["gene_ID", "gene_name", "product_description", "feature", "start", "end", "strand"]
    ].drop_duplicates().rename(columns={"gene_ID": "feature_id"})

    long_df = df.melt(
        id_vars=["gene_ID"],
        value_vars=sample_cols,
        var_name="raw_sample_label",
        value_name="expression_value",
    ).rename(columns={"gene_ID": "feature_id"})

    long_df["feature_id"] = long_df["feature_id"].astype(str).str.strip()
    long_df["expression_value"] = pd.to_numeric(long_df["expression_value"], errors="coerce")
    long_df["expression_unit"] = "raw_counts"

    # Parse raw sample labels like 5A_Hg_t10
    parsed = long_df["raw_sample_label"].str.extract(
        r"^(?P<sample_block>\d+)(?P<replicate>[A-Z])_(?P<treat_code>MG|Hg|PMA)_t(?P<time_min>\d+)$"
    )
    if parsed.isna().any().any():
        bad = long_df.loc[parsed.isna().any(axis=1), "raw_sample_label"].unique().tolist()
        raise ValueError(f"Could not parse some raw sample labels: {bad[:10]}")

    treat_map = {"MG": "control", "Hg": "hgcl2", "PMA": "pma"}
    parsed["treatment"] = parsed["treat_code"].map(treat_map)
    parsed["time_min"] = parsed["time_min"].astype(float)
    parsed["condition"] = (
        parsed["treatment"].astype(str)
        + "_t"
        + parsed["time_min"].astype(int).astype(str)
    )
    parsed["phase"] = parsed["time_min"].map(lambda x: "baseline" if x == 0 else "recovery")
    parsed["time_order"] = parsed["time_min"]

    sample_meta = (
        parsed.assign(raw_sample_label=long_df["raw_sample_label"].values)
        .drop_duplicates(subset=["raw_sample_label"])
        .copy()
    )

    # Canonical accession defined by raw label for internal use
    sample_meta["sample_accession"] = sample_meta["raw_sample_label"]

    long_df = pd.concat([long_df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    long_df["sample_accession"] = long_df["raw_sample_label"]

    master = long_df[
        [
            "feature_id",
            "sample_accession",
            "expression_value",
            "expression_unit",
            "raw_sample_label",
            "replicate",
            "treatment",
            "time_min",
            "condition",
            "phase",
            "time_order",
        ]
    ].copy()

    return master, sample_meta, feature_meta


def attach_series_accessions(
    raw_sample_meta: pd.DataFrame,
    series_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align raw-label metadata with series-matrix metadata by treatment/time/replicate.
    This gives a GEO accession crosswalk but does not control the ingestion itself.
    """
    left = raw_sample_meta.copy()
    right = series_meta.copy()

    left["replicate"] = left["replicate"].astype(str)
    right["replicate"] = right["replicate"].astype(str)

    merged = left.merge(
        right[
            [
                "sample_accession",
                "series_sample_title",
                "sample_description",
                "treatment",
                "time_min",
                "replicate",
            ]
        ],
        on=["treatment", "time_min", "replicate"],
        how="left",
        suffixes=("", "_geo"),
    )

    merged = merged.rename(columns={"sample_accession_geo": "geo_accession"})
    return merged


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    series_matrix = cfg["series_matrix"]
    counts_matrix = cfg["counts_matrix"]
    interim_root = Path(cfg["interim_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(interim_root, logs_root)

    header_map = parse_series_header(series_matrix)
    series_meta = build_series_sample_metadata(header_map)
    master, raw_sample_meta, feature_meta = read_counts_matrix(counts_matrix)
    sample_meta = attach_series_accessions(raw_sample_meta, series_meta)

    master = master.merge(
        sample_meta[
            [
                "raw_sample_label",
                "series_sample_title",
                "sample_description",
                "geo_accession",
            ]
        ],
        on="raw_sample_label",
        how="left",
    )

    variable_dictionary = pd.DataFrame(
        [
            ["feature_id", "Gene identifier from raw counts matrix, using gene_ID"],
            ["sample_accession", "Internal sample key from raw matrix label"],
            ["geo_accession", "Matched GEO accession from series metadata when available"],
            ["raw_sample_label", "Raw matrix sample label, e.g. 5A_Hg_t10"],
            ["expression_value", "Raw count value"],
            ["expression_unit", "Expression unit, raw counts"],
            ["treatment", "control, hgcl2, or pma"],
            ["time_min", "Time in minutes after exposure"],
            ["condition", "Condition label combining treatment and time"],
            ["replicate", "Biological replicate letter from raw matrix label"],
        ],
        columns=["variable", "description"],
    )

    master_out = interim_root / "master_table.parquet"
    sample_out = interim_root / "sample_metadata.csv"
    feature_out = interim_root / "feature_metadata.csv"
    vardict_out = interim_root / "variable_dictionary.csv"
    meta_out = logs_root / "ingest_run_meta.json"

    master.to_parquet(master_out, index=False)
    sample_meta.to_csv(sample_out, index=False)
    feature_meta.to_csv(feature_out, index=False)
    variable_dictionary.to_csv(vardict_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "n_samples": int(master["sample_accession"].nunique()),
        "n_features": int(master["feature_id"].nunique()),
        "n_rows_long_table": int(len(master)),
        "conditions": sorted(master["condition"].dropna().astype(str).unique().tolist()),
        "treatments": sorted(master["treatment"].dropna().astype(str).unique().tolist()),
        "timepoints_min": sorted(master["time_min"].dropna().astype(float).unique().tolist()),
        "expression_unit": "raw_counts",
        "raw_matrix_sample_labels_used": True,
        "geo_accession_crosswalk_attached": bool(sample_meta["geo_accession"].notna().any()),
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {master_out}")
    print(f"[ok] wrote: {sample_out}")
    print(f"[ok] wrote: {feature_out}")
    print(f"[ok] wrote: {vardict_out}")
    print(f"[ok] wrote: {meta_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
