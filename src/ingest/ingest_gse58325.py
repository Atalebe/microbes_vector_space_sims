#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_root = Path(cfg.get("raw_root", "data/raw/gse58325"))
    interim_root = Path(cfg.get("interim_root", "data/interim/gse58325"))
    logs_root = Path(cfg.get("logs_root", "results/logs/gse58325"))

    cfg.setdefault("dataset_name", "gse58325")
    cfg.setdefault("series_matrix", str(raw_root / "GSE58325_series_matrix.txt.gz"))
    cfg.setdefault("raw_dir", str(raw_root / "raw"))
    cfg.setdefault("interim_root", str(interim_root))
    cfg.setdefault("logs_root", str(logs_root))
    return cfg


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def parse_series_header(series_matrix_path: str) -> Dict[str, List[str]]:
    """
    Parse quoted GEO series-matrix header lines such as:
    !Sample_title   "..." "..." "..."
    !Sample_geo_accession   "GSM..." ...
    """
    header_map: Dict[str, List[str]] = {}
    patterns = [
        "!Sample_title",
        "!Sample_geo_accession",
        "!Sample_description",
        "!Sample_characteristics_ch1",
        "!Sample_characteristics_ch2",
        "!Sample_source_name_ch1",
        "!Sample_source_name_ch2",
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


def build_sample_metadata(header_map: Dict[str, List[str]]) -> pd.DataFrame:
    accessions = header_map.get("!Sample_geo_accession", [])
    titles = header_map.get("!Sample_title", [""] * len(accessions))
    descs = header_map.get("!Sample_description", [""] * len(accessions))
    src2 = header_map.get("!Sample_source_name_ch2", [""] * len(accessions))

    n = len(accessions)
    if n == 0:
        raise ValueError("No sample accessions found in GSE58325 series matrix header.")

    rows = []
    for i in range(n):
        title = titles[i] if i < len(titles) else ""
        desc = descs[i] if i < len(descs) else ""
        source2 = src2[i] if i < len(src2) else ""
        text = " ".join([title, desc, source2]).lower()

        # Very forgiving time parsing.
        minute_match = re.search(r"(\d+)\s*min", text)
        time_min = float(minute_match.group(1)) if minute_match else None

        # Conditioning / phase parsing kept intentionally generic because this
        # dataset may encode preconditioning or selection in filenames/titles.
        conditioning_class = "stress_defined"
        stage_label = "unknown"
        if "recovery" in text:
            stage_label = "recovery"
        elif "stress" in text:
            stage_label = "stress"
        elif "control" in text or "untreated" in text:
            stage_label = "control"

        rows.append(
            {
                "sample_accession": accessions[i],
                "sample_title": title,
                "sample_description": desc,
                "source_name_ch2": source2,
                "conditioning_class": conditioning_class,
                "stage_label": stage_label,
                "time_min": time_min,
            }
        )

    sample_meta = pd.DataFrame(rows)
    sample_meta["time_order"] = sample_meta["time_min"].rank(method="dense").where(
        sample_meta["time_min"].notna(), None
    )
    return sample_meta


def discover_fpkm_files(raw_dir: str) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.genes.fpkm_tracking.gz"))
    if not files:
        raise FileNotFoundError(
            f"No *.genes.fpkm_tracking.gz files found in {raw_dir}. "
            "Extract GSE58325_RAW.tar into this directory first."
        )

    rows = []
    for fp in files:
        m = re.match(r"(GSM\d+)_", fp.name)
        if not m:
            continue
        rows.append({"sample_accession": m.group(1), "file_path": str(fp), "file_name": fp.name})
    out = pd.DataFrame(rows).drop_duplicates("sample_accession")
    if out.empty:
        raise ValueError("Could not map any GSE58325 raw files to GSM accessions.")
    return out


def read_one_fpkm(path: str, sample_accession: str) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        df = pd.read_csv(f, sep="\t")

    required_any = ["tracking_id", "gene_id", "gene_short_name", "FPKM"]
    missing_all = all(col not in df.columns for col in required_any)
    if missing_all:
        raise ValueError(f"{path} does not look like a cufflinks genes.fpkm_tracking file.")

    # Prefer gene_id as stable feature identifier. Fall back if needed.
    if "gene_id" in df.columns:
        feature_id = df["gene_id"].astype(str)
    elif "tracking_id" in df.columns:
        feature_id = df["tracking_id"].astype(str)
    else:
        feature_id = pd.Series([f"feature_{i}" for i in range(len(df))], index=df.index)

    gene_symbol = df["gene_short_name"].astype(str) if "gene_short_name" in df.columns else ""
    expr = pd.to_numeric(df.get("FPKM"), errors="coerce")

    out = pd.DataFrame(
        {
            "sample_accession": sample_accession,
            "feature_id": feature_id,
            "gene_symbol": gene_symbol,
            "expression_value": expr,
            "expression_unit": "FPKM",
        }
    )
    return out


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset_name"]
    series_matrix = cfg["series_matrix"]
    raw_dir = cfg["raw_dir"]
    interim_root = Path(cfg["interim_root"])
    logs_root = Path(cfg["logs_root"])

    ensure_dirs(interim_root, logs_root)

    header_map = parse_series_header(series_matrix)
    sample_meta = build_sample_metadata(header_map)
    file_map = discover_fpkm_files(raw_dir)
    sample_meta = sample_meta.merge(file_map, on="sample_accession", how="left")

    missing_files = sample_meta["file_path"].isna().sum()
    if missing_files > 0:
        print(f"[warn] {missing_files} samples in series header have no matched extracted FPKM file.")

    parts = []
    feature_meta_parts = []

    for row in sample_meta.dropna(subset=["file_path"]).itertuples(index=False):
        df = read_one_fpkm(row.file_path, row.sample_accession)
        parts.append(df)

        feature_meta_parts.append(
            df[["feature_id", "gene_symbol"]].drop_duplicates().assign(source_file=row.file_name)
        )

    if not parts:
        raise ValueError("No usable GSE58325 FPKM tables were ingested.")

    master = pd.concat(parts, ignore_index=True)
    feature_meta = pd.concat(feature_meta_parts, ignore_index=True).drop_duplicates("feature_id")

    variable_dictionary = pd.DataFrame(
        [
            ["sample_accession", "GEO sample accession"],
            ["feature_id", "Gene or transcript identifier from fpkm_tracking"],
            ["gene_symbol", "Gene short name from fpkm_tracking"],
            ["expression_value", "Per-sample FPKM value"],
            ["expression_unit", "Expression unit, here FPKM"],
            ["conditioning_class", "Broad conditioning label retained from series header"],
            ["stage_label", "Broad stage label retained from series header"],
            ["time_min", "Parsed minute label when available"],
        ],
        columns=["variable", "description"],
    )

    master = master.merge(
        sample_meta.drop(columns=["file_path", "file_name"], errors="ignore"),
        on="sample_accession",
        how="left",
    )

    master_out = interim_root / "master_table.parquet"
    sample_out = interim_root / "sample_metadata.csv"
    feature_out = interim_root / "feature_metadata.csv"
    vardict_out = interim_root / "variable_dictionary.csv"
    meta_out = logs_root / "ingest_run_meta.json"

    master.to_parquet(master_out, index=False)
    sample_meta.drop(columns=["file_path"], errors="ignore").to_csv(sample_out, index=False)
    feature_meta.to_csv(feature_out, index=False)
    variable_dictionary.to_csv(vardict_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "n_samples": int(sample_meta["sample_accession"].nunique()),
        "n_features": int(master["feature_id"].nunique()),
        "n_rows_long_table": int(len(master)),
        "conditioning_classes": sorted(sample_meta["conditioning_class"].dropna().astype(str).unique().tolist()),
        "stage_labels": sorted(sample_meta["stage_label"].dropna().astype(str).unique().tolist()),
        "expression_unit": "FPKM",
        "series_matrix_data_used": False,
        "raw_archive_data_used": True,
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
