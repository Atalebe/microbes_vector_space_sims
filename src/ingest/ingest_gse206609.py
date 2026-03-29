from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_count_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", compression="gzip")
    df.columns = [str(c).strip() for c in df.columns]

    gene_col = None
    for c in df.columns:
        if c.lower() == "geneid":
            gene_col = c
            break
    if gene_col is None:
        gene_col = df.columns[0]

    count_col = None
    for c in df.columns[::-1]:
        cl = c.lower()
        if cl not in {"geneid", "chr", "start", "end", "strand", "length"}:
            count_col = c
            break
    if count_col is None:
        raise ValueError(f"Could not identify count column in {path}")

    out = df[[gene_col, count_col]].copy()
    out.columns = ["feature_id", "raw_count"]
    out["feature_id"] = out["feature_id"].astype(str)
    return out


def parse_count_filename(path: Path) -> dict:
    name = path.name

    # examples:
    # GSM6257892_30min_LEVO_1_counts.txt.gz
    # GSM6257904_ROC_LEVO_1_counts.txt.gz
    m = re.match(
        r"^(GSM\d+)_(30min|ROC)_([A-Za-z\-]+)_([0-9]+)_counts\.txt(?:\.gz)?$",
        name,
        flags=re.I,
    )
    if not m:
        raise ValueError(f"Unexpected count filename format: {name}")

    sample_accession, phase_token, treatment_token, rep = m.groups()

    phase = "treatment" if phase_token.lower() == "30min" else "recovery"
    treatment_token = treatment_token.upper()
    if treatment_token == "-":
        treatment = "none"
    else:
        treatment = treatment_token.lower()

    phase_order = 0 if phase == "treatment" else 1
    condition = f"{treatment}_{phase}"

    return {
        "sample_accession": sample_accession,
        "sample_title": f"{treatment_token}, {phase.upper()}, REP{rep}",
        "treatment": treatment,
        "phase": phase,
        "phase_token": phase_token,
        "replicate": int(rep),
        "condition": condition,
        "phase_order": phase_order,
        "strain": "K12 MG1655",
        "genotype": "MG1655 pstS-glmS::cat-Mu gyrA-FLAG-kan",
        "media": "M9 glucose" if phase == "treatment" else "LB",
        "time_label": "30_mins_treatment" if phase == "treatment" else "30_mins_recovery",
        "time_minutes_nominal": 30,
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    for k in ["interim_dir", "processed_dir", "logs_dir"]:
        ensure_dir(cfg["paths"][k])

    raw_counts_dir = Path(cfg["paths"]["raw_counts_dir"])
    count_files = sorted(raw_counts_dir.glob("*_counts.txt.gz"))
    if not count_files:
        count_files = sorted(raw_counts_dir.glob("*_counts.txt"))
    if not count_files:
        raise FileNotFoundError("No count files found in raw_counts directory.")

    sample_rows = []
    long_tables = []
    feature_union = set()

    for f in count_files:
        meta = parse_count_filename(f)
        sample_rows.append(meta)

        counts = read_count_file(f)
        counts["sample_accession"] = meta["sample_accession"]
        long_tables.append(counts)
        feature_union.update(counts["feature_id"].tolist())

    sample_meta = pd.DataFrame(sample_rows).sort_values(
        ["phase_order", "treatment", "replicate", "sample_accession"]
    ).reset_index(drop=True)

    master = pd.concat(long_tables, ignore_index=True)
    master = master.merge(sample_meta, on="sample_accession", how="left")

    feature_meta = pd.DataFrame({"feature_id": sorted(feature_union)})

    variable_dictionary = pd.DataFrame(
        [
            {"variable": "feature_id", "description": "Gene or feature identifier from featureCounts"},
            {"variable": "raw_count", "description": "Raw RNA-seq count"},
            {"variable": "sample_accession", "description": "GEO sample accession"},
            {"variable": "treatment", "description": "Drug treatment label: levo, moxi, mmc, none"},
            {"variable": "phase", "description": "Experimental phase: treatment or recovery"},
            {"variable": "replicate", "description": "Biological replicate index"},
            {"variable": "condition", "description": "Combined treatment-phase condition"},
            {"variable": "media", "description": "Media context from GEO metadata"},
            {"variable": "time_label", "description": "Nominal time label parsed from GEO design"},
            {"variable": "time_minutes_nominal", "description": "Nominal time in minutes for the labeled phase"},
        ]
    )

    master.to_parquet(cfg["output"]["master_table"], index=False)
    sample_meta.to_csv(cfg["output"]["sample_metadata"], index=False)
    feature_meta.to_csv(cfg["output"]["feature_metadata"], index=False)
    variable_dictionary.to_csv(cfg["output"]["variable_dictionary"], index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "n_samples": int(sample_meta["sample_accession"].nunique()),
        "conditions": sorted(sample_meta["condition"].dropna().unique().tolist()),
        "phases": sorted(sample_meta["phase"].dropna().unique().tolist()),
        "treatments": sorted(sample_meta["treatment"].dropna().unique().tolist()),
        "n_rows_long_table": int(len(master)),
        "n_features": int(master["feature_id"].nunique()),
    }

    with open(f"{cfg['paths']['logs_dir']}/ingest_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", cfg["output"]["master_table"])
    print("[ok] wrote:", cfg["output"]["sample_metadata"])
    print("[ok] wrote:", cfg["output"]["feature_metadata"])
    print("[ok] wrote:", cfg["output"]["variable_dictionary"])
    print("[ok] wrote:", f"{cfg['paths']['logs_dir']}/ingest_run_meta.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest GSE206609 from supplementary count files.")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
