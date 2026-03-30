from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path("data/processed/gse188790")
    sample_meta_path = base / "gse188790_sample_metadata.csv"
    matrix_path = base / "gse188790_gene_count_matrix.csv"
    summary_path = base / "gse188790_ingest_summary.json"

    if not sample_meta_path.exists():
        raise SystemExit("Missing gse188790_sample_metadata.csv. Run ingest_gse188790.py first.")
    if not matrix_path.exists():
        raise SystemExit("Missing gse188790_gene_count_matrix.csv. Run ingest_gse188790.py first.")

    sample_meta = pd.read_csv(sample_meta_path)
    matrix = pd.read_csv(matrix_path)
    ingest_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    expected_samples = {"UnagedA", "UnagedB", "AgedA", "AgedB"}
    observed_samples = set(sample_meta["sample_name"].astype(str).tolist())

    expected_age_classes = {"unaged", "aged"}
    observed_age_classes = set(sample_meta["age_class"].astype(str).tolist())

    expected_replicates = {1, 2}
    observed_replicates = set(pd.to_numeric(sample_meta["replicate"], errors="coerce").dropna().astype(int).tolist())

    matrix_sample_cols = [c for c in matrix.columns if c != "gene_id"]

    per_sample_library_sizes = {}
    if matrix_sample_cols:
        for c in matrix_sample_cols:
            per_sample_library_sizes[c] = int(pd.to_numeric(matrix[c], errors="coerce").fillna(0).sum())

    report = {
        "dataset": "GSE188790",
        "n_samples_observed": int(len(sample_meta)),
        "n_genes_observed": int(len(matrix)),
        "expected_samples_present": sorted(list(expected_samples & observed_samples)),
        "missing_expected_samples": sorted(list(expected_samples - observed_samples)),
        "unexpected_samples": sorted(list(observed_samples - expected_samples)),
        "expected_age_classes_present": sorted(list(expected_age_classes & observed_age_classes)),
        "missing_expected_age_classes": sorted(list(expected_age_classes - observed_age_classes)),
        "observed_replicates": sorted(list(observed_replicates)),
        "missing_expected_replicates": sorted(list(expected_replicates - observed_replicates)),
        "matrix_sample_columns": matrix_sample_cols,
        "matrix_matches_sample_metadata": sorted(matrix_sample_cols) == sorted(sample_meta["sample_name"].astype(str).tolist()),
        "per_sample_library_sizes": per_sample_library_sizes,
        "ready_for_state_table": (
            len(expected_samples - observed_samples) == 0
            and len(expected_age_classes - observed_age_classes) == 0
            and len(expected_replicates - observed_replicates) == 0
            and len(matrix) > 0
        ),
    }

    (base / "gse188790_audit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
