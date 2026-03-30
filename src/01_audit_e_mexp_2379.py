from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path("data/processed/e_mexp_2379")
    base.mkdir(parents=True, exist_ok=True)

    expected_path = base / "e_mexp_2379_expected_design.csv"
    sdrf_path = base / "e_mexp_2379_sample_table_from_sdrf.csv"
    cand_path = base / "e_mexp_2379_candidate_non_magetab_files.csv"
    ingest_summary_path = base / "e_mexp_2379_ingest_summary.json"

    if not expected_path.exists():
        raise SystemExit(
            "Missing e_mexp_2379_expected_design.csv. Run src/ingest/ingest_e_mexp_2379.py first."
        )

    expected = pd.read_csv(expected_path)
    sdrf = pd.read_csv(sdrf_path) if sdrf_path.exists() else pd.DataFrame()
    candidates = pd.read_csv(cand_path) if cand_path.exists() else pd.DataFrame()
    ingest_summary = json.loads(ingest_summary_path.read_text()) if ingest_summary_path.exists() else {}

    if not sdrf.empty and {"age", "time_min", "replicate"}.issubset(sdrf.columns):
        obs = sdrf[["age", "time_min", "replicate"]].copy()
        obs["age"] = obs["age"].astype(str).str.lower().str.strip()
        obs["time_min"] = pd.to_numeric(obs["time_min"], errors="coerce")
        obs["replicate"] = pd.to_numeric(obs["replicate"], errors="coerce")
        obs = obs.dropna().drop_duplicates()
    else:
        obs = pd.DataFrame(columns=["age", "time_min", "replicate"])

    expected_key = expected[["age", "time_min", "replicate"]].copy()
    expected_key["age"] = expected_key["age"].astype(str).str.lower().str.strip()

    merged = expected_key.merge(obs, how="left", on=["age", "time_min", "replicate"], indicator=True)
    missing_expected = merged.loc[merged["_merge"] == "left_only", ["age", "time_min", "replicate"]]

    extra_observed = obs.merge(expected_key, how="left", on=["age", "time_min", "replicate"], indicator=True)
    extra_observed = extra_observed.loc[extra_observed["_merge"] == "left_only", ["age", "time_min", "replicate"]]

    missing_expected.to_csv(base / "e_mexp_2379_missing_expected_assays.csv", index=False)
    extra_observed.to_csv(base / "e_mexp_2379_extra_observed_assays.csv", index=False)

    report = {
        "expected_assays": int(len(expected)),
        "observed_sdrf_assays": int(len(obs)),
        "missing_expected_assays": int(len(missing_expected)),
        "extra_observed_assays": int(len(extra_observed)),
        "candidate_non_magetab_files_found": int(len(candidates)),
        "sdrf_truncated_flag": bool(ingest_summary.get("sdrf_truncated_flag", False)),
        "ready_for_state_table": bool(len(candidates) > 0),
        "scientific_status": (
            "metadata_valid_but_download_incomplete"
            if bool(ingest_summary.get("sdrf_truncated_flag", False))
            else "metadata_and_sample_sheet_present"
        ),
    }

    (base / "e_mexp_2379_audit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
