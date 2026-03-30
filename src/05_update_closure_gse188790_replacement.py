from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_FILES = [
    "data/processed/microbial_closure/microbial_checklist_status_v3.csv",
    "data/processed/microbial_closure/microbial_adopted_proxy_table_v2.csv",
    "data/processed/microbial_closure/microbial_dataset_suitability_ranking_v2.csv",
    "data/processed/cross_dataset_microbes/microbes_dataset_comparison_v3.csv",
    "data/processed/cross_dataset_microbes/microbes_frontier_variance_comparison_v3.csv",
]


OLD_ACCESSION = "E-MEXP-2379"
NEW_ACCESSION = "GSE188790"


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def ensure_col(df: pd.DataFrame, col: str, default="") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def choose_key_col(df: pd.DataFrame) -> str:
    key = find_col(df, ["dataset", "accession", "dataset_id", "study", "study_id"])
    if key is None:
        key = "dataset"
        df[key] = ""
    return key


def upsert_row(df: pd.DataFrame, key_col: str, key_value: str, payload: dict) -> tuple[pd.DataFrame, str]:
    mask = df[key_col].astype(str).str.strip().eq(key_value)
    if mask.any():
        idx = df.index[mask][0]
        for k, v in payload.items():
            df = ensure_col(df, k, "")
            df.at[idx, k] = v
        action = "updated"
    else:
        for k in payload:
            df = ensure_col(df, k, "")
        new_row = {c: "" for c in df.columns}
        for k, v in payload.items():
            new_row[k] = v
        new_row[key_col] = key_value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "inserted"
    return df, action


def maybe_retire_old_row(df: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, str]:
    mask = df[key_col].astype(str).str.strip().eq(OLD_ACCESSION)
    if not mask.any():
        return df, "not_found"

    idx = df.index[mask][0]

    updates = {
        "status": "validated_metadata_only_archive_incomplete",
        "operational_status": "non_operational",
        "role": "intended microbial age branch, not ingestible",
        "notes": (
            "Metadata scientifically valid but public archive incomplete. "
            "Only IDF and truncated SDRF exposed, no usable expression matrix or raw intensity files."
        ),
        "replacement_dataset": NEW_ACCESSION,
        "included_in_masterpaper": "no",
    }

    for k, v in updates.items():
        df = ensure_col(df, k, "")
        df.at[idx, k] = v

    return df, "updated"


def payload_for_new_dataset(file_path: str) -> dict:
    path_l = file_path.lower()

    base_payload = {
        "dataset": NEW_ACCESSION,
        "title": "Transcriptome of Aged and Unaged E. coli populations",
        "organism": "Escherichia coli",
        "platform": "RNA-seq",
        "role": "operational microbial age branch",
        "status": "operational",
        "operational_status": "ready",
        "age_axis": "explicit aged versus unaged",
        "samples": 4,
        "replicates": 2,
        "genes": 4450,
        "primary_strength": "age ordering",
        "m_strengthened": "yes",
        "r_explicit": "no",
        "conditioning_history_branch": "no",
        "replaces_dataset": OLD_ACCESSION,
        "included_in_masterpaper": "yes",
        "notes": (
            "Accessible GEO RNA-seq replacement for archive-incomplete E-MEXP-2379. "
            "Two unaged and two aged samples with downloadable count files. "
            "Primary scientific role is explicit microbial age ordering."
        ),
    }

    if "adopted_proxy_table" in path_l:
        return {
            "dataset": NEW_ACCESSION,
            "coordinate": "M",
            "proxy_name": "binary_age_status",
            "proxy_definition": "unaged=0, aged=1",
            "proxy_status": "adopted_first_pass",
            "dataset_role": "operational microbial age branch",
            "notes": "Main strengthened coordinate in GSE188790.",
        }

    if "frontier_variance_comparison" in path_l:
        return {
            "dataset": NEW_ACCESSION,
            "branch_role": "age branch",
            "frontier_type": "aged versus unaged transcriptome contrast",
            "variance_status": "first_pass_available",
            "top_signal": "flagellar and chemotaxis enrichment among aged-up genes",
            "notes": "Residual field and first figure set available.",
        }

    if "dataset_suitability_ranking" in path_l:
        payload = dict(base_payload)
        payload.update(
            {
                "suitability_rank": "",
                "suitability_status": "adopted",
                "accessibility": "high",
                "archive_integrity": "complete",
                "replacement_reason": "E-MEXP-2379 archive incomplete",
            }
        )
        return payload

    if "dataset_comparison" in path_l:
        payload = dict(base_payload)
        payload.update(
            {
                "branch_role": "age",
                "recoverability_role": "no",
                "conditioning_memory_role": "no",
                "time_backbone_role": "no",
                "benchmark_role": "no",
            }
        )
        return payload

    return base_payload


def sort_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["dataset", "accession", "dataset_id", "study"]:
        if c in df.columns:
            return df.sort_values(c).reset_index(drop=True)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=TARGET_FILES)
    ap.add_argument("--log-dir", default="results/logs/gse188790")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    changes = []

    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            changes.append(
                {
                    "file": fp,
                    "status": "missing_skipped",
                }
            )
            continue

        df = pd.read_csv(path)
        key_col = choose_key_col(df)

        df, old_action = maybe_retire_old_row(df, key_col)

        payload = payload_for_new_dataset(fp)
        df, new_action = upsert_row(df, key_col, NEW_ACCESSION, payload)

        df = sort_if_possible(df)
        df.to_csv(path, index=False)

        changes.append(
            {
                "file": fp,
                "key_col": key_col,
                "old_dataset_action": old_action,
                "new_dataset_action": new_action,
                "rows_after": int(len(df)),
            }
        )

    summary = {
        "old_dataset": OLD_ACCESSION,
        "new_dataset": NEW_ACCESSION,
        "files_processed": changes,
        "scientific_update": (
            "E-MEXP-2379 retained as scientifically valid but archive-incomplete. "
            "GSE188790 promoted as the operational microbial age branch."
        ),
    }

    out_json = log_dir / "gse188790_closure_update_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
