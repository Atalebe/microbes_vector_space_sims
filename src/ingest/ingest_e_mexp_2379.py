from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


def read_text_any_newline(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def parse_table_loose(path: Path) -> pd.DataFrame:
    text = read_text_any_newline(path).strip()
    if not text:
        return pd.DataFrame()

    first_line = text.split("\n", 1)[0]
    delim = "\t" if first_line.count("\t") >= first_line.count(",") else ","
    try:
        return pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype=str)
    except Exception:
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        if not rows:
            return pd.DataFrame()
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        return pd.DataFrame(rows[1:], columns=rows[0])


def load_local_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def flatten_files(obj: Any, hits: list[dict[str, str]], parent_type: str = "") -> None:
    if isinstance(obj, dict):
        current_type = str(obj.get("type", parent_type))
        if "path" in obj:
            hits.append(
                {
                    "section_type": current_type,
                    "path": str(obj.get("path", "")),
                    "name": str(obj.get("name", "")),
                    "size": str(obj.get("size", "")),
                }
            )
        for v in obj.values():
            flatten_files(v, hits, current_type)
    elif isinstance(obj, list):
        for item in obj:
            flatten_files(item, hits, parent_type)


def collect_remote_files_from_local_json(study_json: dict[str, Any], info_json: dict[str, Any]) -> pd.DataFrame:
    hits: list[dict[str, str]] = []
    flatten_files(study_json, hits, "")
    df = pd.DataFrame(hits).drop_duplicates()

    ftp_link = str(info_json.get("ftpLink", "")).rstrip("/")
    if not df.empty:
        df["url"] = df["path"].map(lambda p: f"{ftp_link}/Files/{p}" if ftp_link and p else "")
    else:
        df = pd.DataFrame(columns=["section_type", "path", "name", "size", "url"])

    return df


def expected_design_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rep in [1, 2]:
        for t in [0, 15, 30, 45, 60]:
            rows.append(
                {
                    "sample_id": f"young_rep{rep}_{t:03d}min",
                    "age": "young",
                    "time_min": t,
                    "replicate": rep,
                    "factor_growth_condition": "YoungCells",
                }
            )
        for t in [0, 75, 150, 225, 300]:
            rows.append(
                {
                    "sample_id": f"old_rep{rep}_{t:03d}min",
                    "age": "old",
                    "time_min": t,
                    "replicate": rep,
                    "factor_growth_condition": "OldCells",
                }
            )
    return pd.DataFrame(rows)


def infer_sample_table_from_sdrf(sdrf: pd.DataFrame) -> pd.DataFrame:
    if sdrf.empty:
        return pd.DataFrame()

    sdrf = sdrf.copy()
    sdrf.columns = [str(c).strip() for c in sdrf.columns]

    out = pd.DataFrame(index=sdrf.index)
    for col in ["Source Name", "Extract Name", "Labeled Extract Name", "Hybridization Name"]:
        if col in sdrf.columns:
            out[col] = sdrf[col]

    if "Factor Value [GROWTH_CONDITION]" in sdrf.columns:
        out["growth_condition_raw"] = sdrf["Factor Value [GROWTH_CONDITION]"]
    else:
        out["growth_condition_raw"] = ""

    if "Factor Value [TIME IN LAG PHASE]" in sdrf.columns:
        out["time_raw"] = sdrf["Factor Value [TIME IN LAG PHASE]"]
    else:
        out["time_raw"] = ""

    joined = sdrf.astype(str).agg(" | ".join, axis=1).str.lower()

    def infer_age(text: str) -> str:
        if "young" in text:
            return "young"
        if "old" in text:
            return "old"
        return ""

    def infer_rep(text: str) -> str:
        m = re.search(r"(youngcells|oldcells)([12])", text, flags=re.I)
        if m:
            return m.group(2)
        m = re.search(r"(rep|replicate)[ _-]?([12])", text, flags=re.I)
        if m:
            return m.group(2)
        return ""

    out["age"] = joined.map(infer_age)
    out["time_min"] = pd.to_numeric(pd.Series(out["time_raw"]).astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    out["replicate"] = pd.to_numeric(joined.map(infer_rep), errors="coerce")
    return out


def choose_candidates(remote_files: pd.DataFrame) -> pd.DataFrame:
    if remote_files.empty:
        return pd.DataFrame(columns=remote_files.columns)

    x = remote_files.copy()
    x["path_l"] = x["path"].astype(str).str.lower()
    keep = (
        x["path_l"].str.endswith((".txt", ".txt.gz", ".tsv", ".csv", ".zip", ".cel", ".gpr", ".soft"))
        & ~x["path_l"].str.endswith((".idf.txt", ".sdrf.txt", ".adf.txt"))
    )
    return x.loc[keep].drop(columns=["path_l"]).drop_duplicates()


def download_url(url: str, dest: Path) -> tuple[bool, str]:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        dest.write_bytes(data)
        return True, ""
    except Exception as e:
        return False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accession", default="E-MEXP-2379")
    ap.add_argument("--raw-dir", default="data/raw/e_mexp_2379")
    ap.add_argument("--processed-dir", default="data/processed/e_mexp_2379")
    ap.add_argument("--download-remote", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    idf_path = raw_dir / f"{args.accession}.idf.txt"
    sdrf_path = raw_dir / f"{args.accession}.sdrf.txt"
    study_json_path = raw_dir / f"{args.accession}.json"
    info_json_path = raw_dir / f"{args.accession}.info.json"

    idf_text = read_text_any_newline(idf_path) if idf_path.exists() else ""
    sdrf = parse_table_loose(sdrf_path) if sdrf_path.exists() else pd.DataFrame()
    study_json = load_local_json(study_json_path)
    info_json = load_local_json(info_json_path)

    remote_files = collect_remote_files_from_local_json(study_json, info_json)
    candidates = choose_candidates(remote_files)
    sample_from_sdrf = infer_sample_table_from_sdrf(sdrf)
    expected = expected_design_frame()

    remote_files.to_csv(proc_dir / "e_mexp_2379_remote_file_inventory.csv", index=False)
    candidates.to_csv(proc_dir / "e_mexp_2379_candidate_non_magetab_files.csv", index=False)
    expected.to_csv(proc_dir / "e_mexp_2379_expected_design.csv", index=False)

    if not sample_from_sdrf.empty:
        sample_from_sdrf.to_csv(proc_dir / "e_mexp_2379_sample_table_from_sdrf.csv", index=False)

    download_log: list[dict[str, str]] = []
    if args.download_remote and not candidates.empty:
        for _, row in candidates.iterrows():
            url = str(row.get("url", ""))
            path = str(row.get("path", "")).strip()
            if not url or not path:
                continue
            dest = raw_dir / Path(path).name
            ok, err = download_url(url, dest)
            download_log.append(
                {
                    "path": path,
                    "url": url,
                    "dest": str(dest),
                    "downloaded": str(ok),
                    "error": err,
                }
            )

    pd.DataFrame(download_log).to_csv(proc_dir / "e_mexp_2379_download_log.csv", index=False)

    summary = {
        "accession": args.accession,
        "idf_exists": idf_path.exists(),
        "sdrf_exists": sdrf_path.exists(),
        "study_json_exists": study_json_path.exists(),
        "info_json_exists": info_json_path.exists(),
        "sdrf_rows_observed": int(len(sdrf)),
        "sdrf_cols_observed": int(len(sdrf.columns)) if not sdrf.empty else 0,
        "sdrf_truncated_flag": bool(len(sdrf) < 20),
        "expected_assays": 20,
        "expected_age_levels": ["young", "old"],
        "expected_young_times": [0, 15, 30, 45, 60],
        "expected_old_times": [0, 75, 150, 225, 300],
        "expected_replicates": [1, 2],
        "remote_file_count_from_json": int(len(remote_files)),
        "candidate_non_magetab_file_count": int(len(candidates)),
        "idf_mentions_young_old": ("young cells" in idf_text.lower() and "old cells" in idf_text.lower()),
    }
    (proc_dir / "e_mexp_2379_ingest_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
