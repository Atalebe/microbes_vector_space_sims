from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import pandas as pd


def parse_sample_name_from_name(name: str) -> tuple[str, str, str]:
    m = re.match(r"^(GSM\d+)_([A-Za-z0-9]+)\.count\.txt\.gz$", Path(name).name)
    if not m:
        raise ValueError(f"Unrecognized count filename format: {name}")

    gsm_id = m.group(1)
    sample_name = m.group(2)
    s = sample_name.lower()

    if "unaged" in s:
        age_class = "unaged"
    elif "aged" in s:
        age_class = "aged"
    else:
        age_class = "unknown"

    return gsm_id, sample_name, age_class


def parse_replicate(sample_name: str) -> int | None:
    m = re.search(r"([A-Z])$", sample_name)
    if not m:
        return None
    return {"A": 1, "B": 2, "C": 3, "D": 4}.get(m.group(1).upper())


def read_count_text(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty count content")

    header = lines[0]
    sep = "\t" if header.count("\t") >= header.count(",") else ","

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=sep,
        engine="python",
        dtype=str,
        comment="#",
    )

    if df.shape[1] < 2:
        raise ValueError("Count table has fewer than 2 columns")

    df = df.iloc[:, :2].copy()
    df.columns = ["gene_id", "count"]
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["gene_id", "count"])
    df = df[df["gene_id"] != ""]
    return df


def read_count_file(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return read_count_text(f.read())


def extract_count_members_from_tar(tar_path: Path, extract_dir: Path) -> list[Path]:
    out = []
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if Path(m.name).name.endswith(".count.txt.gz")]
        for m in members:
            target = extract_dir / Path(m.name).name
            if not target.exists():
                with tf.extractfile(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            out.append(target)
    return sorted(out)


def build_sample_metadata(count_files: list[Path]) -> pd.DataFrame:
    rows = []
    for fp in sorted(count_files):
        gsm_id, sample_name, age_class = parse_sample_name_from_name(fp.name)
        rows.append(
            {
                "gsm_id": gsm_id,
                "sample_name": sample_name,
                "age_class": age_class,
                "replicate": parse_replicate(sample_name),
                "count_file": str(fp),
            }
        )
    return pd.DataFrame(rows)


def merge_count_files(count_files: list[Path]) -> pd.DataFrame:
    merged = None
    for fp in sorted(count_files):
        _, sample_name, _ = parse_sample_name_from_name(fp.name)
        df = read_count_file(fp).rename(columns={"count": sample_name})
        merged = df if merged is None else merged.merge(df, on="gene_id", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["gene_id"])

    for c in [c for c in merged.columns if c != "gene_id"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    return merged


def build_long_table(matrix: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    long_df = matrix.melt(id_vars=["gene_id"], var_name="sample_name", value_name="count")
    out = long_df.merge(sample_meta, on="sample_name", how="left")
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/gse188790")
    ap.add_argument("--processed-dir", default="data/processed/gse188790")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    count_files = sorted(raw_dir.glob("GSM*_*.count.txt.gz"))

    if not count_files:
        tar_path = raw_dir / "GSE188790_RAW.tar"
        if tar_path.exists():
            count_files = extract_count_members_from_tar(tar_path, raw_dir)

    if not count_files:
        raise SystemExit(f"No count files found in {raw_dir}, and no extractable GSE188790_RAW.tar found.")

    sample_meta = build_sample_metadata(count_files)
    matrix = merge_count_files(count_files)
    long_table = build_long_table(matrix, sample_meta)

    sample_meta.to_csv(processed_dir / "gse188790_sample_metadata.csv", index=False)
    matrix.to_csv(processed_dir / "gse188790_gene_count_matrix.csv", index=False)
    long_table.to_csv(processed_dir / "gse188790_gene_count_long.csv", index=False)

    summary = {
        "dataset": "GSE188790",
        "n_count_files": int(len(count_files)),
        "n_samples": int(len(sample_meta)),
        "sample_names": sample_meta["sample_name"].tolist(),
        "age_classes": sorted(sample_meta["age_class"].dropna().unique().tolist()),
        "replicates": sorted([int(x) for x in sample_meta["replicate"].dropna().unique().tolist()]),
        "n_genes": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
    }

    (processed_dir / "gse188790_ingest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
