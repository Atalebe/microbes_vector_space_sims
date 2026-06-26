#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
RAW_DIR="$ROOT/data/raw/e_mexp_2379"
mkdir -p "$RAW_DIR"

ACC="E-MEXP-2379"
API_BASE="https://www.ebi.ac.uk/biostudies/api/v1/studies/${ACC}"

echo "[info] fetching BioStudies metadata for ${ACC}"
curl -L "${API_BASE}" -o "${RAW_DIR}/${ACC}.json"
curl -L "${API_BASE}/info" -o "${RAW_DIR}/${ACC}.info.json"

echo "[info] downloading core MAGE-TAB and study files"
python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path("data/raw/e_mexp_2379")
acc = "E-MEXP-2379"

with open(root / f"{acc}.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(root / f"{acc}.info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

ftp_link = str(info.get("ftpLink", "")).rstrip("/")
if not ftp_link:
    raise SystemExit("No ftpLink found in BioStudies info JSON")

def walk(obj, out_files, out_links):
    if isinstance(obj, dict):
        if "path" in obj and isinstance(obj["path"], str):
            out_files.append(obj["path"])
        if "url" in obj and isinstance(obj["url"], str):
            out_links.append(obj["url"])
        for v in obj.values():
            walk(v, out_files, out_links)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, out_files, out_links)

file_paths = []
links = []
walk(meta, file_paths, links)

# keep only likely study files
wanted = []
for p in file_paths:
    low = p.lower()
    if any(low.endswith(suf) for suf in [
        ".idf.txt", ".sdrf.txt", ".adf.txt", ".txt", ".txt.gz",
        ".raw", ".raw.gz", ".zip", ".tar", ".tar.gz", ".cel", ".cel.gz"
    ]):
        wanted.append(p)

# also catch raw/processed/magetab-like names even if suffix is odd
for p in file_paths:
    low = p.lower()
    if any(tok in low for tok in ["raw", "processed", "mage-tab", "idf", "sdrf", "adf"]):
        wanted.append(p)

# array design links sometimes show up as accession urls
for url in links:
    u = url.strip()
    if u.startswith("A-") or u.startswith("a-"):
        fname = f"{u}.adf.txt"
        wanted.append(fname)

wanted = sorted(set(wanted))

manifest = root / "download_manifest.tsv"
with open(manifest, "w", encoding="utf-8") as out:
    out.write("file\turl\n")
    for p in wanted:
        if p.lower().endswith(".adf.txt") and not p.startswith("Files/"):
            url = f"https://www.ebi.ac.uk/biostudies/files/{p.replace('.adf.txt','')}/{p}"
            out.write(f"{p}\t{url}\n")
        else:
            clean_p = p[6:] if p.startswith("Files/") else p
            url = f"{ftp_link}/Files/{clean_p}"
            out.write(f"{clean_p}\t{url}\n")

print(f"[ok] wrote manifest: {manifest}")

with open(manifest, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        fname, url = line.rstrip("\n").split("\t", 1)
        dest = root / Path(fname).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest.name} already exists")
            continue
        print(f"[get] {dest.name}")
        subprocess.run(["wget", "-c", "-O", str(dest), url], check=False)
PY

echo "[info] listing downloaded files"
find "$RAW_DIR" -maxdepth 1 -type f | sort
