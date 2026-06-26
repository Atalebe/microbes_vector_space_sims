#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="results/logs/downloads"
mkdir -p "$LOG_DIR"
mkdir -p data/raw/gse58325/suppl
mkdir -p data/raw/gse95575/suppl
mkdir -p data/raw/archive_secondary

exec > >(tee -a "$LOG_DIR/conditioning_download_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "[info] starting conditioning dataset download at $(date)"

download_file() {
  local url="$1"
  local out="$2"
  echo "[info] downloading -> $out"
  wget -c --progress=bar:force:noscroll -O "$out" "$url"
  echo "[ok] finished -> $out"
  ls -lh "$out"
}

echo "[info] downloading GSE58325"
download_file \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE58nnn/GSE58325/matrix/GSE58325_series_matrix.txt.gz" \
  "data/raw/gse58325/GSE58325_series_matrix.txt.gz"

download_file \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE58nnn/GSE58325/suppl/GSE58325_RAW.tar" \
  "data/raw/gse58325/suppl/GSE58325_RAW.tar"

echo "[info] downloading GSE95575"
download_file \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE95nnn/GSE95575/matrix/GSE95575_series_matrix.txt.gz" \
  "data/raw/gse95575/GSE95575_series_matrix.txt.gz"

download_file \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE95nnn/GSE95575/suppl/GSE95575_RAW.tar" \
  "data/raw/gse95575/suppl/GSE95575_RAW.tar"

echo "[info] extracting supplementary archives"
mkdir -p data/raw/gse58325/raw
mkdir -p data/raw/gse95575/raw

tar -xf data/raw/gse58325/suppl/GSE58325_RAW.tar -C data/raw/gse58325/raw || true
tar -xf data/raw/gse95575/suppl/GSE95575_RAW.tar -C data/raw/gse95575/raw || true

echo "[ok] done at $(date)"
find data/raw/gse58325 -maxdepth 2 -type f | sort | head -n 20
find data/raw/gse95575 -maxdepth 2 -type f | sort | head -n 20
