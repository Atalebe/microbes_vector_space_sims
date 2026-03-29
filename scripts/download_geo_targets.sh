#!/usr/bin/env bash
set -euo pipefail

download_gse () {
  local acc="$1"
  local stem="${acc:0:${#acc}-3}nnn"
  local outdir="data/raw/${acc,,}"

  mkdir -p "$outdir"

  local base="https://ftp.ncbi.nlm.nih.gov/geo/series/${stem}/${acc}"

  echo "[info] downloading ${acc} into ${outdir}"

  wget -N -P "$outdir" "${base}/matrix/${acc}_series_matrix.txt.gz" || true
  wget -N -P "$outdir" "${base}/soft/${acc}_family.soft.gz" || true
  wget -N -P "$outdir" "${base}/miniml/${acc}_family.xml.tgz" || true
  wget -N -P "$outdir" "${base}/suppl/" -r -np -nH --cut-dirs=6 -A '*.gz,*.tgz,*.tar,*.txt' || true

  echo "[ok] finished ${acc}"
  echo
}

download_gse GSE36599
download_gse GSE36601
download_gse GSE4370
download_gse GSE4371
