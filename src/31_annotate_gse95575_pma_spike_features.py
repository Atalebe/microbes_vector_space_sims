#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def normalize_feature_meta(feature_meta: pd.DataFrame) -> pd.DataFrame:
    fm = feature_meta.copy()

    rename_map = {}
    for col in fm.columns:
        low = str(col).strip().lower()
        if low in {"feature_id", "gene", "locus_tag", "id"}:
            rename_map[col] = "feature_id"
        elif low in {"gene_symbol", "symbol", "gene_name", "name"}:
            rename_map[col] = "gene_symbol"
        elif low in {"gene_desc", "description", "product", "annotation", "gene_description"}:
            rename_map[col] = "gene_description"
        elif low in {"feature", "feature_type"}:
            rename_map[col] = "feature"

    fm = fm.rename(columns=rename_map)

    if "feature_id" not in fm.columns:
        fm = fm.rename(columns={fm.columns[0]: "feature_id"})

    for col in ["feature_id", "gene_symbol", "gene_description", "feature"]:
        if col not in fm.columns:
            fm[col] = pd.NA

    fm["feature_id"] = fm["feature_id"].astype(str).str.strip()
    fm["gene_symbol"] = fm["gene_symbol"].fillna("").astype(str).str.strip()
    fm["gene_description"] = fm["gene_description"].fillna("").astype(str).str.strip()
    fm["feature"] = fm["feature"].fillna("").astype(str).str.strip()

    return fm[["feature_id", "gene_symbol", "gene_description", "feature"]].drop_duplicates()


def classify_feature(row: pd.Series) -> str:
    fid = str(row.get("feature_id", "")).strip().lower()
    sym = str(row.get("gene_symbol", "")).strip().lower()
    desc = str(row.get("gene_description", "")).strip().lower()
    feature = str(row.get("feature", "")).strip().lower()

    text = " ".join(x for x in [fid, sym, desc, feature] if x)

    ribo_rna_symbols = {
        "rrsa", "rrsb", "rrsc", "rrsd", "rrse", "rrsg", "rrsh",
        "rrla", "rrlb", "rrlc", "rrld", "rrle", "rrlg", "rrlh",
        "rrfa", "rrfb", "rrfc", "rrfd", "rrfe", "rrfg", "rrfh",
    }
    regulatory_rna_symbols = {"ssra", "ssrs", "rnpb"}
    stress_reg_symbols = {"rpos", "ahpc", "hns", "cspe"}
    membrane_symbols = {"ompa", "ompf", "lpxc", "lpp", "psts", "nlpd"}
    flagellar_symbols = {"flic", "flgk", "flgl"}
    metal_detox_symbols = {"copa", "cueo"}
    translation_symbols = {"fusa", "rpsb"}
    central_symbols = {"gapa", "leua", "leuc", "cobs", "cobt"}

    # First, explicit symbol rules
    if sym in ribo_rna_symbols:
        return "ribosomal_rna"
    if sym in regulatory_rna_symbols:
        return "regulatory_rescue_rna"
    if sym in stress_reg_symbols:
        return "stress_regulation"
    if sym in membrane_symbols:
        return "membrane_envelope"
    if sym in flagellar_symbols:
        return "motility_flagellar"
    if sym in metal_detox_symbols:
        return "metal_detox"
    if sym in translation_symbols:
        return "translation_ribosome_associated"
    if sym in central_symbols:
        return "core_metabolism_or_biosynthesis"

    # Then feature-type rules
    if feature == "rrna" or "ribosomal rna" in desc:
        return "ribosomal_rna"
    if feature == "ncrna" or "small rna" in desc or "ncrna" in desc:
        return "regulatory_rescue_rna"
    if "tmrna" in desc or "transfer-messenger rna" in desc or sym == "ssra":
        return "regulatory_rescue_rna"

    # Then keyword rules on text
    if "rrna" in text:
        return "ribosomal_rna"
    if "trna" in text:
        return "translation_rna"
    if "ncRNA".lower() in text or "srna" in text or "antisense" in text:
        return "regulatory_rescue_rna"

    if any(k in text for k in [
        "stress", "oxidative", "peroxide", "chaperone", "repair", "dna damage",
        "stationary phase", "sigma s"
    ]):
        return "stress_regulation"

    if any(k in text for k in [
        "outer membrane", "membrane", "envelope", "lipoprotein", "porin", "lipid a"
    ]):
        return "membrane_envelope"

    if any(k in text for k in [
        "flagellar", "flagell", "motility"
    ]):
        return "motility_flagellar"

    if any(k in text for k in [
        "copper", "metal efflux", "detox", "efflux", "export"
    ]):
        return "metal_detox"

    if any(k in text for k in [
        "ribosomal protein", "elongation factor", "translation"
    ]):
        return "translation_ribosome_associated"

    if any(k in text for k in [
        "metabolism", "biosynthesis", "synthase", "dehydrogenase", "phosphate", "leucine"
    ]):
        return "core_metabolism_or_biosynthesis"

    if any(k in text for k in ["hypothetical", "uncharacterized", "unknown"]):
        return "unknown_or_uncharacterized"

    if sym or desc or feature:
        return "other_coding"

    return "unknown_or_uncharacterized"


def annotate_table(df: pd.DataFrame, feature_meta: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature_id"] = out["feature_id"].astype(str).str.strip()

    drop_cols = [c for c in ["gene_symbol", "gene_description", "feature", "functional_class"] if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")

    out = out.merge(feature_meta, on="feature_id", how="left")

    for col in ["gene_symbol", "gene_description", "feature"]:
        out[col] = out[col].fillna("").astype(str)

    out["functional_class"] = out.apply(classify_feature, axis=1)
    return out


def main() -> None:
    processed_dir = Path("data/processed/gse95575")
    interim_dir = Path("data/interim/gse95575")
    log_dir = Path("results/logs/gse95575")

    processed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    feature_meta = pd.read_csv(interim_dir / "feature_metadata.csv")
    feature_meta = normalize_feature_meta(feature_meta)

    spike = pd.read_csv(processed_dir / "gse95575_9C_PMA_t30_top_spike_features.csv")
    up = pd.read_csv(processed_dir / "gse95575_9C_PMA_t30_top_up_vs_peers.csv")
    down = pd.read_csv(processed_dir / "gse95575_9C_PMA_t30_top_down_vs_peers.csv")

    spike_annot = annotate_table(spike, feature_meta)
    up_annot = annotate_table(up, feature_meta)
    down_annot = annotate_table(down, feature_meta)

    spike_out = processed_dir / "gse95575_9C_PMA_t30_top_spike_features_annotated.csv"
    up_out = processed_dir / "gse95575_9C_PMA_t30_top_up_vs_peers_annotated.csv"
    down_out = processed_dir / "gse95575_9C_PMA_t30_top_down_vs_peers_annotated.csv"

    spike_annot.to_csv(spike_out, index=False)
    up_annot.to_csv(up_out, index=False)
    down_annot.to_csv(down_out, index=False)

    summary = {
        "dataset_name": "gse95575",
        "target_sample": "9C_PMA_t30",
        "top_spike_functional_classes": spike_annot.head(30)["functional_class"].value_counts().to_dict(),
        "top_up_functional_classes": up_annot.head(30)["functional_class"].value_counts().to_dict(),
        "top_down_functional_classes": down_annot.head(30)["functional_class"].value_counts().to_dict(),
    }

    log_out = log_dir / "gse95575_9C_PMA_t30_annotation_summary.json"
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {spike_out}")
    print(f"[ok] wrote: {up_out}")
    print(f"[ok] wrote: {down_out}")
    print(f"[ok] wrote: {log_out}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
