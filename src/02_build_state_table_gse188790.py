from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return (x - np.nanmean(x)) / sd
    return 0.67448975 * (x - med) / mad


def cpm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0)
    return counts.divide(lib, axis=1) * 1_000_000.0


def centered_log_cpm(counts: pd.DataFrame, prior_count: float = 1.0) -> pd.DataFrame:
    cpm = cpm_normalize(counts)
    log_cpm = np.log2(cpm + prior_count)
    gene_center = log_cpm.mean(axis=1)
    return log_cpm.subtract(gene_center, axis=0)


def pca_scores(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    var = (S ** 2) / max(1, (X.shape[0] - 1))
    frac = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    return scores, frac[:n_components]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/gse188790")
    args = ap.parse_args()

    base = Path(args.processed_dir)
    sample_meta_path = base / "gse188790_sample_metadata.csv"
    matrix_path = base / "gse188790_gene_count_matrix.csv"

    if not sample_meta_path.exists() or not matrix_path.exists():
        raise SystemExit("Missing ingest outputs. Run ingest and audit first.")

    sample_meta = pd.read_csv(sample_meta_path)
    matrix = pd.read_csv(matrix_path)

    sample_cols = [c for c in matrix.columns if c != "gene_id"]
    counts = matrix.set_index("gene_id")[sample_cols].copy()

    lib_size = counts.sum(axis=0)
    detected_genes = (counts > 0).sum(axis=0)
    log_cpm_centered = centered_log_cpm(counts, prior_count=1.0)

    X = log_cpm_centered.T.values
    scores, frac = pca_scores(X, n_components=2)

    sample_order = sample_cols
    pca_df = pd.DataFrame(
        {
            "sample_name": sample_order,
            "pc1": scores[:, 0] if scores.shape[1] >= 1 else 0.0,
            "pc2": scores[:, 1] if scores.shape[1] >= 2 else 0.0,
        }
    )

    state = sample_meta.merge(pca_df, on="sample_name", how="left")
    state["library_size"] = state["sample_name"].map(lib_size.to_dict()).astype(float)
    state["detected_genes"] = state["sample_name"].map(detected_genes.to_dict()).astype(float)

    age_map = {"unaged": 0.0, "aged": 1.0}
    state["age_order"] = state["age_class"].map(age_map).astype(float)

    age_centroids = (
        state.groupby("age_class")[["pc1", "pc2"]]
        .mean()
        .rename(columns={"pc1": "pc1_centroid", "pc2": "pc2_centroid"})
        .reset_index()
    )
    state = state.merge(age_centroids, on="age_class", how="left")

    state["dist_to_age_centroid"] = np.sqrt(
        (state["pc1"] - state["pc1_centroid"]) ** 2 +
        (state["pc2"] - state["pc2_centroid"]) ** 2
    )

    # First-pass proxy definitions
    # H: transcriptomic depth / expression burden
    state["H_raw"] = np.log10(state["library_size"].clip(lower=1.0))

    # S: structural compactness/coherence, smaller distance to age centroid = more stable/coherent
    state["S_raw"] = -state["dist_to_age_centroid"]

    # M: explicit age ordering axis
    state["M_raw"] = state["age_order"]

    # R: conservative internal regulation proxy, detected gene richness relative to depth
    # This is not recoverability and must be labeled clearly as provisional.
    state["R_raw"] = robust_z(np.log10(state["detected_genes"].clip(lower=1.0))) - robust_z(np.log10(state["library_size"].clip(lower=1.0)))

    state["H"] = robust_z(state["H_raw"])
    state["S"] = robust_z(state["S_raw"])
    state["M"] = robust_z(state["M_raw"])
    state["R"] = robust_z(state["R_raw"])

    outcols = [
        "gsm_id",
        "sample_name",
        "age_class",
        "replicate",
        "library_size",
        "detected_genes",
        "pc1",
        "pc2",
        "dist_to_age_centroid",
        "H_raw",
        "S_raw",
        "M_raw",
        "R_raw",
        "H",
        "S",
        "M",
        "R",
    ]
    state = state[outcols].sort_values(["age_class", "replicate", "sample_name"]).reset_index(drop=True)

    state.to_csv(base / "gse188790_state_table.csv", index=False)

    proxy_table = pd.DataFrame(
        [
            {
                "coordinate": "H",
                "proxy_name": "log10_library_size",
                "definition": "log10 total mapped count burden per sample",
                "status": "adopted_first_pass",
            },
            {
                "coordinate": "S",
                "proxy_name": "negative_distance_to_age_centroid",
                "definition": "minus PCA distance to age-class centroid, higher means more coherent/compact within age class",
                "status": "adopted_first_pass",
            },
            {
                "coordinate": "M",
                "proxy_name": "age_order_binary",
                "definition": "binary age ordering, unaged=0 aged=1",
                "status": "adopted_first_pass",
            },
            {
                "coordinate": "R",
                "proxy_name": "detected_gene_richness_minus_depth_burden",
                "definition": "provisional internal regulation richness proxy, not direct recoverability",
                "status": "provisional_not_explicit_recoverability",
            },
        ]
    )
    proxy_table.to_csv(base / "gse188790_proxy_table.csv", index=False)

    summary = {
        "dataset": "GSE188790",
        "n_samples": int(len(state)),
        "n_genes": int(len(counts)),
        "pc_variance_fraction": {
            "pc1": float(frac[0]) if len(frac) > 0 else 0.0,
            "pc2": float(frac[1]) if len(frac) > 1 else 0.0,
        },
        "age_class_counts": state["age_class"].value_counts().sort_index().to_dict(),
        "library_size_by_sample": {k: int(v) for k, v in lib_size.to_dict().items()},
        "detected_genes_by_sample": {k: int(v) for k, v in detected_genes.to_dict().items()},
        "notes": [
            "GSE188790 is treated as an explicit microbial age branch.",
            "M is the main strengthened coordinate in this dataset.",
            "R is provisional and not interpreted as direct recoverability."
        ],
    }
    (base / "gse188790_state_table_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
