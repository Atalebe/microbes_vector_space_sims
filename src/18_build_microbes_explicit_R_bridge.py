from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def minmax01(x: pd.Series) -> pd.Series:
    xmin = x.min()
    xmax = x.max()
    if xmax == xmin:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


def main() -> None:
    outdir = "data/processed/microbial_closure"
    logdir = "results/logs/microbial_closure"
    ensure_dir(outdir)
    ensure_dir(logdir)

    # -------------------------
    # GSE36599
    # -------------------------
    g1 = pd.read_csv("data/processed/gse36599/gse36599_refined_stability_samples.csv")
    # observed-ish recovery score from stage order and closeness to recovered state
    g1["observed_recovery_progress_raw"] = (
        0.6 * minmax01(g1["condition_order"]) +
        0.4 * (1.0 - minmax01(g1["dist_to_after_recovery"]))
    )
    g1["observed_recovery_progress"] = minmax01(g1["observed_recovery_progress_raw"])
    g1["latent_R"] = g1["R"]

    g1_compare = g1[[
        "sample_accession", "condition", "condition_order",
        "latent_R", "observed_recovery_progress", "stress_burden", "H"
    ]].copy()
    g1_corr = g1_compare[["latent_R", "observed_recovery_progress"]].corr(method="spearman").iloc[0, 1]

    # -------------------------
    # GSE4370
    # -------------------------
    g2 = pd.read_csv("data/processed/gse4370/gse4370_residual_field_samples.csv")
    g2["observed_recovery_progress_raw"] = (
        0.5 * minmax01(g2["time_min"]) +
        0.5 * (1.0 - minmax01(g2["dist_to_end"]))
    )
    g2["observed_recovery_progress"] = minmax01(g2["observed_recovery_progress_raw"])
    g2["latent_R"] = g2["R"]

    g2_compare = g2[[
        "sample_accession", "condition", "time_min",
        "latent_R", "observed_recovery_progress", "stress_burden", "H"
    ]].copy()
    g2_corr = g2_compare[["latent_R", "observed_recovery_progress"]].corr(method="spearman").iloc[0, 1]

    g1_compare.to_csv(f"{outdir}/gse36599_latent_vs_observed_recovery_bridge.csv", index=False)
    g2_compare.to_csv(f"{outdir}/gse4370_latent_vs_observed_recovery_bridge.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "GSE36599",
                "comparison_type": "latent_R_vs_observed_recovery_progress_bridge",
                "spearman_correlation": g1_corr,
                "note": "Bridge metric from stage order and approach to recovered state, not direct survival/regrowth biology.",
            },
            {
                "dataset": "GSE4370",
                "comparison_type": "latent_R_vs_observed_recovery_progress_bridge",
                "spearman_correlation": g2_corr,
                "note": "Bridge metric from recovery time and approach to late endpoint, not direct survival/regrowth biology.",
            },
        ]
    )
    summary.to_csv(f"{outdir}/microbial_latent_R_bridge_summary.csv", index=False)

    log = {
        "datasets": ["GSE36599", "GSE4370"],
        "note": "These are observed-recovery bridge comparisons, not fully explicit biological recoverability measurements.",
    }

    with open(f"{logdir}/latent_R_bridge_summary.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("[ok] wrote:", f"{outdir}/gse36599_latent_vs_observed_recovery_bridge.csv")
    print("[ok] wrote:", f"{outdir}/gse4370_latent_vs_observed_recovery_bridge.csv")
    print("[ok] wrote:", f"{outdir}/microbial_latent_R_bridge_summary.csv")
    print("[ok] wrote:", f"{logdir}/latent_R_bridge_summary.json")
    print("[summary]", summary.to_dict(orient="records"))


if __name__ == "__main__":
    main()
