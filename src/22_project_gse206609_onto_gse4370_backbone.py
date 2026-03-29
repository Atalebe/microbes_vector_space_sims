from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def point_to_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        t = 0.0
        proj = a.copy()
    else:
        t = np.dot(p - a, ab) / ab2
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return proj, t, dist


def main() -> None:
    outdir = "data/processed/cross_dataset_microbes"
    logdir = "results/logs/cross_dataset_microbes"
    ensure_dir(outdir)
    ensure_dir(logdir)

    # GSE4370 temporal centroids
    g4370 = pd.read_csv("data/interim/gse4370/audit_pca_scores.csv")
    b = (
        g4370.groupby(["condition", "time_min"], as_index=False)
        .agg(PC1=("PC1", "mean"), PC2=("PC2", "mean"), PC3=("PC3", "mean"))
        .sort_values("time_min")
        .reset_index(drop=True)
    )

    # GSE206609 recovery drug centroids
    g206 = pd.read_csv("data/interim/gse206609/audit_pca_scores.csv")
    d = (
        g206.loc[(g206["phase"] == "recovery") & (g206["treatment"] != "none")]
        .groupby(["treatment"], as_index=False)
        .agg(PC1=("PC1", "mean"), PC2=("PC2", "mean"), PC3=("PC3", "mean"))
        .sort_values("treatment")
        .reset_index(drop=True)
    )

    # build backbone segments and cumulative lengths
    backbone_pts = b[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    times = b["time_min"].to_numpy(dtype=float)

    seg_lengths = []
    cum_lengths = [0.0]
    for i in range(len(backbone_pts) - 1):
        seg_len = np.linalg.norm(backbone_pts[i + 1] - backbone_pts[i])
        seg_lengths.append(seg_len)
        cum_lengths.append(cum_lengths[-1] + seg_len)

    rows = []
    for _, r in d.iterrows():
        p = r[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)

        best = None
        for i in range(len(backbone_pts) - 1):
            a = backbone_pts[i]
            bb = backbone_pts[i + 1]
            proj, t, dist = point_to_segment_projection(p, a, bb)
            arc = cum_lengths[i] + t * seg_lengths[i]
            inferred_time = times[i] + t * (times[i + 1] - times[i])

            rec = {
                "treatment": r["treatment"],
                "segment_index": i,
                "from_time_min": times[i],
                "to_time_min": times[i + 1],
                "segment_fraction": t,
                "projection_distance": dist,
                "arc_length_position": arc,
                "inferred_gse4370_time_min": inferred_time,
                "proj_PC1": proj[0],
                "proj_PC2": proj[1],
                "proj_PC3": proj[2],
            }
            if best is None or rec["projection_distance"] < best["projection_distance"]:
                best = rec

        rows.append(best)

    proj_df = pd.DataFrame(rows).sort_values("inferred_gse4370_time_min").reset_index(drop=True)
    proj_df.to_csv(f"{outdir}/gse206609_projected_onto_gse4370_backbone.csv", index=False)

    # also write the temporal backbone itself
    b.to_csv(f"{outdir}/gse4370_temporal_backbone_centroids.csv", index=False)

    summary = {
        "projected_treatments": proj_df["treatment"].tolist(),
        "time_order": proj_df[["treatment", "inferred_gse4370_time_min"]].to_dict(orient="records")
    }

    with open(f"{logdir}/gse206609_on_gse4370_projection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/gse206609_projected_onto_gse4370_backbone.csv")
    print("[ok] wrote:", f"{outdir}/gse4370_temporal_backbone_centroids.csv")
    print("[ok] wrote:", f"{logdir}/gse206609_on_gse4370_projection_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
