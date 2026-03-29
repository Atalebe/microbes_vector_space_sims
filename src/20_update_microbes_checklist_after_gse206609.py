from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    outdir = "data/processed/microbial_closure"
    logdir = "results/logs/microbial_closure"
    ensure_dir(outdir)
    ensure_dir(logdir)

    rows = [
        {
            "stage": "Stage I",
            "item": "dataset assembly and audit",
            "status": "checked_off",
            "note": "Three microbial branches ingested and audited, GSE36599, GSE4370, and GSE206609."
        },
        {
            "stage": "Stage I",
            "item": "quality flag scheme",
            "status": "checked_off",
            "note": "Formal quality flags remain in place."
        },
        {
            "stage": "Stage I",
            "item": "dataset suitability ranking note",
            "status": "checked_off",
            "note": "Ranking note exists and now includes explicit-R upgrade context."
        },
        {
            "stage": "Stage II",
            "item": "first-pass H/S/M definition",
            "status": "checked_off",
            "note": "Completed for all branches."
        },
        {
            "stage": "Stage II",
            "item": "unified adopted proxy table",
            "status": "checked_off",
            "note": "Combined proxy table exists."
        },
        {
            "stage": "Stage III",
            "item": "bounded microbial geometry",
            "status": "checked_off",
            "note": "Structured geometry established across all branches."
        },
        {
            "stage": "Stage IV",
            "item": "residual field and latent R",
            "status": "checked_off",
            "note": "Completed for benchmark and temporal branches, and explicit-R branch."
        },
        {
            "stage": "Stage V",
            "item": "explicit recoverability R",
            "status": "checked_off",
            "note": "Upgraded by GSE206609 to a treatment-recovery and DE-burden aware explicit recoverability layer."
        },
        {
            "stage": "Stage VI",
            "item": "trajectory and vector velocity",
            "status": "checked_off",
            "note": "Completed for GSE36599, GSE4370, and GSE206609."
        },
        {
            "stage": "Stage VII",
            "item": "rare cases and frontier logic",
            "status": "checked_off",
            "note": "Frontier catalog exists and now includes MMC as an explicit high-burden frontier."
        },
        {
            "stage": "Stage VIII",
            "item": "validation",
            "status": "checked_off",
            "note": "Backbone sensitivity, leave-one-out, anchor sensitivity, and refined-state checks completed."
        },
        {
            "stage": "Stage IX",
            "item": "variance scaling closure",
            "status": "checked_off",
            "note": "Completed including with-and-without frontier tests, and with-and-without MMC."
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/microbial_checklist_status_updated.csv", index=False)

    summary = {
        "checked_off": int((df["status"] == "checked_off").sum()),
        "partly_checked_off": int((df["status"] == "partly_checked_off").sum()),
        "pending": int((df["status"] == "pending").sum()),
        "stage_v_status": df.loc[df["item"] == "explicit recoverability R", "status"].iloc[0],
    }

    with open(f"{logdir}/checklist_status_updated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_checklist_status_updated.csv")
    print("[ok] wrote:", f"{logdir}/checklist_status_updated_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
