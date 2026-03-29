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
        {"stage": "Stage I", "item": "dataset assembly and audit", "status": "checked_off", "note": "Two datasets ingested and audited."},
        {"stage": "Stage I", "item": "quality flag scheme", "status": "checked_off", "note": "Formal quality flags added in closure pack."},
        {"stage": "Stage I", "item": "dataset suitability ranking note", "status": "checked_off", "note": "Formal ranking table added."},
        {"stage": "Stage II", "item": "first-pass H/S/M definition", "status": "checked_off", "note": "Completed for both datasets."},
        {"stage": "Stage II", "item": "unified adopted proxy table", "status": "checked_off", "note": "Formal combined table added."},
        {"stage": "Stage III", "item": "bounded microbial geometry", "status": "checked_off", "note": "Strong structured geometry in both datasets."},
        {"stage": "Stage IV", "item": "residual field and latent R", "status": "checked_off", "note": "Completed for both datasets."},
        {"stage": "Stage V", "item": "explicit recoverability R", "status": "partly_checked_off", "note": "Bridge metrics added, but no direct regrowth/survival metric yet."},
        {"stage": "Stage VI", "item": "trajectory and vector velocity", "status": "checked_off", "note": "Completed for both datasets."},
        {"stage": "Stage VII", "item": "rare cases and frontier logic", "status": "checked_off", "note": "Formal frontier catalog added."},
        {"stage": "Stage VIII", "item": "validation", "status": "checked_off", "note": "Backbone sensitivity, leave-one-out, refinement, and cross-dataset comparison completed."},
        {"stage": "Stage IX", "item": "variance scaling closure", "status": "checked_off", "note": "Completed, including 45–75 min late-window test for companion branch."},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/microbial_checklist_status.csv", index=False)

    summary = {
        "checked_off": int((df["status"] == "checked_off").sum()),
        "partly_checked_off": int((df["status"] == "partly_checked_off").sum()),
        "pending": int((df["status"] == "pending").sum()),
    }

    with open(f"{logdir}/checklist_status_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{outdir}/microbial_checklist_status.csv")
    print("[ok] wrote:", f"{logdir}/checklist_status_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
