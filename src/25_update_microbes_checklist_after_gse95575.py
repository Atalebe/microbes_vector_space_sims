#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    in_path = Path("data/processed/microbial_closure/microbial_checklist_status_updated.csv")
    out_dir = Path("data/processed/microbial_closure")
    log_dir = Path("results/logs/microbial_closure")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Add or update a memory-specific upgrade line
    extra = pd.DataFrame(
        [
            {
                "stage": "Stage II/VIII",
                "item": "conditioning-memory upgrade with GSE95575",
                "status": "checked_off",
                "note": "GSE95575 added as fourth microbial branch; refined M_v2 adopted after first-pass M failed monotonicity."
            }
        ]
    )

    # Drop old duplicate if rerun
    df = df[df["item"] != "conditioning-memory upgrade with GSE95575"].copy()
    df = pd.concat([df, extra], ignore_index=True)

    # Also upgrade any broad memory wording if present
    mask = df["item"].str.contains("first-pass H/S/M definition", case=False, na=False)
    if mask.any():
        df.loc[mask, "note"] = (
            "Completed for benchmark, temporal, explicit-R, and conditioning-memory branches; "
            "GSE95575 M refined to M_v2 after monotonicity hardening."
        )

    out_path = out_dir / "microbial_checklist_status_v3.csv"
    df.to_csv(out_path, index=False)

    summary = {
        "checked_off": int((df["status"] == "checked_off").sum()),
        "partly_checked_off": int((df["status"] == "partly_checked_off").sum()),
        "pending": int((df["status"] == "pending").sum()),
        "stage_viii_memory_status": "checked_off",
        "n_rows": int(len(df)),
    }

    log_path = log_dir / "checklist_status_v3_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {log_path}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
