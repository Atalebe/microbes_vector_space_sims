#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from utils import load_config, make_state_table, read_table, write_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(cfg["io"]["raw_input_path"])
    state = make_state_table(df, cfg)

    keep_cols = []
    for key in ["sample_id", "strain", "condition", "time"]:
        col = cfg.get("columns", {}).get(key)
        if col and col in state.columns:
            keep_cols.append(col)
    keep_cols += [c for c in ["H", "S", "M", "R_explicit"] if c in state.columns]
    keep_cols += [c for c in cfg.get("preserve_columns", []) if c in state.columns and c not in keep_cols]

    out = state[keep_cols].copy()
    out_path = Path(cfg["io"]["processed_dir"]) / "state_table.parquet"
    write_table(out, out_path)
    write_table(out, out_path.with_suffix(".csv"))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
