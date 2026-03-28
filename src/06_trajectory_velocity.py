#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_col, load_config, read_table


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(Path(cfg["io"]["results_dir"]) / "tables" / "residual_field.csv")
    time_col = get_col(cfg, "time")
    group_col = get_col(cfg, "trajectory_group") or get_col(cfg, "strain") or get_col(cfg, "sample_id")
    if not time_col or time_col not in df.columns or not group_col or group_col not in df.columns:
        print("[warn] missing time or trajectory group column, skipping velocity analysis")
        return 0

    work = df.dropna(subset=[time_col, group_col, "H", "S", "M"]).copy()
    work = work.sort_values([group_col, time_col])

    rows = []
    for gid, sub in work.groupby(group_col):
        sub = sub.sort_values(time_col)
        coords = sub[["H", "S", "M"]].to_numpy(dtype=float)
        times = sub[time_col].to_numpy(dtype=float)
        if len(sub) < 2:
            continue
        deltas = np.diff(coords, axis=0)
        dt = np.diff(times)
        dt[dt == 0] = np.nan
        step_norm = np.linalg.norm(deltas, axis=1)
        speed = step_norm / dt
        net = coords[-1] - coords[0]
        rows.append({
            "trajectory_id": gid,
            "n_points": int(len(sub)),
            "mean_step_norm": float(np.nanmean(step_norm)),
            "mean_speed": float(np.nanmean(speed)),
            "net_H": float(net[0]),
            "net_S": float(net[1]),
            "net_M": float(net[2]),
            "net_norm": float(np.linalg.norm(net)),
        })

    out = pd.DataFrame(rows)
    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_tbl.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tbl / "trajectory_velocity_summary.csv", index=False)
    print("[ok] wrote trajectory velocity summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
