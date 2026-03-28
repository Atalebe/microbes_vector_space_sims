#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import load_config, read_table, subset_required, write_table


def classify_residual(x: float, q1: float, q4: float) -> str:
    if x <= q1:
        return "burdened_or_fragile"
    if x >= q4:
        return "restorative_or_coherent"
    return "intermediate"



def main(config_path: str) -> int:
    cfg = load_config(config_path)
    df = read_table(Path(cfg["io"]["processed_dir"]) / "state_table.parquet")
    df = subset_required(df, ["H", "S", "M"])

    model = LinearRegression()
    X = df[["H", "M"]].values
    y = df["S"].values
    model.fit(X, y)
    pred = model.predict(X)
    df["S_pred"] = pred
    df["R_latent"] = df["S"] - df["S_pred"]

    q1 = float(df["R_latent"].quantile(0.25))
    q4 = float(df["R_latent"].quantile(0.75))
    df["residual_class"] = [classify_residual(x, q1, q4) for x in df["R_latent"]]

    out_tbl = Path(cfg["io"]["results_dir"]) / "tables"
    out_tbl.mkdir(parents=True, exist_ok=True)
    write_table(df, out_tbl / "residual_field.csv")

    pd.DataFrame({
        "term": ["intercept", "coef_H", "coef_M", "q1", "q4"],
        "value": [model.intercept_, model.coef_[0], model.coef_[1], q1, q4],
    }).to_csv(out_tbl / "residual_model_summary.csv", index=False)

    print("[ok] wrote latent recoverability field")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
