from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(processed_dir)
    ensure_dir(logs_dir)

    cond = pd.read_csv(f"{processed_dir}/gse206609_explicit_R_condition_summary.csv")
    final_rank = cond.loc[cond["drug"] != "none"].copy()
    final_rank = final_rank.sort_values(
        ["explicit_R_observed", "R_mean"],
        ascending=[False, False]
    ).reset_index(drop=True)
    final_rank["final_explicit_R_rank_desc"] = final_rank.index + 1
    final_rank.to_csv(f"{processed_dir}/gse206609_explicit_R_final_ranking.csv", index=False)

    summary = {
        "dataset_name": cfg["dataset_name"],
        "final_rank_desc": final_rank["drug"].tolist(),
        "note": "Untreated control excluded from final peer recoverability ranking."
    }

    with open(f"{logs_dir}/explicit_R_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[ok] wrote:", f"{processed_dir}/gse206609_explicit_R_final_ranking.csv")
    print("[ok] wrote:", f"{logs_dir}/explicit_R_final_summary.json")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
