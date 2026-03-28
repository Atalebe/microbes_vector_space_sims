from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if not np.isfinite(mad) or mad == 0:
        std = np.nanstd(s)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=series.index, dtype=float)
        return (s - np.nanmean(s)) / std
    return 0.67448975 * (s - med) / mad



def signed_log1p(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))



def safe_log(series: pd.Series, offset: float = 1e-9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(np.clip(s, a_min=offset, a_max=None))


TRANSFORMS = {
    "identity": lambda s: pd.to_numeric(s, errors="coerce"),
    "log": safe_log,
    "signed_log1p": signed_log1p,
    "robust_z": robust_zscore,
}



def apply_transform(series: pd.Series, transform_name: str) -> pd.Series:
    if transform_name not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform_name}")
    return TRANSFORMS[transform_name](series)



def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")



def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported write type: {suffix}")



def dump_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



def get_col(cfg: Dict[str, Any], key: str) -> Optional[str]:
    return cfg.get("columns", {}).get(key)



def make_state_table(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    coord_cfg = cfg.get("coordinates", {})
    for coord in ["H", "S", "M"]:
        spec = coord_cfg.get(coord, {})
        source = spec.get("source_column")
        transform = spec.get("transform", "identity")
        if source is None or source not in out.columns:
            raise KeyError(f"Missing source column for {coord}: {source}")
        out[coord] = apply_transform(out[source], transform)
        if spec.get("normalize", "robust_z") == "robust_z":
            out[coord] = robust_zscore(out[coord])
    r_spec = coord_cfg.get("R_explicit", {})
    r_col = r_spec.get("source_column")
    if r_col and r_col in out.columns:
        out["R_explicit"] = apply_transform(out[r_col], r_spec.get("transform", "identity"))
        if r_spec.get("normalize", "robust_z") == "robust_z":
            out["R_explicit"] = robust_zscore(out["R_explicit"])
    return out



def subset_required(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df.dropna(subset=keep).copy()
