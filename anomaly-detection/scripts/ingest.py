#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def ingest(path: str, registry_path: str = None) -> tuple:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV at {path}: {e}")
    if df.empty:
        raise ValueError(f"CSV at {path} is empty.")
    print(f"[ingest] Loaded {len(df)} rows x {len(df.columns)} columns from {path.name}")

    if registry_path:
        registry = load_registry(registry_path)
        registry = _validate_against_existing(df, registry)
    else:
        registry = _build_registry(df)

    ts_col = _detect_timestamp_column(df, registry)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)
        registry["__timestamp_col__"] = ts_col
        print(f"[ingest] Timestamp column: '{ts_col}' | Range: {df[ts_col].min()} -> {df[ts_col].max()}")
    else:
        print("[ingest] Warning: no timestamp column detected.")

    numeric_cols = [c for c, m in registry.items()
                    if isinstance(m, dict) and m.get("is_numeric") and not c.startswith("__")]
    binary_cols  = [c for c, m in registry.items()
                    if isinstance(m, dict) and m.get("is_binary") and not c.startswith("__")]
    print(f"[ingest] {len(numeric_cols)} numeric cols | {len(binary_cols)} binary/categorical cols")
    return df, registry


def _build_registry(df: pd.DataFrame) -> dict:
    registry = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            registry[col] = {"dtype": str(df[col].dtype), "is_numeric": False,
                             "is_binary": False, "null_rate": 1.0,
                             "n_unique": 0, "sample_vals": []}
            continue
        is_numeric  = pd.api.types.is_numeric_dtype(df[col])
        unique_vals = set(series.unique())
        is_binary   = unique_vals.issubset({0, 1, True, False, 0.0, 1.0})
        registry[col] = {
            "dtype":      str(df[col].dtype),
            "is_numeric": is_numeric,
            "is_binary":  is_binary,
            "null_rate":  round(float(df[col].isna().mean()), 4),
            "n_unique":   int(df[col].nunique()),
            "sample_vals": [_safe_val(v) for v in series.head(3).tolist()],
        }
        if is_numeric and not is_binary:
            registry[col]["variance_class"] = _variance_class(series)
    return registry


def _validate_against_existing(df: pd.DataFrame, registry: dict) -> dict:
    trained_cols = {k for k in registry if not k.startswith("__")}
    current_cols = set(df.columns)
    missing  = trained_cols - current_cols
    new_cols = current_cols - trained_cols
    if missing:
        print(f"[ingest] Warning: {len(missing)} trained columns missing: {sorted(missing)}")
    if new_cols:
        print(f"[ingest] Info: {len(new_cols)} new columns not in model: {sorted(new_cols)}")
    for col in new_cols:
        series = df[col].dropna()
        registry[col] = {
            "dtype":      str(df[col].dtype),
            "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
            "is_binary":  False,
            "null_rate":  round(float(df[col].isna().mean()), 4),
            "n_unique":   int(df[col].nunique()),
            "sample_vals": [_safe_val(v) for v in series.head(3).tolist()],
            "unmodeled":  True,
        }
    return registry


def _detect_timestamp_column(df: pd.DataFrame, registry: dict) -> Optional[str]:
    if "__timestamp_col__" in registry:
        col = registry["__timestamp_col__"]
        if col in df.columns:
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in ("time", "date", "timestamp", "ts", "datetime")):
            try:
                pd.to_datetime(df[col].head(10), errors="raise")
                return col
            except Exception:
                continue
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _variance_class(series: pd.Series) -> str:
    cv = series.std() / (series.mean() + 1e-9)
    if cv < 0.01:   return "near_constant"
    elif cv < 0.10: return "low_variance"
    elif cv < 0.50: return "moderate_variance"
    else:           return "high_variance"


def _safe_val(v):
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_):    return bool(v)
    return v


def load_registry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Registry not found at {p}. Run training first.")
    with open(p) as f:
        return json.load(f)


def save_registry(registry: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "column_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[ingest] Registry saved -> {out / 'column_registry.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()
    df, registry = ingest(args.input, registry_path=args.registry)
    if args.output:
        save_registry(registry, args.output)
    sys.exit(0)
