#!/usr/bin/env python3
"""
ingest.py — Schema-agnostic CSV ingestion for the anomaly detection framework.

Usage:
    # From train.py or detect.py — import and call:
    from scripts.ingest import ingest, load_registry

    df, registry = ingest("path/to/data.csv")

    # Or standalone, to inspect a new dataset:
    python scripts/ingest.py --input path/to/data.csv [--output trained_model/]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def ingest(path: str, registry_path: str = None) -> tuple:
    """
    Read a CSV file and produce a DataFrame plus a column registry.

    If registry_path is provided (detection mode), load the existing registry
    and validate that the CSV's columns are compatible. New columns in the
    CSV are noted but do not break ingestion.

    Returns:
        df:       sorted, parsed DataFrame
        registry: dict of column metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Read with flexible parsing — handle common CSV variants
    try:
        df = pd.read_csv(
            path,
            infer_datetime_format=True,
            low_memory=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to parse CSV at {path}: {e}")

    if df.empty:
        raise ValueError(f"CSV at {path} is empty.")

    print(f"[ingest] Loaded {len(df)} rows × {len(df.columns)} columns from {path.name}")

    # Build or validate column registry
    if registry_path:
        registry = load_registry(registry_path)
        registry = _validate_against_existing(df, registry)
    else:
        registry = _build_registry(df)

    # Identify and parse timestamp column
    ts_col = _detect_timestamp_column(df, registry)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True, errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)
        registry["__timestamp_col__"] = ts_col
        print(f"[ingest] Timestamp column: '{ts_col}' | Range: {df[ts_col].min()} → {df[ts_col].max()}")
    else:
        print("[ingest] Warning: no timestamp column detected. Rows will be used in file order.")

    # Report summary
    numeric_cols = [c for c, m in registry.items() if m.get("is_numeric") and not c.startswith("__")]
    binary_cols  = [c for c, m in registry.items() if m.get("is_binary")  and not c.startswith("__")]
    print(f"[ingest] {len(numeric_cols)} numeric cols | {len(binary_cols)} binary/categorical cols")

    return df, registry


def _build_registry(df: pd.DataFrame) -> dict:
    """Build a fresh column registry from a DataFrame."""
    registry = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            registry[col] = {"dtype": str(df[col].dtype), "is_numeric": False,
                             "is_binary": False, "null_rate": 1.0,
                             "n_unique": 0, "sample_vals": []}
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        unique_vals = set(series.unique())
        is_binary   = unique_vals.issubset({0, 1, True, False, 0.0, 1.0})

        registry[col] = {
            "dtype":       str(df[col].dtype),
            "is_numeric":  is_numeric,
            "is_binary":   is_binary,
            "null_rate":   round(float(df[col].isna().mean()), 4),
            "n_unique":    int(df[col].nunique()),
            "sample_vals": [_safe_val(v) for v in series.head(3).tolist()],
        }

        if is_numeric and not is_binary:
            registry[col]["variance_class"] = _variance_class(series)

    return registry


def _validate_against_existing(df: pd.DataFrame, registry: dict) -> dict:
    """
    In detection mode: check that the operational CSV is compatible with the
    trained registry. Warn on missing or new columns; do not fail hard.
    """
    trained_cols = {k for k in registry if not k.startswith("__")}
    current_cols = set(df.columns)

    missing = trained_cols - current_cols
    new_cols = current_cols - trained_cols

    if missing:
        print(f"[ingest] Warning: {len(missing)} trained columns not in current CSV — "
              f"they will be ignored in detection: {sorted(missing)}")
    if new_cols:
        print(f"[ingest] Info: {len(new_cols)} new columns in current CSV not in trained model — "
              f"they will be skipped: {sorted(new_cols)}")

    # Add new columns to registry as unmodeled (they won't be scored)
    for col in new_cols:
        series = df[col].dropna()
        registry[col] = {
            "dtype":       str(df[col].dtype),
            "is_numeric":  pd.api.types.is_numeric_dtype(df[col]),
            "is_binary":   False,
            "null_rate":   round(float(df[col].isna().mean()), 4),
            "n_unique":    int(df[col].nunique()),
            "sample_vals": [_safe_val(v) for v in series.head(3).tolist()],
            "unmodeled":   True,   # flag: excluded from detection scoring
        }

    return registry


def _detect_timestamp_column(df: pd.DataFrame, registry: dict) -> str | None:
    """Heuristically identify the timestamp column."""
    # Already set
    if "__timestamp_col__" in registry:
        col = registry["__timestamp_col__"]
        if col in df.columns:
            return col

    # Name-based heuristic
    for col in df.columns:
        if any(kw in col.lower() for kw in ("time", "date", "timestamp", "ts", "datetime")):
            try:
                pd.to_datetime(df[col].head(10), infer_datetime_format=True, errors="raise")
                return col
            except Exception:
                continue

    # Type-based heuristic
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    return None


def _variance_class(series: pd.Series) -> str:
    """Classify a numeric column's variance for human-readable registry summary."""
    cv = series.std() / (series.mean() + 1e-9)
    if cv < 0.01:
        return "near_constant"
    elif cv < 0.10:
        return "low_variance"
    elif cv < 0.50:
        return "moderate_variance"
    else:
        return "high_variance"


def _safe_val(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def load_registry(path: str) -> dict:
    """Load a saved column registry from JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Registry not found at {p}. Run training first.")
    with open(p) as f:
        return json.load(f)


def save_registry(registry: dict, output_dir: str):
    """Save column registry to the trained_model directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "column_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[ingest] Registry saved → {out / 'column_registry.json'}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and register a CSV dataset.")
    parser.add_argument("--input",   required=True, help="Path to CSV file")
    parser.add_argument("--output",  default=None,  help="Directory to save registry (optional)")
    parser.add_argument("--registry", default=None, help="Existing registry to validate against (optional)")
    args = parser.parse_args()

    df, registry = ingest(args.input, registry_path=args.registry)

    print("\n── Column Registry ──────────────────────────────────────────────────")
    for col, meta in registry.items():
        if col.startswith("__"):
            continue
        flags = []
        if meta.get("is_numeric"):  flags.append("numeric")
        if meta.get("is_binary"):   flags.append("binary")
        if meta.get("unmodeled"):   flags.append("UNMODELED")
        vc = meta.get("variance_class", "")
        print(f"  {col:<30} {', '.join(flags) or 'categorical':<20} {vc}")

    if args.output:
        save_registry(registry, args.output)
    else:
        print("\n[ingest] No --output specified. Registry not saved.")

    sys.exit(0)
