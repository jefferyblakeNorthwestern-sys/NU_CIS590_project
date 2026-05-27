#!/usr/bin/env python3
"""
train.py — Executes all four training cycles and saves model artifacts.

Usage:
    python scripts/train.py \
        --input path/to/historical.csv \
        --output trained_model/ \
        [--target-fpr 0.02] \
        [--exclude-windows windows.json] \
        [--train-frac 0.60] \
        [--val-frac 0.20]

Outputs (all written to --output directory):
    column_registry.json
    baselines.json
    rate_baselines.json
    correlation_map.json
    thresholds.json
    pattern_library.json
    training_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import ingest, save_registry


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_numeric_cols(registry: dict) -> list:
    return [c for c, m in registry.items()
            if isinstance(m, dict) and m.get("is_numeric") and not m.get("is_binary") and not c.startswith("__")]

def get_binary_cols(registry: dict) -> list:
    return [c for c, m in registry.items()
            if isinstance(m, dict) and m.get("is_binary") and not c.startswith("__")]


# ── Cycle 1 — Baseline Fitting ─────────────────────────────────────────────────

def fit_baselines(train_df: pd.DataFrame, registry: dict) -> tuple:
    baselines, rate_baselines = {}, {}

    for col in get_numeric_cols(registry):
        if col not in train_df.columns:
            continue
        s = train_df[col].dropna()
        if len(s) < 10:
            continue
        baselines[col] = {
            "mean":         float(s.mean()),
            "std":          float(s.std()) or 1e-6,
            "p01":          float(s.quantile(0.01)),
            "p99":          float(s.quantile(0.99)),
            "median":       float(s.median()),
            "autocorr_lag1": float(s.autocorr(lag=1)) if len(s) > 1 else 0.0,
        }
        deltas = s.diff().dropna()
        rate_baselines[col] = {
            "mean_rate":  float(deltas.mean()),
            "std_rate":   float(deltas.std()) or 1e-6,
            "p99_rate":   float(deltas.abs().quantile(0.99)),
            "p999_rate":  float(deltas.abs().quantile(0.999)),
        }

    for col in get_binary_cols(registry):
        if col not in train_df.columns:
            continue
        vc = train_df[col].value_counts(normalize=True).to_dict()
        baselines[col] = {"value_dist": {str(k): round(v, 4) for k, v in vc.items()}}

    print(f"[cycle 1] Baselines fit for {len(baselines)} columns "
          f"({len(get_numeric_cols(registry))} numeric, {len(get_binary_cols(registry))} binary)")
    return baselines, rate_baselines


# ── Cycle 2 — Correlation Mapping ─────────────────────────────────────────────

def fit_correlations(train_df: pd.DataFrame, registry: dict,
                     min_corr: float = 0.50) -> dict:
    numeric_cols = [c for c in get_numeric_cols(registry) if c in train_df.columns]
    if len(numeric_cols) < 2:
        print("[cycle 2] Fewer than 2 numeric columns — correlation map empty.")
        return {}

    corr_matrix = train_df[numeric_cols].corr()
    correlation_map = {}

    for i, ci in enumerate(numeric_cols):
        for cj in numeric_cols[i+1:]:
            r = corr_matrix.loc[ci, cj]
            if pd.isna(r) or abs(r) < min_corr:
                continue
            key = f"{ci}::{cj}"
            correlation_map[key] = {
                "col_a":     ci,
                "col_b":     cj,
                "pearson_r": round(float(r), 4),
                "direction": "positive" if r > 0 else "negative",
            }

    print(f"[cycle 2] {len(correlation_map)} correlated pairs found (|r| ≥ {min_corr})")
    return correlation_map


# ── Deterministic layer (used in cycle 3 for threshold tuning) ────────────────

def run_deterministic(df: pd.DataFrame, baselines: dict, rate_baselines: dict,
                      correlation_map: dict, thresholds: dict, registry: dict) -> list:
    signals = []
    ts_col  = registry.get("__timestamp_col__")

    for col, stats in baselines.items():
        if col not in df.columns:
            continue

        # Z-score filter (numeric only)
        if "mean" in stats:
            series = df[col].dropna()
            z_scores = (series - stats["mean"]) / stats["std"]
            flagged  = z_scores[z_scores.abs() > thresholds["z_score_cutoff"]]
            for idx, z in flagged.items():
                ts = str(df.at[idx, ts_col]) if ts_col and ts_col in df.columns else str(idx)
                dev = abs(z) / thresholds["z_score_cutoff"]
                signals.append({
                    "column":    col,
                    "filter":    "zscore",
                    "timestamp": ts,
                    "observed":  float(df.at[idx, col]),
                    "expected":  stats["mean"],
                    "deviation": round(float(abs(z)), 3),
                    "severity":  "high" if dev > 2.5 else ("medium" if dev > 1.5 else "low"),
                })

        # Rate-of-change filter
        if col in rate_baselines:
            rb       = rate_baselines[col]
            limit    = rb["p99_rate"] * thresholds["rate_cutoff_multiplier"]
            deltas   = df[col].diff().abs()
            flagged  = deltas[deltas > limit].dropna()
            for idx, rate in flagged.items():
                ts = str(df.at[idx, ts_col]) if ts_col and ts_col in df.columns else str(idx)
                dev = rate / (limit + 1e-9)
                signals.append({
                    "column":    col,
                    "filter":    "rate_of_change",
                    "timestamp": ts,
                    "observed":  float(rate),
                    "expected":  limit,
                    "deviation": round(float(dev), 3),
                    "severity":  "high" if dev > 2.5 else ("medium" if dev > 1.5 else "low"),
                })

    # Mutual deviation filter
    mdc = thresholds["mutual_deviation_cutoff"]
    for key, pair in correlation_map.items():
        ca, cb = pair["col_a"], pair["col_b"]
        if ca not in df.columns or cb not in df.columns:
            continue
        if ca not in baselines or cb not in baselines:
            continue
        sa, sb = baselines[ca]["std"], baselines[cb]["std"]
        r      = pair["pearson_r"]
        delta_a = df[ca].diff()
        delta_b = df[cb].diff()
        expected_db = r * delta_a * (sb / (sa + 1e-9))
        residual    = (delta_b - expected_db).abs()
        threshold   = mdc * sb
        flagged     = residual[residual > threshold].dropna()
        for idx in flagged.index:
            ts = str(df.at[idx, ts_col]) if ts_col and ts_col in df.columns else str(idx)
            dev = float(residual.at[idx]) / (threshold + 1e-9)
            signals.append({
                "column":           cb,
                "filter":           "mutual_deviation",
                "timestamp":        ts,
                "observed":         float(delta_b.at[idx]) if not pd.isna(delta_b.at[idx]) else None,
                "expected":         float(expected_db.at[idx]) if not pd.isna(expected_db.at[idx]) else None,
                "deviation":        round(dev, 3),
                "severity":         "high" if dev > 2.5 else ("medium" if dev > 1.5 else "low"),
                "correlated_with":  [ca],
            })

    return signals


# ── Cycle 3 — Threshold Tuning ────────────────────────────────────────────────

def tune_thresholds(val_df: pd.DataFrame, baselines: dict, rate_baselines: dict,
                    correlation_map: dict, registry: dict,
                    target_fpr: float = 0.02, max_iterations: int = 20) -> dict:
    thresholds = {
        "z_score_cutoff":           3.0,
        "rate_cutoff_multiplier":   1.0,
        "mutual_deviation_cutoff":  2.0,
        "alarm_threshold":          0.60,
    }

    for iteration in range(max_iterations):
        signals = run_deterministic(val_df, baselines, rate_baselines,
                                    correlation_map, thresholds, registry)
        # FPR = unique rows flagged / total rows (not signal count)
        flagged_rows = len(set(s["timestamp"] for s in signals))
        fpr = flagged_rows / max(len(val_df), 1)

        print(f"[cycle 3] Iteration {iteration+1}: FPR={fpr:.4f} "
              f"(target ≤ {target_fpr})  z_cutoff={thresholds['z_score_cutoff']:.2f}")

        if fpr <= target_fpr:
            break

        excess = fpr / (target_fpr + 1e-9)
        thresholds["z_score_cutoff"]          = min(thresholds["z_score_cutoff"] * (excess ** 0.25), 6.0)
        thresholds["rate_cutoff_multiplier"]  = min(thresholds["rate_cutoff_multiplier"] * (excess ** 0.20), 4.0)
        thresholds["mutual_deviation_cutoff"] = min(thresholds["mutual_deviation_cutoff"] * (excess ** 0.20), 5.0)

    thresholds["achieved_val_fpr"]   = round(fpr, 6)
    thresholds["tuning_iterations"]  = iteration + 1
    print(f"[cycle 3] Final: FPR={fpr:.4f}  z_cutoff={thresholds['z_score_cutoff']:.3f}")
    return thresholds


# ── Cycle 4 — Pattern Library Extraction ──────────────────────────────────────

def extract_patterns(df: pd.DataFrame, anomaly_windows: list,
                     baselines: dict, registry: dict,
                     min_window_rows: int = 3) -> list:
    ts_col = registry.get("__timestamp_col__")
    if not ts_col or not anomaly_windows:
        print("[cycle 4] No anomaly windows provided — pattern library empty.")
        return []

    patterns = []
    for w in anomaly_windows:
        try:
            start = pd.to_datetime(w["start"])
            end   = pd.to_datetime(w["end"])
        except Exception:
            print(f"[cycle 4] Skipping window with unparseable timestamps: {w}")
            continue

        window_df = df[(df[ts_col] >= start) & (df[ts_col] <= end)]
        if len(window_df) < min_window_rows:
            print(f"[cycle 4] Window '{w.get('label')}' too short ({len(window_df)} rows) — skipped.")
            continue

        signature = {}
        for col, stats in baselines.items():
            if "mean" not in stats or col not in window_df.columns:
                continue
            col_mean = float(window_df[col].mean())
            z = (col_mean - stats["mean"]) / (stats["std"] + 1e-9)
            if abs(z) > 1.0:
                signature[col] = {
                    "mean_z":    round(z, 3),
                    "direction": "high" if z > 0 else "low",
                    "magnitude": "strong" if abs(z) > 3 else "moderate",
                }

        patterns.append({
            "pattern_id":    len(patterns),
            "label":         w.get("label", f"unlabeled_{len(patterns)}"),
            "signature":     signature,
            "window_length": len(window_df),
            "columns_count": len(signature),
        })
        print(f"[cycle 4] Pattern '{patterns[-1]['label']}' extracted — "
              f"{len(signature)} columns deviated")

    return patterns


# ── Chronological split ───────────────────────────────────────────────────────

def split(df: pd.DataFrame, train_frac=0.60, val_frac=0.20):
    n  = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


# ── Save artifact ─────────────────────────────────────────────────────────────

def save(obj, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / filename
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {filename} → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model.")
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          default="trained_model/")
    parser.add_argument("--target-fpr",      type=float, default=0.02)
    parser.add_argument("--exclude-windows", default=None,
                        help="JSON file with labeled anomaly windows to exclude from training")
    parser.add_argument("--train-frac",      type=float, default=0.60)
    parser.add_argument("--val-frac",        type=float, default=0.20)
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load anomaly windows if provided
    anomaly_windows = []
    if args.exclude_windows:
        with open(args.exclude_windows) as f:
            anomaly_windows = json.load(f)
        print(f"[train] Loaded {len(anomaly_windows)} labeled anomaly windows")

    # Ingest
    print("\n── Cycle 0: Ingestion ───────────────────────────────────────────────")
    df, registry = ingest(args.input)
    save_registry(registry, output_dir)

    # Exclude anomaly windows from training data
    ts_col = registry.get("__timestamp_col__")
    if anomaly_windows and ts_col:
        mask = pd.Series([True] * len(df), index=df.index)
        for w in anomaly_windows:
            s, e = pd.to_datetime(w["start"]), pd.to_datetime(w["end"])
            mask &= ~((df[ts_col] >= s) & (df[ts_col] <= e))
        excluded = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        print(f"[train] Excluded {excluded} rows from anomaly windows")

    # Split
    train_df, val_df, test_df = split(df, args.train_frac, args.val_frac)
    print(f"[train] Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    save({"train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
          "train_frac": args.train_frac, "val_frac": args.val_frac},
         output_dir, "data_split.json")

    # Cycle 1
    print("\n── Cycle 1: Baseline Fitting ────────────────────────────────────────")
    baselines, rate_baselines = fit_baselines(train_df, registry)
    save(baselines,      output_dir, "baselines.json")
    save(rate_baselines, output_dir, "rate_baselines.json")

    # Cycle 2
    print("\n── Cycle 2: Correlation Mapping ─────────────────────────────────────")
    correlation_map = fit_correlations(train_df, registry)
    save(correlation_map, output_dir, "correlation_map.json")

    # Cycle 3
    print("\n── Cycle 3: Threshold Tuning ────────────────────────────────────────")
    thresholds = tune_thresholds(val_df, baselines, rate_baselines,
                                 correlation_map, registry,
                                 target_fpr=args.target_fpr)
    save(thresholds, output_dir, "thresholds.json")

    # Cycle 4
    print("\n── Cycle 4: Pattern Library ─────────────────────────────────────────")
    patterns = extract_patterns(df, anomaly_windows, baselines, registry)
    save(patterns, output_dir, "pattern_library.json")

    # Training summary
    summary = {
        "input_file":          str(args.input),
        "columns_registered":  sum(1 for k in registry if not k.startswith("__")),
        "numeric_cols":        len(get_numeric_cols(registry)),
        "binary_cols":         len(get_binary_cols(registry)),
        "correlated_pairs":    len(correlation_map),
        "thresholds":          thresholds,
        "patterns_extracted":  len(patterns),
        "train_rows":          len(train_df),
        "val_rows":            len(val_df),
        "test_rows":           len(test_df),
    }
    save(summary, output_dir, "training_summary.json")

    print("\n── Training Complete ─────────────────────────────────────────────────")
    print(f"  Columns registered:  {summary['columns_registered']}")
    print(f"  Correlated pairs:    {summary['correlated_pairs']}")
    print(f"  Val FPR achieved:    {thresholds['achieved_val_fpr']:.4f} "
          f"(target ≤ {args.target_fpr})")
    print(f"  Patterns extracted:  {summary['patterns_extracted']}")
    print(f"  Artifacts saved to:  {output_dir}/")


if __name__ == "__main__":
    main()
