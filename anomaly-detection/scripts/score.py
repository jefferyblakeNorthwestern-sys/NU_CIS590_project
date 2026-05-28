#!/usr/bin/env python3
"""
score.py — Score detection reports against BATADAL ground truth attack labels.
Supports full pipeline scoring and ablation study configurations.

Usage:
    # Score a single report against the test attack list
    python3 anomaly-detection/scripts/score.py \
        --reports reports/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV

    # Run ablation study (all four configurations)
    python3 anomaly-detection/scripts/score.py \
        --reports reports/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV \
        --ablation
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


# ── Parse attack label file ────────────────────────────────────────────────────

def load_attack_windows(labels_path: str) -> list:
    """
    Parse a BATADAL attack list CSV into a list of attack window dicts.
    Handles both dataset2 (dd/mm/YY HH) and test (dd/mm/YY HH) formats.
    """
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]

    # Find start/end columns flexibly
    start_col = next(c for c in df.columns if "Starting" in c or "Start" in c)
    end_col   = next(c for c in df.columns if "Ending" in c or "End" in c)
    id_col    = next(c for c in df.columns if c.strip() == "ID")
    desc_col  = next((c for c in df.columns if "Description" in c), None)

    windows = []
    for _, row in df.iterrows():
        try:
            start = pd.to_datetime(str(row[start_col]).strip(), dayfirst=True)
            end   = pd.to_datetime(str(row[end_col]).strip(),   dayfirst=True)
            windows.append({
                "id":          int(row[id_col]),
                "start":       start,
                "end":         end,
                "duration_h":  int((end - start).total_seconds() / 3600),
                "description": str(row[desc_col]).strip() if desc_col else "",
            })
        except Exception as e:
            print(f"  [score] Warning: could not parse row {row[id_col]}: {e}")

    print(f"[score] Loaded {len(windows)} attack windows from {Path(labels_path).name}")
    return windows


# ── Build per-timestep ground truth series ────────────────────────────────────

def build_ground_truth(df: pd.DataFrame, ts_col: str,
                        attack_windows: list) -> pd.Series:
    """
    For each row in df, return 1 if it falls within any attack window, else 0.
    """
    labels = pd.Series(0, index=df.index)
    for w in attack_windows:
        mask = (df[ts_col] >= w["start"]) & (df[ts_col] <= w["end"])
        labels[mask] = 1
    return labels


# ── Score a set of AnomalySignals against ground truth ────────────────────────

def score_signals(signals: list, df: pd.DataFrame, ts_col: str,
                  attack_windows: list, config_name: str) -> dict:
    """
    Given a list of AnomalySignal dicts, compute per-attack and aggregate metrics.
    A true positive = at least one signal fired within the attack window.
    TTD = first signal timestamp - attack start timestamp.
    """
    ground_truth = build_ground_truth(df, ts_col, attack_windows)

    # Map signal timestamps to row indices
    flagged_ts = set()
    for s in signals:
        try:
            ts = pd.to_datetime(s["timestamp"])
            flagged_ts.add(ts)
        except Exception:
            pass

    # Per-attack scoring
    attack_results = []
    for w in attack_windows:
        window_ts = df[(df[ts_col] >= w["start"]) & (df[ts_col] <= w["end"])][ts_col]
        first_alert = None
        for ts in sorted(flagged_ts):
            if w["start"] <= ts <= w["end"]:
                first_alert = ts
                break

        tp = first_alert is not None
        ttd_h = int((first_alert - w["start"]).total_seconds() / 3600) if tp else None

        attack_results.append({
            "attack_id":    w["id"],
            "start":        str(w["start"]),
            "end":          str(w["end"]),
            "duration_h":   w["duration_h"],
            "detected":     tp,
            "first_alert":  str(first_alert) if first_alert else None,
            "ttd_hours":    ttd_h,
            "description":  w["description"][:60] if w["description"] else "",
        })

    # Aggregate metrics
    n_attacks  = len(attack_windows)
    tp_count   = sum(1 for r in attack_results if r["detected"])
    fn_count   = n_attacks - tp_count

    # False positives: flagged timestamps outside any attack window
    attack_mask = ground_truth == 1
    fp_count = sum(
        1 for ts in flagged_ts
        if ts in df[ts_col].values and
        ground_truth[df[df[ts_col] == ts].index].values[0] == 0
    )

    recall    = tp_count / n_attacks if n_attacks > 0 else 0.0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    ttd_values = [r["ttd_hours"] for r in attack_results if r["ttd_hours"] is not None]
    mean_ttd   = round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    median_ttd = round(float(np.median(ttd_values)), 1)       if ttd_values else None

    return {
        "config":         config_name,
        "tp":             tp_count,
        "fn":             fn_count,
        "fp":             fp_count,
        "recall":         round(recall, 3),
        "precision":      round(precision, 3),
        "f1":             round(f1, 3),
        "mean_ttd_h":     mean_ttd,
        "median_ttd_h":   median_ttd,
        "attack_detail":  attack_results,
    }


# ── Extract signals for each ablation configuration ───────────────────────────

def signals_deterministic_only(report: dict) -> list:
    """All signals from the deterministic layer — no agent filtering."""
    return report.get("signals", [])


def signals_single_agent(report: dict, agent_name: str) -> list:
    """
    Simulate single-agent alarm: return all signals if the named agent alarmed,
    else return empty list (no alarm).
    """
    verdicts = report.get("verdicts", [])
    agent = next((v for v in verdicts if v["agent"] == agent_name), None)
    if agent and agent.get("alarm_valid"):
        return report.get("signals", [])
    return []


def signals_full_pipeline(report: dict) -> list:
    """Full pipeline: only return signals if the orchestrator raised an alarm."""
    if report.get("alarm_valid"):
        return report.get("signals", [])
    return []


def signals_zscore_baseline(report: dict) -> list:
    """Baseline: z-score filter only, alarm on any HIGH signal."""
    high_signals = [s for s in report.get("signals", [])
                    if s.get("filter") == "zscore" and s.get("severity") == "high"]
    return high_signals


# ── Print results table ───────────────────────────────────────────────────────

def print_table(results: list):
    print()
    print("=" * 90)
    print("  ABLATION STUDY RESULTS")
    print("=" * 90)
    header = f"  {'Configuration':<30} {'TP':>4} {'FN':>4} {'FP':>4} {'Recall':>8} {'Precision':>10} {'F1':>6} {'Mean TTD':>10}"
    print(header)
    print("  " + "-" * 86)
    for r in results:
        ttd = f"{r['mean_ttd_h']}h" if r['mean_ttd_h'] is not None else "N/A"
        print(f"  {r['config']:<30} {r['tp']:>4} {r['fn']:>4} {r['fp']:>4} "
              f"{r['recall']:>8.3f} {r['precision']:>10.3f} {r['f1']:>6.3f} {ttd:>10}")
    print("=" * 90)
    print()


def print_attack_detail(result: dict):
    print(f"\n  Per-attack detail — {result['config']}")
    print("  " + "-" * 80)
    for a in result["attack_detail"]:
        status = "DETECTED" if a["detected"] else "MISSED  "
        ttd    = f"TTD={a['ttd_hours']}h" if a["ttd_hours"] is not None else "TTD=N/A"
        print(f"  Attack {a['attack_id']:>2}  {status}  {ttd:<10}  {a['description']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score detection reports against BATADAL labels.")
    parser.add_argument("--reports", required=True,  help="Directory containing report JSON files")
    parser.add_argument("--labels",  required=True,  help="BATADAL attack list CSV")
    parser.add_argument("--data",    required=True,  help="Operational CSV (for timestamp reference)")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--detail",   action="store_true", help="Print per-attack breakdown")
    args = parser.parse_args()

    # Load attack windows
    attack_windows = load_attack_windows(args.labels)

    # Load operational data for timestamp reference
    df = pd.read_csv(args.data, low_memory=False)
    ts_col = next((c for c in df.columns
                   if any(kw in c.lower() for kw in ("time","date","datetime","ts"))), None)
    if not ts_col:
        print("[score] Error: no timestamp column found in data CSV.")
        sys.exit(1)
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Load all report JSON files
    report_dir = Path(args.reports)
    report_files = sorted(report_dir.glob("report_*.json"))
    if not report_files:
        print(f"[score] No report JSON files found in {report_dir}")
        sys.exit(1)

    print(f"[score] Found {len(report_files)} report(s) in {report_dir}")

    # Merge signals across all reports (in case of multiple detection runs)
    all_signals = []
    all_reports = []
    for rf in report_files:
        with open(rf) as f:
            report = json.load(f)
        all_reports.append(report)
        all_signals.extend(report.get("signals", []))

    print(f"[score] Total signals across all reports: {len(all_signals)}")

    if args.ablation:
        # Use the first report for agent-level ablation
        report = all_reports[0]

        configs = [
            ("1. Z-score baseline",      signals_zscore_baseline(report)),
            ("2. Deterministic only",     signals_deterministic_only(report)),
            ("3. Statistical agent only", signals_single_agent(report, "statistical")),
            ("4. Behavioral agent only",  signals_single_agent(report, "behavioral")),
            ("5. Correlation agent only", signals_single_agent(report, "correlation")),
            ("6. Full pipeline",          signals_full_pipeline(report)),
        ]

        results = []
        for name, sigs in configs:
            result = score_signals(sigs, df, ts_col, attack_windows, name)
            results.append(result)
            if args.detail:
                print_attack_detail(result)

        print_table(results)

        # Save results
        out_path = report_dir / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[score] Results saved -> {out_path}")

    else:
        # Single report scoring
        result = score_signals(all_signals, df, ts_col, attack_windows, "Full pipeline")
        print_table([result])
        if args.detail:
            print_attack_detail(result)

        out_path = report_dir / "score_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[score] Results saved -> {out_path}")


if __name__ == "__main__":
    main()
