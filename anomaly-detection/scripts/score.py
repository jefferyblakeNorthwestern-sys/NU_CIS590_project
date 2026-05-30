#!/usr/bin/env python3
"""
score.py -- Score detection reports against BATADAL ground truth attack labels.
Supports full pipeline scoring, ablation study, and agent disagreement analysis.

Usage:
    # Score a single report
    python3 anomaly-detection/scripts/score.py \
        --reports reports/run5/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV

    # Full ablation study
    python3 anomaly-detection/scripts/score.py \
        --reports reports/run5/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV --ablation --detail

    # Agent disagreement analysis
    python3 anomaly-detection/scripts/score.py \
        --reports reports/run5/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV --disagreement
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# -- Parse attack label file ---------------------------------------------------

def load_attack_windows(labels_path):
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]
    start_col = next(c for c in df.columns if "Starting" in c or "Start" in c)
    end_col   = next(c for c in df.columns if "Ending"   in c or "End"   in c)
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
            print(f"  [score] Warning: could not parse row: {e}")

    print(f"[score] Loaded {len(windows)} attack windows from {Path(labels_path).name}")
    return windows


# -- Build ground truth series -------------------------------------------------

def build_ground_truth(df, ts_col, attack_windows):
    labels = pd.Series(0, index=df.index)
    for w in attack_windows:
        mask = (df[ts_col] >= w["start"]) & (df[ts_col] <= w["end"])
        labels[mask] = 1
    return labels


# -- Score signals against ground truth ----------------------------------------

def score_signals(signals, df, ts_col, attack_windows, config_name):
    ground_truth = build_ground_truth(df, ts_col, attack_windows)
    flagged_ts = set()
    for s in signals:
        try:
            flagged_ts.add(pd.to_datetime(s["timestamp"]))
        except Exception:
            pass

    attack_results = []
    for w in attack_windows:
        first_alert = None
        for ts in sorted(flagged_ts):
            if w["start"] <= ts <= w["end"]:
                first_alert = ts
                break
        tp = first_alert is not None
        ttd_h = int((first_alert - w["start"]).total_seconds() / 3600) if tp else None
        attack_results.append({
            "attack_id":   w["id"],
            "start":       str(w["start"]),
            "end":         str(w["end"]),
            "duration_h":  w["duration_h"],
            "detected":    tp,
            "first_alert": str(first_alert) if first_alert else None,
            "ttd_hours":   ttd_h,
            "description": w["description"][:60],
        })

    n_attacks = len(attack_windows)
    tp_count  = sum(1 for r in attack_results if r["detected"])
    fn_count  = n_attacks - tp_count
    fp_count  = sum(
        1 for ts in flagged_ts
        if ts in df[ts_col].values and
        ground_truth[df[df[ts_col] == ts].index].values[0] == 0
    )

    recall    = tp_count / n_attacks if n_attacks > 0 else 0.0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    ttd_values   = [r["ttd_hours"] for r in attack_results if r["ttd_hours"] is not None]
    mean_ttd     = round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    median_ttd   = round(float(np.median(ttd_values)), 1)       if ttd_values else None

    return {
        "config":        config_name,
        "tp":            tp_count,
        "fn":            fn_count,
        "fp":            fp_count,
        "recall":        round(recall, 3),
        "precision":     round(precision, 3),
        "f1":            round(f1, 3),
        "mean_ttd_h":    mean_ttd,
        "median_ttd_h":  median_ttd,
        "attack_detail": attack_results,
    }


# -- Disagreement analysis -----------------------------------------------------

def analyze_disagreement(reports, attack_windows, df, ts_col):
    """
    For each attack window, extract each agent's confidence and alarm_valid
    from the report verdicts. Identify and categorize disagreement patterns.
    """
    agent_names = ["statistical", "behavioral", "correlation"]
    results = []

    for w in attack_windows:
        window_result = {
            "attack_id":   w["id"],
            "start":       str(w["start"]),
            "end":         str(w["end"]),
            "duration_h":  w["duration_h"],
            "description": w["description"][:60],
            "agents":      {},
            "pattern":     None,
            "unanimous":   None,
            "agreement_count": 0,
        }

        # Find report(s) whose analysis window overlaps this attack window
        # Use all reports and pick the one with signals in this window
        best_report = None
        best_signal_count = 0
        for report in reports:
            signals_in_window = [
                s for s in report.get("signals", [])
                if _ts_in_window(s.get("timestamp"), w["start"], w["end"])
            ]
            if len(signals_in_window) > best_signal_count:
                best_signal_count = len(signals_in_window)
                best_report = report

        if best_report is None:
            best_report = reports[0]  # fallback to first report

        # Extract per-agent verdicts
        verdicts = best_report.get("verdicts", [])
        alarming = []
        for agent in agent_names:
            v = next((x for x in verdicts if x["agent"] == agent), None)
            if v:
                conf         = round(v.get("confidence", 0.0), 3)
                alarm_valid  = v.get("alarm_valid", False)
                anomaly_class = v.get("anomaly_class", "unknown")
                window_result["agents"][agent] = {
                    "confidence":    conf,
                    "alarm_valid":   alarm_valid,
                    "anomaly_class": anomaly_class,
                }
                if alarm_valid:
                    alarming.append(agent)
            else:
                window_result["agents"][agent] = {
                    "confidence":    0.0,
                    "alarm_valid":   False,
                    "anomaly_class": "unknown",
                }

        n_alarming = len(alarming)
        window_result["agreement_count"] = n_alarming
        window_result["alarming_agents"] = alarming

        # Classify disagreement pattern
        if n_alarming == 3:
            window_result["pattern"] = "unanimous_alarm"
            window_result["unanimous"] = True
        elif n_alarming == 0:
            window_result["pattern"] = "unanimous_no_alarm"
            window_result["unanimous"] = True
        elif n_alarming == 2:
            dissenter = [a for a in agent_names if a not in alarming][0]
            window_result["pattern"] = f"majority_alarm_dissent_{dissenter}"
            window_result["unanimous"] = False
        else:
            lone = alarming[0]
            window_result["pattern"] = f"lone_alarm_{lone}"
            window_result["unanimous"] = False

        results.append(window_result)

    return results


def _ts_in_window(ts_str, start, end):
    try:
        ts = pd.to_datetime(ts_str)
        return start <= ts <= end
    except Exception:
        return False


def print_disagreement(results):
    print()
    print("=" * 90)
    print("  AGENT DISAGREEMENT ANALYSIS")
    print("=" * 90)
    print(f"  {'Attack':<8} {'Pattern':<40} {'Stat':>6} {'Behav':>6} {'Corr':>6} {'Agree':>6}")
    print("  " + "-" * 86)

    for r in results:
        agents = r["agents"]
        stat_conf  = agents.get("statistical",  {}).get("confidence", 0.0)
        behav_conf = agents.get("behavioral",   {}).get("confidence", 0.0)
        corr_conf  = agents.get("correlation",  {}).get("confidence", 0.0)
        stat_alarm  = "Y" if agents.get("statistical",  {}).get("alarm_valid") else "N"
        behav_alarm = "Y" if agents.get("behavioral",   {}).get("alarm_valid") else "N"
        corr_alarm  = "Y" if agents.get("correlation",  {}).get("alarm_valid") else "N"

        print(f"  Attack {r['attack_id']:>2}  {r['pattern']:<40} "
              f"{stat_conf:>4.2f}{stat_alarm}  {behav_conf:>4.2f}{behav_alarm}  "
              f"{corr_conf:>4.2f}{corr_alarm}  {r['agreement_count']}/3")

    print("=" * 90)

    # Summary statistics
    unanimous     = sum(1 for r in results if r["unanimous"])
    disagreements = [r for r in results if not r["unanimous"]]
    patterns      = {}
    for r in disagreements:
        patterns[r["pattern"]] = patterns.get(r["pattern"], 0) + 1

    print(f"
  Unanimous verdicts:    {unanimous}/{len(results)}")
    print(f"  Disagreements:         {len(disagreements)}/{len(results)}")
    if patterns:
        print("  Disagreement patterns:")
        for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"    {pat:<45} {count}x")

    # Per-agent alarm rate
    print()
    for agent in ["statistical", "behavioral", "correlation"]:
        alarmed = sum(1 for r in results if r["agents"].get(agent, {}).get("alarm_valid"))
        confs   = [r["agents"].get(agent, {}).get("confidence", 0.0) for r in results]
        print(f"  {agent.capitalize():<14} alarmed {alarmed}/{len(results)} windows  "
              f"mean_conf={round(sum(confs)/len(confs), 3) if confs else 0:.3f}  "
              f"max_conf={max(confs) if confs else 0:.3f}  "
              f"min_conf={min(confs) if confs else 0:.3f}")
    print()


# -- Ablation signal extractors ------------------------------------------------

def signals_deterministic_only(report):
    return report.get("signals", [])

def signals_single_agent(report, agent_name):
    verdicts = report.get("verdicts", [])
    agent = next((v for v in verdicts if v["agent"] == agent_name), None)
    if agent and agent.get("alarm_valid"):
        return report.get("signals", [])
    return []

def signals_full_pipeline(report):
    if report.get("alarm_valid"):
        return report.get("signals", [])
    return []

def signals_zscore_baseline(report):
    return [s for s in report.get("signals", [])
            if s.get("filter") == "zscore" and s.get("severity") == "high"]


# -- Print table ---------------------------------------------------------------

def print_table(results):
    print()
    print("=" * 90)
    print("  ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"  {'Configuration':<30} {'TP':>4} {'FN':>4} {'FP':>4} {'Recall':>8} {'Precision':>10} {'F1':>6} {'Mean TTD':>10}")
    print("  " + "-" * 86)
    for r in results:
        ttd = f"{r['mean_ttd_h']}h" if r['mean_ttd_h'] is not None else "N/A"
        print(f"  {r['config']:<30} {r['tp']:>4} {r['fn']:>4} {r['fp']:>4} "
              f"{r['recall']:>8.3f} {r['precision']:>10.3f} {r['f1']:>6.3f} {ttd:>10}")
    print("=" * 90)


def print_attack_detail(result):
    print(f"
  Per-attack detail -- {result['config']}")
    print("  " + "-" * 80)
    for a in result["attack_detail"]:
        status = "DETECTED" if a["detected"] else "MISSED  "
        ttd    = f"TTD={a['ttd_hours']}h" if a["ttd_hours"] is not None else "TTD=N/A"
        print(f"  Attack {a['attack_id']:>2}  {status}  {ttd:<10}  {a['description']}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports",      required=True)
    parser.add_argument("--labels",       required=True)
    parser.add_argument("--data",         required=True)
    parser.add_argument("--ablation",     action="store_true")
    parser.add_argument("--detail",       action="store_true")
    parser.add_argument("--disagreement", action="store_true")
    args = parser.parse_args()

    attack_windows = load_attack_windows(args.labels)

    df = pd.read_csv(args.data, low_memory=False)
    ts_col = next((c for c in df.columns
                   if any(kw in c.lower() for kw in ("time","date","datetime","ts"))), None)
    if not ts_col:
        print("[score] Error: no timestamp column found.")
        sys.exit(1)
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)

    report_dir   = Path(args.reports)
    report_files = sorted(report_dir.glob("report_*.json"))
    if not report_files:
        print(f"[score] No report JSON files found in {report_dir}")
        sys.exit(1)

    print(f"[score] Found {len(report_files)} report(s) in {report_dir}")

    all_reports = []
    all_signals = []
    for rf in report_files:
        with open(rf) as f:
            report = json.load(f)
        all_reports.append(report)
        all_signals.extend(report.get("signals", []))

    print(f"[score] Total signals: {len(all_signals)}")

    # Disagreement analysis
    if args.disagreement:
        disagreement_results = analyze_disagreement(
            all_reports, attack_windows, df, ts_col
        )
        print_disagreement(disagreement_results)

        out_path = report_dir / "disagreement_results.json"
        with open(out_path, "w") as f:
            json.dump(disagreement_results, f, indent=2, default=str)
        print(f"[score] Disagreement results saved -> {out_path}")

    # Ablation study
    if args.ablation or not args.disagreement:
        report = all_reports[0]
        configs = [
            ("1. Z-score baseline",      signals_zscore_baseline(report)),
            ("2. Deterministic only",     signals_deterministic_only(report)),
            ("3. Statistical agent only", signals_single_agent(report, "statistical")),
            ("4. Behavioral agent only",  signals_single_agent(report, "behavioral")),
            ("5. Correlation agent only", signals_single_agent(report, "correlation")),
            ("6. Full pipeline",          signals_full_pipeline(report)),
        ]

        ablation_results = []
        for name, sigs in configs:
            result = score_signals(sigs, df, ts_col, attack_windows, name)
            ablation_results.append(result)
            if args.detail:
                print_attack_detail(result)

        print_table(ablation_results)

        out_path = report_dir / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(ablation_results, f, indent=2, default=str)
        print(f"[score] Ablation results saved -> {out_path}")


if __name__ == "__main__":
    main()
