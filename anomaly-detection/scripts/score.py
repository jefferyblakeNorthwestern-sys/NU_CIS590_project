#!/usr/bin/env python3
"""
score.py -- Score detection reports against BATADAL ground truth attack labels.

Usage:
    python3 anomaly-detection/scripts/score.py \
        --reports reports/run5/ \
        --labels data/BATADAL_test_attack_list.csv \
        --data data/BATADAL_test_dataset.CSV \
        [--ablation] [--detail] [--disagreement]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_attack_windows(labels_path):
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]
    start_col = next(c for c in df.columns if "Starting" in c or "Start" in c)
    end_col   = next(c for c in df.columns if "Ending" in c or "End" in c)
    id_col    = next(c for c in df.columns if c.strip() == "ID")
    desc_col  = next((c for c in df.columns if "Description" in c), None)
    windows = []
    for _, row in df.iterrows():
        try:
            start = pd.to_datetime(str(row[start_col]).strip(), dayfirst=True)
            end   = pd.to_datetime(str(row[end_col]).strip(), dayfirst=True)
            windows.append({
                "id":          int(row[id_col]),
                "start":       start,
                "end":         end,
                "duration_h":  int((end - start).total_seconds() / 3600),
                "description": str(row[desc_col]).strip() if desc_col else "",
            })
        except Exception as e:
            print("  [score] Warning: could not parse row: " + str(e))
    print("[score] Loaded " + str(len(windows)) + " attack windows from " + Path(labels_path).name)
    return windows


def build_ground_truth(df, ts_col, attack_windows):
    labels = pd.Series(0, index=df.index)
    for w in attack_windows:
        mask = (df[ts_col] >= w["start"]) & (df[ts_col] <= w["end"])
        labels[mask] = 1
    return labels


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
        tp    = first_alert is not None
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
    ttd_vals  = [r["ttd_hours"] for r in attack_results if r["ttd_hours"] is not None]
    mean_ttd  = round(sum(ttd_vals) / len(ttd_vals), 1) if ttd_vals else None
    med_ttd   = round(float(np.median(ttd_vals)), 1)    if ttd_vals else None

    return {
        "config":        config_name,
        "tp":            tp_count,
        "fn":            fn_count,
        "fp":            fp_count,
        "recall":        round(recall, 3),
        "precision":     round(precision, 3),
        "f1":            round(f1, 3),
        "mean_ttd_h":    mean_ttd,
        "median_ttd_h":  med_ttd,
        "attack_detail": attack_results,
    }


def _ts_in_window(ts_str, start, end):
    try:
        ts = pd.to_datetime(ts_str)
        return start <= ts <= end
    except Exception:
        return False


def analyze_disagreement(reports, attack_windows, df, ts_col):
    agent_names = ["statistical", "behavioral", "correlation"]
    results = []

    for w in attack_windows:
        # Find best matching report for this window
        best_report = reports[0]
        best_count  = 0
        for report in reports:
            count = sum(
                1 for s in report.get("signals", [])
                if _ts_in_window(s.get("timestamp"), w["start"], w["end"])
            )
            if count > best_count:
                best_count  = count
                best_report = report

        verdicts   = best_report.get("verdicts", [])
        agents_out = {}
        alarming   = []

        for agent in agent_names:
            v = next((x for x in verdicts if x["agent"] == agent), None)
            if v:
                conf  = round(v.get("confidence", 0.0), 3)
                alarm = v.get("alarm_valid", False)
                aclass = v.get("anomaly_class", "unknown")
                agents_out[agent] = {"confidence": conf, "alarm_valid": alarm, "anomaly_class": aclass}
                if alarm:
                    alarming.append(agent)
            else:
                agents_out[agent] = {"confidence": 0.0, "alarm_valid": False, "anomaly_class": "unknown"}

        n_alarming = len(alarming)

        if n_alarming == 3:
            pattern   = "unanimous_alarm"
            unanimous = True
        elif n_alarming == 0:
            pattern   = "unanimous_no_alarm"
            unanimous = True
        elif n_alarming == 2:
            dissenter = [a for a in agent_names if a not in alarming][0]
            pattern   = "majority_alarm_dissent_" + dissenter
            unanimous = False
        else:
            pattern   = "lone_alarm_" + alarming[0]
            unanimous = False

        results.append({
            "attack_id":       w["id"],
            "start":           str(w["start"]),
            "end":             str(w["end"]),
            "duration_h":      w["duration_h"],
            "description":     w["description"][:60],
            "agents":          agents_out,
            "alarming_agents": alarming,
            "agreement_count": n_alarming,
            "pattern":         pattern,
            "unanimous":       unanimous,
        })

    return results


def print_disagreement(results):
    agent_names = ["statistical", "behavioral", "correlation"]
    print("")
    print("=" * 100)
    print("  AGENT DISAGREEMENT ANALYSIS")
    print("=" * 100)
    print("  Attack   Pattern                                    Stat        Behav       Corr        Agree")
    print("  " + "-" * 96)

    for r in results:
        agents = r["agents"]
        rows = []
        for agent in agent_names:
            a     = agents.get(agent, {})
            conf  = a.get("confidence", 0.0)
            alarm = "Y" if a.get("alarm_valid") else "N"
            rows.append(str(round(conf, 2)) + alarm)

        print("  Attack {:>2}  {:<45} {:<10} {:<10} {:<10} {}/3".format(
            r["attack_id"],
            r["pattern"],
            rows[0], rows[1], rows[2],
            r["agreement_count"]
        ))

    print("=" * 100)
    print("")

    unanimous     = sum(1 for r in results if r["unanimous"])
    disagreements = [r for r in results if not r["unanimous"]]
    patterns      = {}
    for r in disagreements:
        patterns[r["pattern"]] = patterns.get(r["pattern"], 0) + 1

    print("  Summary:")
    print("  Unanimous verdicts : " + str(unanimous) + "/" + str(len(results)))
    print("  Disagreements      : " + str(len(disagreements)) + "/" + str(len(results)))

    if patterns:
        print("  Disagreement patterns:")
        for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print("    " + pat + " : " + str(count) + "x")

    print("")
    print("  Per-agent statistics across all attack windows:")
    for agent in agent_names:
        alarmed = sum(1 for r in results if r["agents"].get(agent, {}).get("alarm_valid"))
        confs   = [r["agents"].get(agent, {}).get("confidence", 0.0) for r in results]
        mean_c  = round(sum(confs) / len(confs), 3) if confs else 0.0
        max_c   = round(max(confs), 3) if confs else 0.0
        min_c   = round(min(confs), 3) if confs else 0.0
        print("  " + agent.capitalize().ljust(16) +
              " alarmed=" + str(alarmed) + "/" + str(len(results)) +
              "  mean_conf=" + str(mean_c) +
              "  max=" + str(max_c) +
              "  min=" + str(min_c))
    print("")


def print_table(results):
    print("")
    print("=" * 90)
    print("  ABLATION STUDY RESULTS")
    print("=" * 90)
    print("  {:<30} {:>4} {:>4} {:>4} {:>8} {:>10} {:>6} {:>10}".format(
        "Configuration", "TP", "FN", "FP", "Recall", "Precision", "F1", "Mean TTD"))
    print("  " + "-" * 86)
    for r in results:
        ttd = str(r["mean_ttd_h"]) + "h" if r["mean_ttd_h"] is not None else "N/A"
        print("  {:<30} {:>4} {:>4} {:>4} {:>8.3f} {:>10.3f} {:>6.3f} {:>10}".format(
            r["config"], r["tp"], r["fn"], r["fp"],
            r["recall"], r["precision"], r["f1"], ttd))
    print("=" * 90)


def print_attack_detail(result):
    print("  Per-attack detail -- " + result["config"])
    print("  " + "-" * 80)
    for a in result["attack_detail"]:
        status = "DETECTED" if a["detected"] else "MISSED  "
        ttd    = "TTD=" + str(a["ttd_hours"]) + "h" if a["ttd_hours"] is not None else "TTD=N/A"
        print("  Attack {:>2}  {}  {:<12}  {}".format(
            a["attack_id"], status, ttd, a["description"]))


def signals_deterministic_only(report):
    return report.get("signals", [])

def signals_single_agent(report, agent_name):
    v = next((x for x in report.get("verdicts", []) if x["agent"] == agent_name), None)
    if v and v.get("alarm_valid"):
        return report.get("signals", [])
    return []

def signals_full_pipeline(report):
    if report.get("alarm_valid"):
        return report.get("signals", [])
    return []

def signals_zscore_baseline(report):
    return [s for s in report.get("signals", [])
            if s.get("filter") == "zscore" and s.get("severity") == "high"]


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
        print("[score] No report JSON files found in " + str(report_dir))
        sys.exit(1)

    print("[score] Found " + str(len(report_files)) + " report(s) in " + str(report_dir))

    all_reports = []
    all_signals = []
    for rf in report_files:
        with open(rf) as f:
            report = json.load(f)
        all_reports.append(report)
        all_signals.extend(report.get("signals", []))

    print("[score] Total signals: " + str(len(all_signals)))

    if args.disagreement:
        dis_results = analyze_disagreement(all_reports, attack_windows, df, ts_col)
        print_disagreement(dis_results)
        out_path = report_dir / "disagreement_results.json"
        with open(out_path, "w") as f:
            json.dump(dis_results, f, indent=2, default=str)
        print("[score] Disagreement results saved -> " + str(out_path))

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
        print("[score] Ablation results saved -> " + str(out_path))


if __name__ == "__main__":
    main()
