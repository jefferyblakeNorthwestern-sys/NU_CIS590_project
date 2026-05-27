#!/usr/bin/env python3
"""
detect.py — Runs the full detection pipeline on an operational CSV.

Usage:
    python scripts/detect.py \
        --input path/to/operational.csv \
        --model trained_model/ \
        [--output reports/]
        [--window-start "2017-01-16 09:00"]
        [--window-end   "2017-01-16 11:00"]

Produces a Cybersecurity Incident Report in both JSON and text formats.
If no window is specified, analyzes the entire CSV.
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.ingest import ingest
from scripts.train  import (run_deterministic, get_numeric_cols,
                             get_binary_cols, split)


# ── Load model artifacts ──────────────────────────────────────────────────────

def load_model(model_dir: str) -> dict:
    d = Path(model_dir)
    required = ["column_registry.json", "baselines.json", "rate_baselines.json",
                "correlation_map.json", "thresholds.json"]
    for f in required:
        if not (d / f).exists():
            raise FileNotFoundError(f"Missing model artifact: {d / f}. Run training first.")

    def load(name):
        with open(d / name) as f:
            return json.load(f)

    model = {
        "registry":        load("column_registry.json"),
        "baselines":       load("baselines.json"),
        "rate_baselines":  load("rate_baselines.json"),
        "correlation_map": load("correlation_map.json"),
        "thresholds":      load("thresholds.json"),
        "patterns":        load("pattern_library.json") if (d / "pattern_library.json").exists() else [],
    }
    print(f"[detect] Model loaded from {d}/")
    return model


# ── Pattern matching ──────────────────────────────────────────────────────────

def score_pattern(current_sig: dict, pattern: dict) -> dict:
    matched, partial, missed = [], [], []
    for col, pat in pattern["signature"].items():
        if col not in current_sig:
            missed.append(col)
            continue
        cur = current_sig[col]
        dir_match = (cur["observed"] > cur["expected"]) == (pat["direction"] == "high")
        if dir_match and cur.get("magnitude") == pat["magnitude"]:
            matched.append(col)
        elif dir_match:
            partial.append(col)
        else:
            missed.append(col)

    total = len(pattern["signature"])
    score = (len(matched) + 0.5 * len(partial)) / total if total > 0 else 0.0
    return {
        "pattern_id":  pattern["pattern_id"],
        "label":       pattern["label"],
        "similarity":  round(score, 3),
        "matched":     matched,
        "partial":     partial,
        "missed":      missed,
    }


def match_patterns(signals: list, baselines: dict, patterns: list) -> list:
    if not patterns or not signals:
        return []

    # Build current deviation signature
    current_sig = {}
    for s in signals:
        col = s["column"]
        if col in baselines and "mean" in baselines[col]:
            z = (s["observed"] - baselines[col]["mean"]) / (baselines[col]["std"] + 1e-9) \
                if s.get("observed") is not None else 0.0
            current_sig[col] = {
                "observed": s.get("observed"),
                "expected": s.get("expected"),
                "z":        z,
                "magnitude": "strong" if abs(z) > 3.0 else "moderate",
            }

    results = [score_pattern(current_sig, p) for p in patterns]
    results = [r for r in results if r["similarity"] >= 0.40]
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:3]


# ── Agent verdicts ────────────────────────────────────────────────────────────

def run_statistical_agent(signals: list, thresholds: dict, df: pd.DataFrame) -> dict:
    if not signals:
        return _nominal_verdict("statistical")

    high   = [s for s in signals if s["severity"] == "high"]
    medium = [s for s in signals if s["severity"] == "medium"]
    low    = [s for s in signals if s["severity"] == "low"]

    severity_score = (len(high) * 1.0 + len(medium) * 0.5 + len(low) * 0.1) / (len(signals) + 1e-9)

    # Persistence: how many unique timestamps are flagged?
    unique_ts    = len(set(s["timestamp"] for s in signals))
    total_rows   = max(len(df), 1)
    persistence  = min(unique_ts / total_rows, 1.0)

    # Spread: how many distinct columns?
    unique_cols  = len(set(s["column"] for s in signals))
    spread_score = min(unique_cols / 5.0, 1.0)   # normalize to 5 as "many"

    base = severity_score * 0.3 + persistence * 0.4 + spread_score * 0.3

    high_devs = [s["deviation"] / thresholds["z_score_cutoff"]
                 for s in high if s["filter"] == "zscore"]
    norm_dev  = float(np.mean(high_devs)) if high_devs else 1.0
    confidence = min(base * norm_dev, 1.0)

    if unique_cols == 1 and len(high) == 0:
        anomaly_class = "sensor_noise"
    elif persistence > 0.3 and unique_cols >= 3:
        anomaly_class = "cyber_event"
    elif unique_cols >= 2:
        anomaly_class = "operational_fault"
    else:
        anomaly_class = "sensor_noise"

    return {
        "agent":               "statistical",
        "confidence":          round(confidence, 3),
        "alarm_valid":         confidence >= thresholds["alarm_threshold"] * 0.8,
        "anomaly_class":       anomaly_class,
        "columns_implicated":  list(set(s["column"] for s in signals)),
        "evidence":            (
            f"{len(signals)} anomaly signals detected: {len(high)} HIGH, {len(medium)} MED, {len(low)} LOW. "
            f"Persistence: {persistence:.1%} of rows flagged. "
            f"Spread: {unique_cols} distinct columns. "
            f"Severity score: {severity_score:.3f}. "
            f"Normalized deviation (high signals): {norm_dev:.2f}×."
        ),
        "uncertainty":         (
            "Confidence would increase with longer sustained window of anomaly. "
            "Single-column anomalies are difficult to distinguish from sensor fault without "
            "correlation context."
        ),
    }


def run_behavioral_agent(signals: list, patterns: list, pattern_matches: list) -> dict:
    if not signals:
        return _nominal_verdict("behavioral")

    confidence = 0.0
    anomaly_class = "nominal"
    evidence_parts = []

    if pattern_matches:
        top = pattern_matches[0]
        confidence = top["similarity"] * 0.9
        evidence_parts.append(
            f"Top pattern match: '{top['label']}' (similarity={top['similarity']:.2f}). "
            f"Matched {len(top['matched'])} columns exactly, {len(top['partial'])} partially, "
            f"{len(top['missed'])} unmatched."
        )
        if "cyber" in top["label"].lower() or "attack" in top["label"].lower() or "falsif" in top["label"].lower():
            anomaly_class = "cyber_event"
        elif "fault" in top["label"].lower():
            anomaly_class = "operational_fault"
        else:
            anomaly_class = "cyber_event" if confidence > 0.65 else "operational_fault"
    else:
        # Cluster by co-occurrence
        from collections import Counter
        col_pairs = Counter()
        cols = [s["column"] for s in signals]
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                col_pairs[tuple(sorted([c1, c2]))] += 1
        tight_pairs = [(p, cnt) for p, cnt in col_pairs.items() if cnt >= 2]
        if tight_pairs:
            confidence    = min(0.45 + 0.05 * len(tight_pairs), 0.75)
            anomaly_class = "operational_fault"
            evidence_parts.append(
                f"No pattern library match. {len(tight_pairs)} co-occurring column pairs detected, "
                f"suggesting a clustered behavioral event rather than isolated noise."
            )
        else:
            confidence    = 0.20
            anomaly_class = "sensor_noise"
            evidence_parts.append("No pattern matches and no co-occurrence clusters — signals appear scattered.")

    return {
        "agent":               "behavioral",
        "confidence":          round(confidence, 3),
        "alarm_valid":         confidence >= 0.50,
        "anomaly_class":       anomaly_class,
        "columns_implicated":  list(set(s["column"] for s in signals)),
        "evidence":            " ".join(evidence_parts),
        "pattern_matches":     pattern_matches,
        "uncertainty":         (
            "Pattern matching is bounded by the training library size. "
            "Novel attack patterns not seen in training will produce low similarity scores "
            "even if they are genuine events."
        ),
    }


def run_correlation_agent(signals: list, correlation_map: dict, baselines: dict,
                          thresholds: dict) -> dict:
    if not signals:
        return _nominal_verdict("correlation")

    flagged_cols = set(s["column"] for s in signals)

    # Find all trained pairs that involve at least one flagged column
    relevant_pairs = {k: v for k, v in correlation_map.items()
                      if v["col_a"] in flagged_cols or v["col_b"] in flagged_cols}

    if not relevant_pairs:
        # No correlated pairs — flagged columns are orphans
        orphan_ratio = 1.0
        decoupling_score = 0.0
        evidence = (
            f"{len(flagged_cols)} flagged columns have no trained correlated partners. "
            f"Orphaned signals — more consistent with sensor noise than coordinated event."
        )
        anomaly_class = "sensor_noise"
        confidence = 0.20
    else:
        # Check mutual deviation signals for pair breakdowns
        mutual_flags = {s["column"] for s in signals if s["filter"] == "mutual_deviation"}
        pairs_broken = sum(1 for k, v in relevant_pairs.items()
                          if v["col_a"] in mutual_flags or v["col_b"] in mutual_flags)
        decoupling_score = pairs_broken / max(len(relevant_pairs), 1)

        # Orphan columns: flagged but no correlated partner also flagged
        paired_cols = set()
        for v in relevant_pairs.values():
            if v["col_a"] in flagged_cols and v["col_b"] in flagged_cols:
                paired_cols.update([v["col_a"], v["col_b"]])
        orphan_ratio = 1 - (len(paired_cols) / max(len(flagged_cols), 1))

        confidence = min(decoupling_score * 0.7 + (1 - orphan_ratio) * 0.3, 1.0)

        if decoupling_score > 0.5 and orphan_ratio < 0.4:
            anomaly_class = "cyber_event"
        elif decoupling_score > 0.2:
            anomaly_class = "operational_fault"
        else:
            anomaly_class = "sensor_noise"

        evidence = (
            f"{len(relevant_pairs)} correlated pairs involving flagged columns. "
            f"{pairs_broken} pairs broken (decoupling score: {decoupling_score:.2f}). "
            f"Orphan ratio: {orphan_ratio:.2f} "
            f"({'high — signals are isolated' if orphan_ratio > 0.6 else 'low — signals cluster together'})."
        )

    return {
        "agent":               "correlation",
        "confidence":          round(confidence, 3),
        "alarm_valid":         confidence >= 0.45,
        "anomaly_class":       anomaly_class,
        "columns_implicated":  list(flagged_cols),
        "evidence":            evidence,
        "uncertainty":         (
            "Correlation map is built on training-period data. If operational baseline has "
            "shifted legitimately, some pair decouplings may be false positives."
        ),
    }


def _nominal_verdict(agent: str) -> dict:
    return {
        "agent":               agent,
        "confidence":          0.0,
        "alarm_valid":         False,
        "anomaly_class":       "nominal",
        "columns_implicated":  [],
        "evidence":            "No anomaly signals received — no analysis performed.",
        "uncertainty":         "N/A",
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def orchestrate(verdicts: list, signals: list, thresholds: dict,
                pattern_matches: list, df: pd.DataFrame,
                registry: dict, model_dir: str) -> dict:

    # Confidence fusion
    confs = [v["confidence"] for v in verdicts]
    base  = sum(confs) / len(confs)
    n_alarming = sum(1 for v in verdicts if v["alarm_valid"])
    mult  = {0: 0.60, 1: 0.85, 2: 1.10, 3: 1.25}[n_alarming]
    fused = min(base * mult, 1.0)

    alarm_threshold = thresholds.get("alarm_threshold", 0.60)
    alarm_valid     = fused >= alarm_threshold

    # Anomaly class: majority vote
    classes    = [v["anomaly_class"] for v in verdicts]
    order      = ["cyber_event", "operational_fault", "sensor_noise", "nominal"]
    anomaly_class = max(set(classes), key=lambda c: (classes.count(c), -order.index(c)))

    # Validity
    if fused >= alarm_threshold and n_alarming >= 2:
        validity = "VALID"
    elif fused >= alarm_threshold or n_alarming >= 1:
        validity = "UNCERTAIN"
    else:
        validity = "INVALID"

    # Confidence band
    if fused >= 0.80:   band = "VERY HIGH"
    elif fused >= 0.60: band = "HIGH"
    elif fused >= 0.35: band = "MEDIUM"
    else:               band = "LOW"

    # Recommended action
    if validity == "VALID" and band in ("VERY HIGH", "HIGH"):
        action = "ESCALATE: Forward to incident response team immediately"
    elif validity in ("VALID", "UNCERTAIN"):
        action = "INVESTIGATE: Analyst review required before escalation decision"
    elif band == "MEDIUM":
        action = "MONITOR: Continue observation; re-evaluate at next detection cycle"
    else:
        action = "DISMISS: Insufficient evidence; log and continue"

    # Earliest timestamp
    ts_col = registry.get("__timestamp_col__")
    if signals and ts_col:
        timestamps = sorted([s["timestamp"] for s in signals if s.get("timestamp")])
        onset_ts   = timestamps[0] if timestamps else "unknown"
    else:
        onset_ts = "unknown"

    # Primary vs secondary indicators
    col_counts = {}
    for s in signals:
        col_counts[s["column"]] = col_counts.get(s["column"], 0) + 1
    sorted_cols = sorted(col_counts.items(), key=lambda x: -x[1])
    primary   = [c for c, _ in sorted_cols[:3]]
    secondary = [c for c, _ in sorted_cols[3:]]

    # Correlated pair breakdowns
    flagged_cols = set(s["column"] for s in signals)
    mutual_flags = {s["column"] for s in signals if s["filter"] == "mutual_deviation"}
    pair_breakdowns = []
    for k, pair in model_dir["correlation_map"].items():
        if (pair["col_a"] in flagged_cols or pair["col_b"] in flagged_cols) and \
           (pair["col_a"] in mutual_flags or pair["col_b"] in mutual_flags):
            pair_breakdowns.append(pair)

    # All uncertainties
    uncertainties = list(set(v["uncertainty"] for v in verdicts if v["uncertainty"] != "N/A"))

    return {
        "report_id":          str(uuid.uuid4())[:8].upper(),
        "generated":          datetime.utcnow().isoformat() + "Z",
        "fused_confidence":   round(fused, 3),
        "alarm_valid":        alarm_valid,
        "alarm_threshold":    alarm_threshold,
        "thresholds":         thresholds,
        "validity":           validity,
        "anomaly_class":      anomaly_class,
        "confidence_band":    band,
        "n_alarming":         n_alarming,
        "multiplier":         mult,
        "base_confidence":    round(base, 3),
        "action":             action,
        "onset_timestamp":    onset_ts,
        "primary_indicators": primary,
        "secondary_indicators": secondary,
        "pair_breakdowns":    pair_breakdowns,
        "pattern_matches":    pattern_matches,
        "uncertainties":      uncertainties,
        "verdicts":           verdicts,
        "signals":            signals,
    }


# ── Report formatting ─────────────────────────────────────────────────────────

def format_report(r: dict, df: pd.DataFrame, data_source: str,
                  window_start, window_end) -> str:
    def sep(title=""):
        line = "─" * 70
        return f"\n  {title}\n{line}" if title else f"\n{line}"

    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  CYBERSECURITY ANOMALY DETECTION REPORT                              ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        f"  REPORT ID:       {r['report_id']}",
        f"  GENERATED:       {r['generated']}",
        f"  DATA SOURCE:     {Path(data_source).name}",
        f"  ANALYSIS WINDOW: {window_start} → {window_end}  ({len(df)} records)",
    ]

    lines.append(sep("SECTION 1 — VERDICT"))
    alarm_str = "⚠  ACTIVE ALARM" if r["alarm_valid"] else "✓  NO ALARM"
    lines += [
        f"  ALARM STATUS:    {alarm_str}",
        f"  VALIDITY:        {r['validity']}",
        f"  ANOMALY CLASS:   {r['anomaly_class'].upper().replace('_', ' ')}",
        f"  CONFIDENCE:      {r['fused_confidence']:.2f}  ({r['confidence_band']})",
    ]

    lines.append(sep("SECTION 2 — INCIDENT SUMMARY"))
    pm_str = (f"{r['pattern_matches'][0]['label']} (similarity={r['pattern_matches'][0]['similarity']:.2f})"
              if r["pattern_matches"] else "No match")
    n_flagged = len(set(s["column"] for s in r["signals"]))
    n_total   = sum(1 for k in r["verdicts"][0].get("columns_implicated", []))
    lines += [
        f"  EVENT TYPE:       {r['anomaly_class'].replace('_', ' ').title()}",
        f"  PATTERN MATCH:    {pm_str}",
        f"  ONSET TIMESTAMP:  {r['onset_timestamp']}",
        f"  SCOPE:            {n_flagged} columns flagged  |  "
        f"{len(r['signals'])} total anomaly signals",
        "",
        "  NARRATIVE:",
        f"    Analysis of {len(df)} records identified {len(r['signals'])} anomaly signals "
        f"across {n_flagged} columns.",
        f"    Fused confidence: {r['fused_confidence']:.2f} ({r['confidence_band']}) — "
        f"{r['n_alarming']}/3 agents raised alarms.",
        f"    Classification: {r['anomaly_class'].replace('_', ' ')}. "
        f"Recommended action: {r['action'].split(':')[0]}.",
    ]

    lines.append(sep("SECTION 3 — TECHNICAL INDICATORS"))
    lines.append("  PRIMARY INDICATORS:")
    for col in r["primary_indicators"]:
        col_sigs = [s for s in r["signals"] if s["column"] == col]
        for s in col_sigs[:2]:
            lines.append(
                f"    {col:<30}  filter={s['filter']:<16}  "
                f"deviation={s['deviation']:.2f}  severity={s['severity'].upper()}"
            )
            lines.append(
                f"    {'':30}  observed={s.get('observed', 'N/A')}  "
                f"expected={s.get('expected', 'N/A')}  ts={s['timestamp']}"
            )

    if r["secondary_indicators"]:
        lines.append("  SECONDARY INDICATORS:")
        for col in r["secondary_indicators"][:5]:
            col_sigs = [s for s in r["signals"] if s["column"] == col]
            if col_sigs:
                s = col_sigs[0]
                lines.append(
                    f"    {col:<30}  filter={s['filter']:<16}  "
                    f"deviation={s['deviation']:.2f}  severity={s['severity'].upper()}"
                )

    lines.append("  CORRELATED PAIR BREAKDOWNS:")
    if r["pair_breakdowns"]:
        for p in r["pair_breakdowns"][:5]:
            lines.append(f"    {p['col_a']} ↔ {p['col_b']}  trained_r={p['pearson_r']}")
    else:
        lines.append("    No significant correlation breakdowns detected")

    lines.append(sep("SECTION 4 — AGENT VERDICTS"))
    for v in r["verdicts"]:
        lines += [
            f"  {v['agent'].upper()} AGENT",
            f"    Confidence:    {v['confidence']:.2f}",
            f"    Alarm Valid:   {'YES' if v['alarm_valid'] else 'NO'}",
            f"    Class:         {v['anomaly_class'].upper().replace('_', ' ')}",
            f"    Finding:       {v['evidence'][:200]}{'...' if len(v['evidence']) > 200 else ''}",
            f"    Uncertainty:   {v['uncertainty'][:150]}{'...' if len(v['uncertainty']) > 150 else ''}",
            "",
        ]
        if v["agent"] == "behavioral" and v.get("pattern_matches"):
            lines.append("    Top Pattern Matches:")
            for pm in v["pattern_matches"]:
                lines.append(
                    f"      {pm['label']:<35} similarity={pm['similarity']:.2f}  "
                    f"matched={len(pm['matched'])}/{len(pm['matched'])+len(pm['partial'])+len(pm['missed'])}"
                )
            lines.append("")

    lines.append("  FUSION SUMMARY")
    lines += [
        f"    Base confidence (avg):     {r['base_confidence']:.3f}",
        f"    Agents alarming:           {r['n_alarming']}/3",
        f"    Agreement multiplier:      {r['multiplier']}",
        f"    Fused confidence:          {r['fused_confidence']:.3f}",
        f"    Alarm threshold:           {r['alarm_threshold']}",
    ]

    lines.append(sep("SECTION 5 — ANALYST GUIDANCE"))
    lines += [
        f"  RECOMMENDED ACTION:",
        f"    {r['action']}",
        "",
        "  KEY UNCERTAINTIES:",
    ]
    for u in r["uncertainties"]:
        lines.append(f"    • {u[:160]}")

    lines += [
        "",
        "  ADDITIONAL CONTEXT REQUESTED:",
        "    • Were control commands or maintenance activities scheduled in this window?",
        "    • Are any flagged columns known to have intermittent sensor issues?",
        "    • Have operational parameters changed recently (new equipment, process changes)?",
    ]

    lines.append(sep("SECTION 6 — METADATA"))
    lines += [
        f"  Model artifacts:      {model_dir if isinstance(model_dir, str) else 'trained_model/'}",
        f"  Detection cycle:      {r['generated']}",
        f"  Data source:          {Path(data_source).name}",
        f"  Records analyzed:     {len(df)}",
        f"  Window:               {window_start} -> {window_end}",
        f"  Total signals fired:  {len(r['signals'])}",
        f"    HIGH:               {sum(1 for s in r['signals'] if s['severity'] == 'high')}",
        f"    MEDIUM:             {sum(1 for s in r['signals'] if s['severity'] == 'medium')}",
        f"    LOW:                {sum(1 for s in r['signals'] if s['severity'] == 'low')}",
        f"  Pattern library:      {len(r['pattern_matches'])} matches found",
        f"  Alarm threshold:      {r['alarm_threshold']}",
        f"  z_score_cutoff:       {r.get('thresholds', {}).get('z_score_cutoff', 'N/A')}",
        "",
        "═" * 72,
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection on operational CSV.")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--model",        default="trained_model/")
    parser.add_argument("--output",       default="reports/")
    parser.add_argument("--window-start", default=None)
    parser.add_argument("--window-end",   default=None)
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Ingest operational CSV using existing registry
    df, registry = ingest(args.input,
                          registry_path=str(Path(args.model) / "column_registry.json"))

    # Window filter
    ts_col = registry.get("__timestamp_col__")
    if args.window_start or args.window_end:
        if not ts_col:
            print("[detect] Warning: no timestamp column found; ignoring window filters.")
        else:
            if args.window_start:
                df = df[df[ts_col] >= pd.to_datetime(args.window_start)]
            if args.window_end:
                df = df[df[ts_col] <= pd.to_datetime(args.window_end)]
            df = df.reset_index(drop=True)

    window_start = str(df[ts_col].min()) if ts_col and ts_col in df.columns else "unknown"
    window_end   = str(df[ts_col].max()) if ts_col and ts_col in df.columns else "unknown"
    print(f"[detect] Analysis window: {window_start} → {window_end}  ({len(df)} rows)")

    # Deterministic layer
    signals = run_deterministic(df, model["baselines"], model["rate_baselines"],
                                model["correlation_map"], model["thresholds"], registry)
    print(f"[detect] Deterministic layer: {len(signals)} anomaly signals "
          f"({sum(1 for s in signals if s['severity']=='high')} HIGH)")

    # Pattern matching
    pattern_matches = match_patterns(signals, model["baselines"], model["patterns"])

    # Reasoning agents (parallel in production; sequential here for simplicity)
    stat_verdict  = run_statistical_agent(signals, model["thresholds"], df)
    behav_verdict = run_behavioral_agent(signals, model["patterns"], pattern_matches)
    corr_verdict  = run_correlation_agent(signals, model["correlation_map"],
                                          model["baselines"], model["thresholds"])
    verdicts = [stat_verdict, behav_verdict, corr_verdict]

    # Pass model dict reference for pair breakdown lookup
    result = orchestrate(verdicts, signals, model["thresholds"],
                         pattern_matches, df, registry, model)

    # Format and save
    report_text = format_report(result, df, args.input, window_start, window_end)
    print("\n" + report_text)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_id = result["report_id"]
    with open(out_dir / f"report_{report_id}.txt", "w") as f:
        f.write(report_text)
    with open(out_dir / f"report_{report_id}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[detect] Report saved → {out_dir}/report_{report_id}.{{txt,json}}")


if __name__ == "__main__":
    main()
