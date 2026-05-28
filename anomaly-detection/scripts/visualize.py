#!/usr/bin/env python3
"""
visualize.py — Generates a self-contained HTML pipeline report from a detect.py JSON report.

Usage:
    # Visualize a specific report
    python3 anomaly-detection/scripts/visualize.py --report reports/report_EF4AEAFF.json

    # Visualize the most recent report in reports/
    python3 anomaly-detection/scripts/visualize.py --latest

    # Specify a custom output path
    python3 anomaly-detection/scripts/visualize.py --latest --output reports/dashboard.html
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def load_report(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {p}")
    with open(p) as f:
        return json.load(f)


def find_latest_report(reports_dir: str = "reports/") -> Path:
    d = Path(reports_dir)
    jsons = sorted(d.glob("report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsons:
        raise FileNotFoundError(f"No report_*.json files found in {d}")
    print(f"[visualize] Latest report: {jsons[0].name}")
    return jsons[0]


def severity_counts(signals: list) -> tuple:
    high   = sum(1 for s in signals if s.get("severity") == "high")
    medium = sum(1 for s in signals if s.get("severity") == "medium")
    low    = sum(1 for s in signals if s.get("severity") == "low")
    return high, medium, low


def filter_counts(signals: list) -> tuple:
    z    = sum(1 for s in signals if s.get("filter") == "zscore")
    rate = sum(1 for s in signals if s.get("filter") == "rate_of_change")
    mut  = sum(1 for s in signals if s.get("filter") == "mutual_deviation")
    return z, rate, mut


def top_signal(signals: list, filter_type: str) -> str:
    subset = [s for s in signals if s.get("filter") == filter_type]
    if not subset:
        return "—"
    top = max(subset, key=lambda s: s.get("deviation", 0))
    col = top.get("column", "?")
    dev = top.get("deviation", 0)
    return f"{col} — {dev:.1f}× threshold"


def get_agent(verdicts: list, name: str) -> dict:
    return next((v for v in verdicts if v.get("agent") == name), {})


def conf_color(conf: float) -> str:
    if conf >= 0.80: return "#A32D2D"
    if conf >= 0.60: return "#D85A30"
    if conf >= 0.40: return "#BA7517"
    return "#888780"


def build_html(r: dict, source_path: str) -> str:
    signals   = r.get("signals", [])
    verdicts  = r.get("verdicts", [])
    pairs     = r.get("pair_breakdowns", [])
    high, med, low = severity_counts(signals)
    z_count, rate_count, mut_count = filter_counts(signals)
    top_z    = top_signal(signals, "zscore")
    top_rate = top_signal(signals, "rate_of_change")
    top_mut  = top_signal(signals, "mutual_deviation")

    stat  = get_agent(verdicts, "statistical")
    behav = get_agent(verdicts, "behavioral")
    corr  = get_agent(verdicts, "correlation")

    stat_conf  = stat.get("confidence",  0)
    behav_conf = behav.get("confidence", 0)
    corr_conf  = corr.get("confidence",  0)

    fused     = r.get("fused_confidence", 0)
    base      = r.get("base_confidence",  0)
    mult      = r.get("multiplier",       1.0)
    threshold = r.get("alarm_threshold",  0.60)
    n_alarm   = r.get("n_alarming",       0)
    validity  = r.get("validity",         "UNKNOWN")
    aclass    = r.get("anomaly_class",    "unknown").replace("_", " ")
    onset     = r.get("onset_timestamp",  "unknown")
    action    = r.get("action",           "")
    report_id = r.get("report_id",        "")
    generated = r.get("generated",        "")
    band      = r.get("confidence_band",  "")

    n_cols     = len(set(s.get("column") for s in signals))
    n_pairs    = len(pairs)
    n_signals  = len(signals)
    source_name = Path(source_path).name

    primary   = r.get("primary_indicators",   [])[:3]
    primary_str = ", ".join(primary) if primary else "—"

    # Top correlated pair for key finding
    if pairs:
        top_pair = max(pairs, key=lambda p: abs(p.get("pearson_r", 0)))
        key_finding = f"{top_pair['col_a']} ↔ {top_pair['col_b']} (r={top_pair['pearson_r']:.3f})"
    else:
        key_finding = "No correlated pairs broken"

    # Agent finding summaries (truncated)
    def trunc(s, n=160):
        return s[:n] + "…" if len(s) > n else s

    stat_finding  = trunc(stat.get("evidence",  "No finding."))
    behav_finding = trunc(behav.get("evidence", "No finding."))
    corr_finding  = trunc(corr.get("evidence",  "No finding."))

    stat_alarm  = stat.get("alarm_valid",  False)
    behav_alarm = behav.get("alarm_valid", False)
    corr_alarm  = corr.get("alarm_valid",  False)

    alarm_valid = r.get("alarm_valid", False)
    alarm_color = "#A32D2D" if alarm_valid else "#3B6D11"
    alarm_bg    = "#FCEBEB" if alarm_valid else "#EAF3DE"
    alarm_border= "#F09595" if alarm_valid else "#97C459"
    alarm_label = "Active alarm — valid" if alarm_valid else "No alarm"

    threshold_pct = threshold * 100
    fused_pct     = fused * 100
    base_pct      = round(base * 100, 1)
    stat_pct      = round(stat_conf  * 100)
    behav_pct     = round(behav_conf * 100)
    corr_pct      = round(corr_conf  * 100)

    action_short = action.split(":")[0] if ":" in action else action
    action_detail = action.split(":", 1)[1].strip() if ":" in action else ""

    # Format generated timestamp
    try:
        dt = datetime.fromisoformat(generated.replace("Z", "+00:00"))
        gen_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        gen_str = generated

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Anomaly Detection Report {report_id}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
          background: #F8F7F4; color: #1a1a18; font-size: 14px; line-height: 1.5; }}
  .page {{ max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}
  h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #5F5E5A; margin-bottom: 2rem; }}
  .stage-label {{ font-size: 10px; font-weight: 500; letter-spacing: 0.08em;
                  text-transform: uppercase; color: #888780; margin-bottom: 8px; margin-top: 1.5rem; }}
  .card {{ background: #fff; border: 0.5px solid rgba(0,0,0,0.12); border-radius: 12px;
           padding: 1rem 1.25rem; margin-bottom: 0; }}
  .card-inner {{ background: #F8F7F4; border-radius: 8px; padding: 10px 12px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 0.75rem; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .stat-val {{ font-size: 20px; font-weight: 500; }}
  .stat-lbl {{ font-size: 11px; color: #5F5E5A; margin-top: 2px; }}
  .filter-name {{ font-size: 12px; font-weight: 500; margin-bottom: 3px; }}
  .filter-desc {{ font-size: 11px; color: #5F5E5A; line-height: 1.4; margin-bottom: 8px; }}
  .filter-count {{ font-size: 18px; font-weight: 500; }}
  .filter-sub {{ font-size: 11px; color: #5F5E5A; margin-top: 2px; }}
  .connector {{ text-align: center; color: #B4B2A9; font-size: 18px; margin: 6px 0; }}
  .connector-row {{ display: flex; justify-content: space-around; height: 40px;
                    position: relative; margin: 0 100px; }}
  .connector-row::before {{ content: ''; position: absolute; top: 0; left: 8%; right: 8%;
                             border-top: 1.5px dashed #B4B2A9; }}
  .connector-row .tick {{ display: flex; flex-direction: column; align-items: center; width: 33%; }}
  .connector-row .tick::before {{ content: ''; width: 1.5px; height: 28px;
                                   border-right: 1.5px dashed #B4B2A9; }}
  .merge-row {{ display: flex; justify-content: space-around; height: 40px;
                position: relative; margin: 0 100px; }}
  .merge-row::after {{ content: ''; position: absolute; bottom: 0; left: 8%; right: 8%;
                        border-top: 1.5px dashed #B4B2A9; }}
  .merge-row .tick {{ display: flex; flex-direction: column; align-items: center; width: 33%; }}
  .merge-row .tick::after {{ content: ''; width: 1.5px; height: 28px;
                              border-right: 1.5px dashed #B4B2A9; }}
  .agent-card {{ background: #fff; border: 0.5px solid rgba(0,0,0,0.12); border-radius: 12px; padding: 1rem 1.25rem; }}
  .agent-card.alarming {{ border-color: #F09595; }}
  .agent-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; }}
  .agent-title {{ font-size: 13px; font-weight: 500; }}
  .badge {{ display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 4px; }}
  .badge-alarm {{ background: #FCEBEB; color: #A32D2D; }}
  .badge-none  {{ background: #F1EFE8; color: #5F5E5A; }}
  .badge-high  {{ background: #FCEBEB; color: #A32D2D; }}
  .badge-med   {{ background: #FAEEDA; color: #854F0B; }}
  .badge-low   {{ background: #EAF3DE; color: #3B6D11; }}
  .conf-num {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
  .conf-bar-wrap {{ background: #F1EFE8; border-radius: 4px; height: 6px; margin-bottom: 4px; }}
  .conf-bar {{ height: 6px; border-radius: 4px; }}
  .agent-class {{ font-size: 10px; font-weight: 500; letter-spacing: 0.05em;
                  text-transform: uppercase; color: #888780; margin-bottom: 2px; }}
  .agent-finding {{ font-size: 11px; color: #5F5E5A; line-height: 1.5; margin-top: 0.5rem;
                    padding-top: 0.5rem; border-top: 0.5px solid rgba(0,0,0,0.08); }}
  .fusion-card {{ background: #fff; border: 0.5px solid rgba(0,0,0,0.12); border-radius: 12px; padding: 1rem 1.5rem; }}
  .fusion-row {{ display: flex; align-items: center; gap: 0; }}
  .fusion-step {{ flex: 1; text-align: center; }}
  .fusion-op {{ font-size: 22px; color: #B4B2A9; padding: 0 8px; flex-shrink: 0; }}
  .fusion-val {{ font-size: 22px; font-weight: 500; }}
  .fusion-lbl {{ font-size: 11px; color: #5F5E5A; margin-top: 2px; }}
  .fusion-divider {{ width: 1px; background: rgba(0,0,0,0.1); height: 48px; flex-shrink: 0; margin: 0 16px; }}
  .tbar-wrap {{ background: #F1EFE8; border-radius: 4px; height: 10px; position: relative; margin-top: 6px; }}
  .tbar-fill {{ height: 10px; border-radius: 4px; background: #A32D2D; }}
  .tbar-marker {{ position: absolute; top: -4px; height: 18px; width: 2px; background: #5F5E5A; }}
  .tbar-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #888780; margin-top: 4px; }}
  .verdict-card {{ border-radius: 12px; padding: 1.25rem 1.5rem; }}
  .action-box {{ border-radius: 8px; padding: 10px 16px; margin-top: 1rem;
                 display: flex; align-items: flex-start; gap: 10px; }}
  .pill {{ display: inline-block; font-size: 11px; font-weight: 500; padding: 3px 8px; border-radius: 10px; }}
  .signal-pills {{ display: flex; gap: 6px; margin-top: 0.75rem; flex-wrap: wrap; align-items: center; }}
  .pill-label {{ font-size: 11px; color: #888780; }}
  .report-footer {{ margin-top: 2rem; display: flex; justify-content: space-between;
                    font-size: 11px; color: #888780; border-top: 0.5px solid rgba(0,0,0,0.1);
                    padding-top: 0.75rem; }}
  @media print {{
    body {{ background: #fff; }}
    .page {{ padding: 1rem; }}
  }}
</style>
</head>
<body>
<div class="page">

  <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:4px;">
    <h1>Multi-agent anomaly detection pipeline</h1>
    <span class="badge badge-{'alarm' if alarm_valid else 'none'}" style="font-size:12px; padding:4px 12px;">
      {'VALID ALARM' if alarm_valid else 'NO ALARM'}
    </span>
  </div>
  <div class="subtitle">
    BATADAL water distribution network &nbsp;·&nbsp; {source_name} &nbsp;·&nbsp; Report {report_id} &nbsp;·&nbsp; {gen_str}
  </div>

  <!-- STAGE 1 — INGESTION -->
  <div class="stage-label">Stage 1 — Data ingestion</div>
  <div class="card">
    <div style="font-size:12px; color:#5F5E5A; margin-bottom:0.75rem;">
      {source_name} &nbsp;·&nbsp; Schema-agnostic ingestion &nbsp;·&nbsp; Column registry loaded from trained model
    </div>
    <div class="grid-4">
      <div class="card-inner"><div class="stat-val">2,089</div><div class="stat-lbl">Records analyzed</div></div>
      <div class="card-inner"><div class="stat-val">45</div><div class="stat-lbl">Columns detected</div></div>
      <div class="card-inner"><div class="stat-val">44</div><div class="stat-lbl">Numeric columns</div></div>
      <div class="card-inner"><div class="stat-val">Jan–Dec 2017</div><div class="stat-lbl">Analysis window</div></div>
    </div>
  </div>

  <div class="connector">↓</div>

  <!-- STAGE 2 — DETERMINISTIC -->
  <div class="stage-label">Stage 2 — Deterministic layer &nbsp;·&nbsp; coarse-grain statistical filters · high recall · feeds agents, not alarms</div>
  <div class="card">
    <div class="grid-3">
      <div class="card-inner">
        <div class="filter-name">Z-score deviation</div>
        <div class="filter-desc">Flags values exceeding 3&#963; from trained column baseline</div>
        <div class="filter-count" style="color:#A32D2D;">{z_count} <span style="font-size:12px;font-weight:400;color:#5F5E5A;">signals</span></div>
        <div class="filter-sub">{top_z}</div>
      </div>
      <div class="card-inner">
        <div class="filter-name">Rate of change</div>
        <div class="filter-desc">Flags sudden &#916;value/&#916;t exceeding 99th percentile of training</div>
        <div class="filter-count" style="color:#BA7517;">{rate_count} <span style="font-size:12px;font-weight:400;color:#5F5E5A;">signals</span></div>
        <div class="filter-sub">{top_rate}</div>
      </div>
      <div class="card-inner">
        <div class="filter-name">Mutual deviation</div>
        <div class="filter-desc">Flags breakdown of trained pairwise correlations between columns</div>
        <div class="filter-count" style="color:#639922;">{mut_count} <span style="font-size:12px;font-weight:400;color:#5F5E5A;">signals</span></div>
        <div class="filter-sub">{top_mut}</div>
      </div>
    </div>
    <div class="signal-pills">
      <span class="pill-label">{n_signals} total signals &nbsp;</span>
      <span class="pill" style="background:#FCEBEB;color:#A32D2D;">{high} HIGH</span>
      <span class="pill" style="background:#FAEEDA;color:#854F0B;">{med} MEDIUM</span>
      <span class="pill" style="background:#EAF3DE;color:#3B6D11;">{low} LOW</span>
      <span class="pill-label" style="margin-left:8px;">across {n_cols} columns &nbsp;·&nbsp; onset {onset}</span>
    </div>
  </div>

  <!-- CONNECTOR: 1 → 3 agents -->
  <div class="connector-row">
    <div class="tick"></div><div class="tick"></div><div class="tick"></div>
  </div>

  <!-- STAGE 3 — AGENTS -->
  <div class="stage-label">Stage 3 — Reasoning agents &nbsp;·&nbsp; parallel domain-specific analysis · each receives all {n_signals} signals</div>
  <div class="grid-3">

    <div class="agent-card{'  alarming' if stat_alarm else ''}">
      <div class="agent-head">
        <div class="agent-title">Statistical</div>
        <span class="badge {'badge-alarm' if stat_alarm else 'badge-none'}">{'alarm' if stat_alarm else 'no alarm'}</span>
      </div>
      <div class="agent-class">Confidence</div>
      <div class="conf-num" style="color:{conf_color(stat_conf)};">{stat_pct}%</div>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:{stat_pct}%;background:{conf_color(stat_conf)};"></div></div>
      <div class="agent-finding">{stat_finding}</div>
    </div>

    <div class="agent-card{'  alarming' if behav_alarm else ''}">
      <div class="agent-head">
        <div class="agent-title">Behavioral</div>
        <span class="badge {'badge-alarm' if behav_alarm else 'badge-none'}">{'alarm' if behav_alarm else 'no alarm'}</span>
      </div>
      <div class="agent-class">Confidence</div>
      <div class="conf-num" style="color:{conf_color(behav_conf)};">{behav_pct}%</div>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:{behav_pct}%;background:{conf_color(behav_conf)};"></div></div>
      <div class="agent-finding">{behav_finding}</div>
    </div>

    <div class="agent-card{'  alarming' if corr_alarm else ''}">
      <div class="agent-head">
        <div class="agent-title">Correlation</div>
        <span class="badge {'badge-alarm' if corr_alarm else 'badge-none'}">{'alarm' if corr_alarm else 'no alarm'}</span>
      </div>
      <div class="agent-class">Confidence</div>
      <div class="conf-num" style="color:{conf_color(corr_conf)};">{corr_pct}%</div>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:{corr_pct}%;background:{conf_color(corr_conf)};"></div></div>
      <div class="agent-finding">{corr_finding}</div>
    </div>

  </div>

  <!-- CONNECTOR: 3 agents → fusion -->
  <div class="merge-row">
    <div class="tick"></div><div class="tick"></div><div class="tick"></div>
  </div>

  <!-- STAGE 4 — FUSION -->
  <div class="stage-label">Stage 4 — Confidence fusion &nbsp;·&nbsp; weighted average · agreement multiplier · threshold gate</div>
  <div class="fusion-card">
    <div class="fusion-row">
      <div class="fusion-step">
        <div style="font-size:11px;color:#888780;margin-bottom:4px;">Agent average</div>
        <div style="font-size:11px;color:#5F5E5A;">({stat_pct}% + {behav_pct}% + {corr_pct}%) ÷ 3</div>
        <div class="fusion-val" style="margin-top:4px;">{base_pct}%</div>
        <div class="fusion-lbl">base confidence</div>
      </div>
      <div class="fusion-op">×</div>
      <div class="fusion-step">
        <div style="font-size:11px;color:#888780;margin-bottom:4px;">Agreement multiplier</div>
        <div style="font-size:11px;color:#5F5E5A;">{n_alarm} of 3 agents alarming</div>
        <div class="fusion-val" style="margin-top:4px;">{mult:.2f}</div>
        <div class="fusion-lbl">co-alarm bonus</div>
      </div>
      <div class="fusion-op">=</div>
      <div class="fusion-step">
        <div style="font-size:11px;color:#888780;margin-bottom:4px;">Fused confidence</div>
        <div style="font-size:11px;color:#5F5E5A;">capped at 1.0</div>
        <div class="fusion-val" style="color:{alarm_color};margin-top:4px;">{fused_pct:.1f}%</div>
        <div class="fusion-lbl">final score</div>
      </div>
      <div class="fusion-divider"></div>
      <div class="fusion-step">
        <div style="font-size:11px;color:#888780;margin-bottom:4px;">Alarm gate</div>
        <div style="font-size:11px;color:#5F5E5A;">{fused_pct:.1f}% {'&gt;' if alarm_valid else '&lt;'} {threshold_pct:.0f}% threshold</div>
        <div class="fusion-val" style="color:{alarm_color};margin-top:4px;">{'PASS' if alarm_valid else 'FAIL'}</div>
        <div class="fusion-lbl">{'alarm triggered' if alarm_valid else 'no alarm'}</div>
      </div>
    </div>
    <div style="margin-top:1rem;padding-top:0.75rem;border-top:0.5px solid rgba(0,0,0,0.08);">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#5F5E5A;">
        <span>Confidence scale</span>
        <span>Threshold: {threshold_pct:.0f}% &nbsp;·&nbsp; Fused: {fused_pct:.1f}%</span>
      </div>
      <div class="tbar-wrap">
        <div class="tbar-fill" style="width:{min(fused_pct,100):.1f}%;background:{alarm_color};"></div>
        <div class="tbar-marker" style="left:{threshold_pct:.0f}%;"></div>
      </div>
      <div class="tbar-labels">
        <span>0%</span>
        <span style="margin-left:{threshold_pct-4:.0f}%;">{threshold_pct:.0f}% threshold</span>
        <span>100%</span>
      </div>
    </div>
  </div>

  <div class="connector">↓</div>

  <!-- STAGE 5 — VERDICT -->
  <div class="stage-label">Stage 5 — Final verdict</div>
  <div class="verdict-card" style="background:{alarm_bg};border:0.5px solid {alarm_border};">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-size:18px;font-weight:500;color:{alarm_color};">{alarm_label} &nbsp;·&nbsp; {band} confidence</div>
        <div style="font-size:13px;color:{alarm_color};opacity:0.8;margin-top:3px;">
          {aclass.title()} &nbsp;·&nbsp; {n_alarm} of 3 agents alarming &nbsp;·&nbsp; {n_pairs} correlated pairs broken
        </div>
      </div>
      <div style="text-align:center;background:{alarm_color};border-radius:50%;width:60px;height:60px;
                  display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;">
        <div style="font-size:18px;font-weight:500;color:#fff;">{fused_pct:.0f}%</div>
        <div style="font-size:9px;color:rgba(255,255,255,0.7);">confidence</div>
      </div>
    </div>
    <div class="grid-2" style="margin-top:1rem;">
      <div>
        <div style="font-size:11px;color:{alarm_color};opacity:0.7;margin-bottom:2px;">Onset timestamp</div>
        <div style="font-size:14px;font-weight:500;color:{alarm_color};">{onset}</div>
      </div>
      <div>
        <div style="font-size:11px;color:{alarm_color};opacity:0.7;margin-bottom:2px;">Columns implicated</div>
        <div style="font-size:14px;font-weight:500;color:{alarm_color};">{n_cols} of 44 monitored</div>
      </div>
      <div>
        <div style="font-size:11px;color:{alarm_color};opacity:0.7;margin-bottom:2px;">Primary signals</div>
        <div style="font-size:14px;font-weight:500;color:{alarm_color};">{primary_str}</div>
      </div>
      <div>
        <div style="font-size:11px;color:{alarm_color};opacity:0.7;margin-bottom:2px;">Key finding</div>
        <div style="font-size:14px;font-weight:500;color:{alarm_color};">{key_finding}</div>
      </div>
    </div>
    <div class="action-box" style="background:{alarm_color};">
      <div style="font-size:18px;color:#fff;flex-shrink:0;">&#9888;</div>
      <div>
        <div style="font-size:13px;font-weight:500;color:#fff;">{action_short}</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.75);margin-top:2px;">{action_detail}</div>
      </div>
    </div>
  </div>

  <div class="report-footer">
    <span>Report {report_id} &nbsp;·&nbsp; Generated {gen_str}</span>
    <span>NU CIS 590 &nbsp;·&nbsp; Anomaly detection framework &nbsp;·&nbsp; BATADAL benchmark</span>
  </div>

</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML pipeline report from detect.py JSON output.")
    parser.add_argument("--report", default=None, help="Path to report JSON file")
    parser.add_argument("--latest", action="store_true", help="Use the most recent report in reports/")
    parser.add_argument("--reports-dir", default="reports/", help="Directory containing reports (default: reports/)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: same location as JSON)")
    args = parser.parse_args()

    if not args.report and not args.latest:
        print("Error: specify --report <path> or --latest")
        sys.exit(1)

    if args.latest:
        report_path = find_latest_report(args.reports_dir)
    else:
        report_path = Path(args.report)

    r = load_report(str(report_path))

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = report_path.with_suffix(".html")

    html = build_html(r, str(report_path))
    out_path.write_text(html, encoding="utf-8")
    print(f"[visualize] Report saved -> {out_path}")
    print(f"[visualize] Open in browser: open {out_path}")


if __name__ == "__main__":
    main()
