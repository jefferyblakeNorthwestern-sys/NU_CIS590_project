---
name: anomaly-detection
description: >
  Schema-agnostic multi-agent framework for detecting anomalies in time-series operational data
  and classifying them as valid cyber events or benign deviations. Use this skill whenever the
  user is working with sensor data, ICS/SCADA monitoring, operational technology (OT) security,
  time-series anomaly detection, industrial control system anomaly analysis, or any dataset where
  the goal is to distinguish cyberattacks or intrusions from normal operational variance.
  Also trigger for tasks involving training anomaly detectors on historical data, reading
  operational CSV data for security analysis, tuning detection thresholds, validating alarms,
  or generating cybersecurity incident reports from sensor telemetry.
---

# Schema-Agnostic Anomaly Detection Framework

## Design Philosophy

This framework treats the dataset as a black box until training time. It makes no assumptions
about column names, signal types, units, or domain physics. The training phase discovers
structure from the data itself. The detection phase applies that learned structure to flag
deviations and route them to reasoning agents that decide alarm validity.

The two-question architecture drives every decision:
1. **Is this anomalous?** — answered by the deterministic layer
2. **Is this a valid cyber event?** — answered by the reasoning agents

---

## Framework Architecture

```
CSV (training or operational)
        |
        v
+-----------------------------+
|   CSV Ingestion Layer       |  Schema discovery · type inference · normalization
|   (schema-agnostic)         |  -> column registry · baseline statistics
+-----------------------------+
        |  (training path branches here — see references/training.md)
        v
+-----------------------------+
|   Deterministic Layer       |  Z-score · Rate-of-change · Mutual deviation
|   (coarse-grain filters)    |  -> AnomalySignal structs routed to agents
+-----------------------------+
        |
        v
+---------------------------------------------------------+
|               Reasoning Agents (parallel)               |
|   Statistical · Behavioral · Correlation                |
|   Each returns: confidence · validity · evidence        |
+---------------------------------------------------------+
        |
        v
+-----------------------------+
|   Orchestration Agent       |  Fuses verdicts · classifies event type
|   (LLM-driven)              |  -> Cybersecurity Incident Report
+-----------------------------+
```

---

## Execution Modes

| Mode       | When                                       | What happens                                            |
|------------|--------------------------------------------|---------------------------------------------------------|
| `train`    | First run on a new dataset, or retraining  | Reads historical CSV, builds baselines, fits thresholds |
| `detect`   | Operational run after training             | Reads operational CSV, runs deterministic + agents      |
| `validate` | After training, before production          | Runs detect on labeled holdout, scores alarm validity   |

---

## Step 1 — CSV Ingestion

Read the CSV without assumptions. The framework discovers structure.
See `scripts/ingest.py` for full implementation.

The `registry` is the framework's only persistent schema artifact, saved to
`trained_model/column_registry.json`. Nothing else references column names directly.

---

## Step 2 — Training

Builds the statistical model that defines "normal." Full methodology:
-> **`references/training.md`**

Training artifacts saved to `trained_model/`:

| Artifact              | Description                                          |
|-----------------------|------------------------------------------------------|
| column_registry.json  | Schema map with type flags                           |
| baselines.json        | Per-column mean, sigma, percentiles, autocorrelation |
| rate_baselines.json   | Per-column expected delta-value/delta-t distribution |
| correlation_map.json  | Pairwise column correlations from normal-period data |
| thresholds.json       | Tuned detection thresholds                           |
| pattern_library.json  | Extracted multi-column co-occurrence patterns        |

---

## Step 3 — Deterministic Layer

Three coarse-grain statistical filters. High recall, not precision — they feed agents, not alarms.

| Filter           | Detects                                     |
|------------------|---------------------------------------------|
| Z-score          | Magnitude outliers vs trained baseline      |
| Rate-of-change   | Sudden spikes/drops within normal range     |
| Mutual deviation | Correlation breakdowns between paired cols  |

---

## Step 4 — Reasoning Agents

Three agents run in parallel. See -> **`references/agents.md`**

| Agent       | Question                                                     |
|-------------|--------------------------------------------------------------|
| Statistical | Are deviations consistent with a deliberate attack vs noise? |
| Behavioral  | Do deviations match known event patterns from training?      |
| Correlation | Has the relational structure between columns broken down?    |

---

## Step 5 — Orchestration

Fuses agent verdicts, runs LLM reasoning, produces Cybersecurity Incident Report.
See -> **`references/orchestration.md`**

---

## Quick Reference

| Task                          | Read                             |
|-------------------------------|----------------------------------|
| Training a new dataset        | `references/training.md`         |
| Agent implementation          | `references/agents.md`           |
| Orchestration + report format | `references/orchestration.md`    |
| Pattern matching methodology  | `references/pattern_matching.md` |
| CSV ingestion                 | `scripts/ingest.py`              |
