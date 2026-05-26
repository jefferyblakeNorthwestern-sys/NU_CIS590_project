# NU CIS 590 — Anomaly Detection Skill

Schema-agnostic multi-agent framework for detecting anomalies in time-series
operational data and classifying them as valid cyber events or benign deviations.

## Structure

```
anomaly-detection/
├── SKILL.md                      ← Framework entry point and architecture overview
├── references/
│   ├── training.md               ← Training methodology and execution cycles
│   ├── agents.md                 ← Statistical, Behavioral, Correlation agents
│   ├── pattern_matching.md       ← Schema-agnostic pattern similarity matching
│   └── orchestration.md          ← Confidence fusion and incident report format
└── scripts/
    ├── ingest.py                 ← Schema-agnostic CSV ingestion
    ├── train.py                  ← Orchestrates all four training cycles
    └── detect.py                 ← Runs detection pipeline, produces incident report
```

## Quickstart

### 1. Train on historical data
```bash
python scripts/train.py \
  --input path/to/historical.csv \
  --output trained_model/ \
  --target-fpr 0.02 \
  --exclude-windows windows.json   # optional: labeled anomaly windows
```

### 2. Detect on operational data
```bash
python scripts/detect.py \
  --input path/to/operational.csv \
  --model trained_model/ \
  --output reports/
```

### 3. Optional: inspect a new dataset schema
```bash
python scripts/ingest.py --input path/to/data.csv
```

## Dependencies
```
pip install pandas numpy
```

## Output

Every detection run produces a **Cybersecurity Incident Report** in both `.txt`
and `.json` formats covering:
- Alarm status, validity, and confidence
- Incident narrative and onset timestamp
- Technical indicators with deviation magnitudes
- Per-agent verdicts (Statistical, Behavioral, Correlation)
- Analyst guidance and recommended action

## Dataset Compatibility

The framework makes no assumptions about column names, units, or domain.
Train it on any time-series CSV and it discovers structure automatically.

