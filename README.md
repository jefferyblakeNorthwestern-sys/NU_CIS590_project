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

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train on historical data
```bash
python anomaly-detection/scripts/train.py \
  --input data/historical.csv \
  --output trained_model/ \
  --target-fpr 0.02 \
  --exclude-windows data/windows.json   # optional: labeled anomaly windows
```

### 3. Detect on operational data
```bash
python anomaly-detection/scripts/detect.py \
  --input data/operational.csv \
  --model trained_model/ \
  --output reports/
```

### 4. Inspect a new dataset schema
```bash
python anomaly-detection/scripts/ingest.py --input data/your_data.csv
```

## Output

Every detection run produces a **Cybersecurity Incident Report** in `.txt` and `.json`:
- Alarm status, validity, and confidence band
- Incident narrative and onset timestamp
- Technical indicators with deviation magnitudes
- Per-agent verdicts (Statistical, Behavioral, Correlation)
- Analyst guidance and recommended action

## Dataset Compatibility

The framework makes no assumptions about column names, units, or domain.
Train it on any time-series CSV — structure is discovered automatically.
