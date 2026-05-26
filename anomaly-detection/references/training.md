# Training Methodology

Training builds the statistical model that defines "normal" for a specific dataset.
It produces model artifacts that detection mode loads at runtime.
Training makes no assumptions about dataset schema — it discovers structure from data.

---

## Prerequisites

- Historical CSV representing normal operation (or labeled dataset where normal rows can be isolated)
- Consistent sampling interval (or one that can be inferred)
- Minimum ~500 timesteps of normal operation

---

## Training Data Split

Split chronologically — never shuffle time-series data:

```
|------------ Historical CSV ------------|
|  Train (60%)  |  Val (20%)  |  Test (20%)  |
```

- **Train:** Fit baselines, correlations, thresholds
- **Val:** Tune threshold cutoffs (score false positive rate)
- **Test:** Held out until final validation only

---

## Training Cycle 1 — Baseline Fitting

**Input:** Train split
**Output:** `trained_model/baselines.json`, `trained_model/rate_baselines.json`

For every numeric column:
- mean, std, p01, p99, median, autocorr_lag1
- Rate-of-change distribution: mean_rate, std_rate, p99_rate, p999_rate

For binary/categorical columns:
- Value frequency distribution

---

## Training Cycle 2 — Correlation Mapping

**Input:** Train split
**Output:** `trained_model/correlation_map.json`

Pairwise Pearson correlation across all numeric columns.
Retain only pairs with |correlation| > 0.50.

---

## Training Cycle 3 — Threshold Tuning

**Input:** Val split
**Output:** `trained_model/thresholds.json`

Run deterministic layer over validation split with default thresholds.
Binary search on z_score_cutoff and rate_cutoff until FPR <= target (default 2%).

Default starting values:
```
z_score_cutoff:           3.0
rate_cutoff_multiplier:   1.0   (x p99_rate per column)
mutual_deviation_cutoff:  2.0   (sigma multiplier on correlation residuals)
alarm_threshold:          0.60
```

---

## Training Cycle 4 — Pattern Library Extraction

**Input:** Labeled anomaly windows (optional)
**Output:** `trained_model/pattern_library.json`

Patterns stored as z-score deviation signatures — not raw values.
Schema: which columns deviated, direction, magnitude relative to baseline.
This makes patterns transferable across datasets with different scales.

If no labels: skip or run unlabeled cluster extraction (k-means, k=5 default).

---

## Executing a Full Training Run

```bash
python anomaly-detection/scripts/train.py \
  --input data/historical.csv \
  --output trained_model/ \
  --target-fpr 0.02 \
  --exclude-windows data/windows.json
```

`windows.json` format:
```json
[
  {"start": "2017-01-16 09:00", "end": "2017-01-19 06:00", "label": "attack_type_A"},
  {"start": "2017-02-05 12:00", "end": "2017-02-05 18:00", "label": "pump_fault"}
]
```

---

## Training Quality Checks

1. `achieved_val_fpr` <= target in thresholds.json
2. >= 80% of numeric columns have std > 0
3. Spot-check 3-5 correlation pairs for plausibility
4. Pattern count matches expected labeled window count
