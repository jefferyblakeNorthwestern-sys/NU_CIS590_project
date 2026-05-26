# Pattern Matching Methodology

Patterns are stored and matched in **z-score space** — not raw value space.
This makes the pattern library transferable across datasets with different scales,
units, or schemas, as long as the deviation structure is similar.

---

## What a Pattern Is

A deviation signature: which columns deviated, in which direction, by how much
relative to their own baseline during a known event window.

```json
{
  "pattern_id": 0,
  "label": "coordinated_sensor_falsification",
  "signature": {
    "col_A": {"mean_z": 4.2,  "direction": "high", "magnitude": "strong"},
    "col_B": {"mean_z": -3.1, "direction": "low",  "magnitude": "strong"},
    "col_C": {"mean_z": 1.4,  "direction": "high", "magnitude": "moderate"}
  },
  "window_length": 24,
  "columns_count": 3
}
```

No raw values stored — only each column's relationship to its own baseline.

---

## Matching Algorithm

### Step 1 — Build current deviation signature
From AnomalySignals, compute per-column z-score, direction, and magnitude.

### Step 2 — Score against each pattern
- Exact direction + magnitude: full weight
- Direction match only: 0.5 weight
- No direction match: 0 weight
- Score = weighted_matched / total_pattern_columns

### Step 3 — Rank and threshold
- Retain matches with similarity > 0.40
- Report top-3
- Strong match: similarity > 0.65

---

## Cross-Dataset Pattern Transfer

Because patterns are in z-score space:

| Transfer                              | Works?  | Condition                                 |
|---------------------------------------|---------|-------------------------------------------|
| Same dataset, different time period   | Yes     | Column names must match                   |
| Different dataset, same physical system | Yes   | Use column registry type flags to remap   |
| Different dataset, different system   | Partial | Manual review required for remap < 0.75  |

### Column Remapping
Match columns by registry profile similarity (dtype, binary flag, variance class).
remap_confidence < 0.75 requires manual review before use.

---

## Pattern Library Maintenance

- **Add:** After analyst confirms new event type, extract signature and append
- **Retire:** Patterns with no matches in > 6 months — flag for review
- **Merge:** Patterns with similarity > 0.90 — consider combining
- **Never modify signatures retroactively** — preserves historical record

---

## Unlabeled Cluster Extraction (no labels available)

1. Run deterministic layer over training split
2. Collect windows where 3 or more columns fire simultaneously
3. Build deviation signatures for each multi-column window
4. Cluster via k-means in z-score space (k=5 default)
5. Store centroids as `unlabeled_cluster_{k}` patterns

At detection time, matches to unlabeled clusters appear in the report with lower
confidence and a recommendation for analyst review.
