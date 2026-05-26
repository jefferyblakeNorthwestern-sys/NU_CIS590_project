# Reasoning Agents

Three agents run in parallel after the deterministic layer. Each receives the same
inputs and answers a distinct question about the anomaly.

---

## Shared Inputs

```
anomaly_signals:    list of AnomalySignal structs
column_registry:    trained_model/column_registry.json
baselines:          trained_model/baselines.json
rate_baselines:     trained_model/rate_baselines.json
correlation_map:    trained_model/correlation_map.json
pattern_library:    trained_model/pattern_library.json
thresholds:         trained_model/thresholds.json
```

---

## Shared Output Schema

```json
{
  "agent":               "statistical | behavioral | correlation",
  "confidence":          0.0,
  "alarm_valid":         true,
  "anomaly_class":       "cyber_event | operational_fault | sensor_noise | nominal",
  "columns_implicated":  [],
  "evidence":            "<structured reasoning citing specific values>",
  "uncertainty":         "<what evidence would most change this verdict>"
}
```

---

## Statistical Agent

**Question:** Are deviations statistically consistent with a deliberate, sustained
anomaly — or with noise, transient spikes, or sensor drift?

**Key checks:**
1. **Severity distribution** — count HIGH/MED/LOW signals
2. **Persistence** — fraction of timesteps flagged in window
3. **Spread** — how many distinct columns are flagged
4. **Baseline distance** — normalized deviation = |z| / z_cutoff

**Confidence:**
```
base = severity_score x 0.3 + persistence x 0.4 + spread_score x 0.3
confidence = base x mean_normalized_deviation
```

**Anomaly class:**
- `sensor_noise`: low severity, low persistence, single column
- `operational_fault`: multi-column, sustained, single causal source
- `cyber_event`: multi-column, sustained, inconsistent with single physical cause
- `nominal`: insufficient signal

---

## Behavioral Agent

**Question:** Does this pattern match known event signatures from the pattern library?

**With pattern library:** Score each pattern using z-score direction + magnitude matching.
Report top-3 matches. Confidence = top similarity x 0.9.

**Without pattern library:** Cluster AnomalySignals by co-occurrence.
Tight multi-column clusters -> operational_fault. Scattered -> sensor_noise.

**Pattern similarity scoring:**
- Exact direction + magnitude match: full weight
- Direction match only: 0.5 weight
- No direction match: 0 weight
- Score = weighted_matched / total_pattern_columns
- Strong match: similarity > 0.65

---

## Correlation Agent

**Question:** Has the relational structure between columns broken down in a way
consistent with deliberate manipulation?

**Key checks:**
1. **Correlated pair status** — which trained pairs involve flagged columns
2. **Decoupling score** — pairs_broken / pairs_expected_to_hold
3. **Directionality** — both deviate vs one deviates
4. **Orphaned columns** — flagged with no correlated partners also flagging

**Confidence:**
```
confidence = decoupling_score x 0.7 + (1 - orphan_ratio) x 0.3
```

High orphan ratio -> noise. Low orphan ratio -> signals cluster -> event.

**Anomaly class:**
- `cyber_event`: high decoupling + directional inconsistencies
- `operational_fault`: decoupled but direction consistent with single propagating cause
- `sensor_noise`: low decoupling, high orphan ratio
- `nominal`: no meaningful decoupling
