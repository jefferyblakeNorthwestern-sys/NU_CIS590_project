# Orchestration and Incident Report

The orchestrator fuses three agent verdicts into a final confidence score,
runs the LLM reasoning step, and produces the Cybersecurity Incident Report.

---

## Confidence Fusion

```python
base_confidence = (stat.confidence + behav.confidence + corr.confidence) / 3

agents_alarming = count(agent.alarm_valid == true)
agreement_multiplier = {0: 0.60, 1: 0.85, 2: 1.10, 3: 1.25}[agents_alarming]

fused = min(base_confidence x agreement_multiplier, 1.0)
alarm_valid = fused >= thresholds.alarm_threshold   # default 0.60
```

**Anomaly class:** majority vote across three agents; ties go to more conservative class.

---

## Validity Classification

| Condition                                             | Validity  |
|-------------------------------------------------------|-----------|
| fused >= threshold AND agents_alarming >= 2           | VALID     |
| fused >= threshold AND agents_alarming == 1           | UNCERTAIN |
| fused < threshold AND any agent alarm_valid: true     | UNCERTAIN |
| All agents alarm_valid: false OR fused < 0.35         | INVALID   |

UNCERTAIN alarms always route to analyst review — never auto-dismissed.

---

## LLM Reasoning Step

Five structured questions the LLM must answer before producing the report:

1. What is the most likely explanation for this anomaly pattern?
2. Is this consistent with a deliberate cyber event or an operational/environmental cause?
3. Which columns are central vs incidental?
4. What is the earliest timestamp showing something was wrong?
5. What is your confidence, and what single piece of information would most change it?

---

## Cybersecurity Incident Report Structure

```
SECTION 1 — VERDICT
  ALARM STATUS / VALIDITY / ANOMALY CLASS / CONFIDENCE + BAND

  Confidence bands:
    0.00-0.35  LOW       — insufficient evidence; analyst review recommended
    0.35-0.60  MEDIUM    — plausible anomaly; investigate before escalating
    0.60-0.80  HIGH      — strong evidence; escalate per IR procedure
    0.80-1.00  VERY HIGH — near-certain; immediate response warranted

SECTION 2 — INCIDENT SUMMARY
  EVENT TYPE / PATTERN MATCH / ONSET TIMESTAMP / SCOPE / NARRATIVE

SECTION 3 — TECHNICAL INDICATORS
  PRIMARY INDICATORS (central to event)
  SECONDARY INDICATORS (incidental/downstream)
  CORRELATED PAIR BREAKDOWNS

SECTION 4 — AGENT VERDICTS
  Per-agent: confidence, alarm_valid, class, finding, uncertainty
  Behavioral: top-3 pattern matches with similarity scores
  FUSION SUMMARY: base confidence, multiplier, fused, threshold

SECTION 5 — ANALYST GUIDANCE
  RECOMMENDED ACTION (ESCALATE / INVESTIGATE / MONITOR / DISMISS)
  KEY UNCERTAINTIES
  PRECURSOR INDICATORS
  ADDITIONAL CONTEXT REQUESTED

SECTION 6 — METADATA
  Model artifacts, thresholds applied, signals evaluated
```

---

## No-Alarm Reports

If the deterministic layer fires zero AnomalySignals, produce a condensed report:

```
ALARM STATUS:  NO ALARM
VALIDITY:      INVALID
CONFIDENCE:    0.00
NARRATIVE:     No statistically significant deviations detected...
```

Every detection cycle produces a report — making detection gaps visible.
