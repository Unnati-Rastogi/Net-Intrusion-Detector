# Transparent Risk Scoring

**Status:** implemented as an OFFLINE layer (`src/nids/risk.py`,
`reports/risk_assessment.json`). It is **not** wired into the API, the
dashboard, or the prediction path. Alert prioritization and event grouping
are later stages.

## 1. Motivation

The model outputs a probability for every connection, but analysts do not
triage probabilities — they triage queues. Two records flagged at
P(anomaly) = 0.55 are not equally worth looking at: one may be flagged on
unanimous, coherent evidence, the other by a bare majority of contested
signals. The risk layer turns prediction + SHAP evidence into a bounded,
transparent **prioritization score** so that model alerts can be ordered by
how firmly the model's own evidence supports them.

## 2. Inputs

Per record (exactly the outputs of `explain_samples(..., top_k=None)`):

- `anomaly_score` — the production model's P(anomaly) (the same score
  `predict.py` exposes; threshold 0.5).
- per-feature SHAP contributions `φ_f` for **all 41** original features
  (one-hot groups already summed; the two dropped constants are exactly 0).
- `base_value` — the model's mean P(anomaly) over the explanation background.

Identity/lineage fields carried alongside (never used by the risk engine):
`sample_id` is the record's position within the deterministic report sample,
and `source_row_id` is its zero-based row in the source dataframe
(Train_data.csv), propagated verbatim from the explanation records.

SHAP is never recomputed inside `risk.py`; the risk engine is pure
arithmetic on these inputs (~800 µs/sample measured).

## 3. Formula

```
net    = Σ φ_f
mass   = Σ |φ_f|
consensus_ratio  α = net / mass          ∈ [-1, 1]     (mass = 0 → α = 0)

risk = 100 · anomaly_score · (1 + α) / 2                ∈ [0, 100]
```

| Level | Range | Meaning |
|---|---|---|
| LOW | < 20 | not prioritized |
| MEDIUM | 20–49.9 | elevated evidence, sub-threshold or weakly supported |
| HIGH | 50–79.9 | model-flagged with supportive evidence |
| CRITICAL | ≥ 80 | flagged with strong consensus (e.g. s=1.0, α≥0.6) |

## 4. Normalization and boundedness

- `α` is a **ratio**, not a magnitude: it is +1 when every attributed
  feature pushes in the direction of the net effect, 0 when the evidence is
  fully contested, −1 when it unanimously pushes the other way.
- The factor `(1 + α)/2` therefore lives in `[0, 1]` and multiplies the
  score: `risk = 100` requires P(anomaly) = 1 **and** unanimous evidence;
  `risk ≥ 0` always. No empirical clipping is used anywhere.
- Deterministic: fixed thresholds, no randomness, no fitted parameters.

## 5. Threshold methodology

Thresholds are **fixed policy values anchored to the model's operating
point** (0.5, the decision threshold `predict.py` uses), not fitted to data:

- **HIGH = 50**: where a fully-consensual flagged record lands
  (s = 0.5, α = 1 → 50) — "the model flagged this and its evidence agrees".
- **CRITICAL = 80**: flagged with strong consensus (s = 1.0, α ≥ 0.6 → 80).
- **MEDIUM = 20**: elevated but sub-threshold evidence (e.g. s = 0.2 with
  full consensus).

Quantile-based thresholds were **rejected**: the score distribution is
strongly bimodal (75th percentile of the consensus-weighted score ≈ 95 on
the 1,000-row sample), so distribution quantiles would encode the sample's
flag rate, not a review policy.

## 6. Interpretation — and what the score is NOT

Risk scores are **model-derived prioritization scores** evaluated on the
labelled training distribution. They are **not**:

- a probability of an attack or of compromise,
- an estimate of financial or security impact,
- causal risk (no intervention was performed on the network),
- a claim about actual attack severity.

The risk **level** prioritizes model alerts; it says nothing about what the
attack would actually do.

## 7. Why SHAP evidence is used — and why this is not double counting

SHAP values explain the model output, and P(anomaly) **is** the model
output: `base + Σφ = P(anomaly)` exactly (verified per record). Two
consequences:

1. **Adding attribution magnitude to the score double-counts.** An additive
   candidate (`0.7·score + 0.3·normalized mass`) was tested and is
   **rank-identical to the score alone** (Spearman 1.000 on a 1,000-row
   sample): attribution mass is coupled to the net through the additivity
   identity and barely varies (0.50–0.79 here). It was rejected.
2. The information attributions add **beyond** the score is their
   **directional agreement** `α`. It is used as a bounded multiplicative
   factor: it measurably reorders flagged records by evidence coherence
   (Spearman 0.879 vs the score-only order) — a flagged record with
   divided evidence (s = 0.665, α = 0.29 → risk 42.9, MEDIUM) is demoted
   toward review priority, while unanimous evidence keeps full priority
   (s = 1.0, α = 0.85 → 92.3, CRITICAL).

The engine also **re-verifies** the additivity identity per record and
rejects inconsistent contribution lists; split-threshold tie records (see
the SHAP stage) receive exactly one tree-vote of documented slack.

## 8. Example

From `reports/risk_assessment.json` (1,000-row labelled training sample,
seed 42):

| Case | s | α | risk | level |
|---|---|---|---|---|
| Clearly normal (min s) | 0.000 | −0.998 | 0.0 | LOW |
| Clearly anomalous (max s) | 1.000 | +0.847 | 92.3 | CRITICAL |
| Flagged, weakest consensus | 0.665 | +0.291 | 42.9 | MEDIUM |
| Sub-threshold score, evidence against | 0.210 | −0.447 | 5.8 | LOW |
| Borderline (min \|s−0.5\|) | 0.665 | +0.291 | 42.9 | MEDIUM |

## 9. Limitations

- The consensus factor uses **this model's** attributions; a different
  model or background changes α.
- A record whose attributed evidence unanimously contradicts a flagged
  score is only reachable through the split-threshold tie path; it is
  scored as low risk by construction (and flagged with the tie note).
- All evidence is training-distribution evidence; nothing here measures
  performance on novel attacks.
- The level names are alert-prioritization vocabulary, not incident
  severity (impact, scope, asset value are out of scope).
