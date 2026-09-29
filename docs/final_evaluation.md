# Final Evaluation Report

**Status:** final technical evidence package for the frozen implementation.
Every number below comes from the committed `reports/` artifacts; each source
report was regenerated through its official pipeline on 2026-09-29 and
verified identical modulo timestamps/provenance metadata (see
[Reproducibility](#8-reproducibility)). No model, preprocessing, SHAP, risk,
or prioritization behavior was changed for this report.

**What this report is not:** it is not a winner declaration, a production
performance estimate, or a claim about real-world attack behaviour. The
measured evidence is presented as measured.

---

## 1. Evaluation Methodology

### Dataset

| File | Role | Shape | Notes |
|---|---|---|---|
| `data/Train_data.csv` | labelled training data | 25,192 × 42 | `class` ∈ {normal (13,449), anomaly (11,743)}; SHA-256 `1e40eb66…e43b07a` |
| `data/Test_data.csv` | unlabelled partition | 22,544 × 41 | **no ground truth** — never evaluated, merged, or tuned on; SHA-256 `a58db8c6…19caf8` |

### Cross-validation setup (all headline metrics)

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, **out-of-fold**
  predictions over the 25,192 labelled rows.
- The full pipeline (one-hot preprocessing included) is **refit inside every
  fold** on that fold's training rows only — no leakage.
- Per-fold metrics → mean ± population standard deviation (`ddof=0`).
- Metrics: accuracy, precision/recall/F1 for the anomaly class (positive
  label 1), precision/recall for the normal class, macro F1, one out-of-fold
  confusion matrix per candidate.
- Scope: the **labelled training distribution only**. These numbers are
  cross-validation numbers, not held-out-test or production estimates.

## 2. Model Comparison

All three candidates share one preprocessing implementation
(`build_pipeline()`), identical folds, and one shared CV routine; each is
scored once with its documented, seeded configuration (no per-candidate
tuning). Source: `reports/model_comparison.json` / `.csv`.

| Model | Accuracy | Precision (anomaly) | Recall (anomaly) | F1 (anomaly) | Precision (normal) | Recall (normal) | Macro F1 |
|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9743 ± 0.0025 | 0.9769 ± 0.0007 | 0.9678 ± 0.0053 | 0.9723 ± 0.0028 | 0.9721 ± 0.0045 | 0.9800 ± 0.0006 | 0.9742 ± 0.0026 |
| Random Forest (production) | 0.9975 ± 0.0013 | 0.9988 ± 0.0004 | 0.9957 ± 0.0025 | 0.9973 ± 0.0014 | 0.9963 ± 0.0022 | 0.9990 ± 0.0004 | 0.9974 ± 0.0013 |
| XGBoost (candidate) | 0.9979 ± 0.0010 | 0.9986 ± 0.0006 | 0.9968 ± 0.0024 | 0.9977 ± 0.0011 | 0.9973 ± 0.0020 | 0.9988 ± 0.0005 | 0.9979 ± 0.0010 |

Out-of-fold confusion matrices — **rows = actual, columns = predicted**,
class order `[normal (0), anomaly (1)]`:

| Model | TN (normal→normal) | FP (normal→anomaly) | FN (anomaly→normal) | TP (anomaly→anomaly) |
|---|---|---|---|---|
| Logistic Regression | 13,180 | 269 | 378 | 11,365 |
| Random Forest | 13,435 | 14 | 50 | 11,693 |
| XGBoost | 13,433 | 16 | 37 | 11,706 |

Figure: `reports/figures/model_comparison.png`, `reports/figures/confusion_matrices.png`.

### Reading the differences (no winner declared)

The model-selection artifact records `decision: not_declared`
(`docs/model_selection_framework.md`); so does this report. The measured
differences between the two tree candidates are small relative to fold
variability:

- Accuracy: XGB − RF = **+0.0004**; the per-fold std of each is ≈ 0.0010–0.0013,
  so the gap is **within one fold standard deviation**.
- Anomaly recall: XGB − RF = **+0.0011** against a fold std of ≈ 0.0024–0.0025 —
  likewise **within one fold standard deviation**.
- In confusion-matrix terms, XGB misses 37 anomaly records out-of-fold where RF
  misses 50 (13 fewer of 11,743 anomaly rows), with FP counts of 16 vs 14.
  Both candidates sit near the ceiling of this dataset; the aggregate accuracy
  difference alone would call them equivalent, while the error cells differ —
  which is exactly why the selection framework requires looking inside the
  confusion matrix rather than ranking on one number.

**No universal superiority is claimed for any candidate.** The production
configuration remains Random Forest as the existing system state (Section 3).

## 3. Production Model

- `RandomForestClassifier(n_estimators=200, class_weight='balanced',
  random_state=42)` inside the shared sklearn `Pipeline`
  (const-column drop → one-hot → numeric passthrough → classifier).
- Approved configuration from 5-fold stratified CV on `Train_data.csv` only;
  the saved full-data artifact is `models/rf_ids_pipeline.joblib`
  (SHA-256 `7c6097c8cdd3d8946818ae88b61a062fc9cb9e67950efc3699538308ee8a3c13`).
- It remains production **because that is the existing system configuration**:
  the documented selection procedure
  (`docs/model_selection_framework.md`) — pre-registered criteria, security-
  relevant trade-offs, determinism re-certification — has not been executed,
  and this report does not execute it. The XGBoost candidate is an evaluation
  artifact only.

## 4. Explainability (SHAP)

### Methodology (unchanged)

`shap.TreeExplainer` on the **saved production pipeline**,
`feature_perturbation="interventional"`, deterministic **100-row training
background (seed 42, no replacement)**. Attributions are computed on the 116
transformed columns and rolled back to the **original 41 feature names** by
summing each one-hot group (dropped constants contribute exactly 0). The
explained output is the model's P(anomaly) (class index 1). Additivity —
`base + Σφ = P(anomaly)` — is verified per record against a 1e-6 gate; the
committed sample explanations show errors ≤ 1.7e-9. Split-threshold tie
records receive exactly one tree-vote (1/200) of documented slack.

### Global importance (1,000-row labelled sample, seed 42)

Global importance = mean |SHAP| per original feature
(`reports/explanations.json`, `global_importance`).

| Rank | Feature | mean \|SHAP\| | share of total |
|---|---|---|---|
| 1 | src_bytes | 0.09535 | 17.3% |
| 2 | dst_bytes | 0.05459 | 9.9% |
| 3 | flag | 0.04876 | 8.8% |
| 4 | service | 0.03679 | 6.7% |
| 5 | logged_in | 0.03391 | 6.1% |
| 6 | dst_host_srv_serror_rate | 0.02669 | 4.8% |
| 7 | dst_host_srv_count | 0.02445 | 4.4% |
| 8 | dst_host_serror_rate | 0.02332 | 4.2% |
| 9 | serror_rate | 0.02030 | 3.7% |
| 10 | count | 0.01955 | 3.5% |
| 11 | protocol_type | 0.01883 | 3.4% |
| 12 | dst_host_same_srv_rate | 0.01879 | 3.4% |
| 13 | dst_host_diff_srv_rate | 0.01733 | 3.1% |
| 14 | srv_serror_rate | 0.01718 | 3.1% |
| 15 | diff_srv_rate | 0.01576 | 2.9% |

Figure: `reports/figures/shap_global_importance.png`.

### Representative local explanations

Each example is one record of the committed evidence; SHAP contributions are
verbatim from the committed artifacts, with the model's own direction labels.

**Clear normal** — Train row 106 (`sample_explanations[0]`): prediction
`normal`, P(anomaly) = 0.000 (base 0.517).

| Feature | Value | SHAP contribution | Direction |
|---|---|---|---|
| src_bytes | 219 | −0.091469 | decreases_anomaly_score |
| dst_bytes | 1,212 | −0.059961 | decreases_anomaly_score |
| flag | SF | −0.048308 | decreases_anomaly_score |
| logged_in | 1 | −0.044944 | decreases_anomaly_score |
| service | http | −0.033685 | decreases_anomaly_score |

**Borderline / less-consistent anomaly** — risk sample_id 406 (Train row
10,738): prediction `anomaly`, P(anomaly) = 0.675, consensus α = +0.299,
risk 43.85 → MEDIUM. Evidence is divided (four features push up, `flag=SF`
pushes down), which lowers the record's review priority.

| Feature | Value | SHAP contribution | Direction |
|---|---|---|---|
| hot | 2 | +0.109409 | increases_anomaly_score |
| root_shell | 1 | +0.080292 | increases_anomaly_score |
| srv_rerror_rate | 0.5 | +0.048213 | increases_anomaly_score |
| num_access_files | 1 | +0.044481 | increases_anomaly_score |
| flag | SF | −0.033493 | decreases_anomaly_score |

**Sub-threshold score, evidence against** — risk sample_id 491 (Train row
2,720): P(anomaly) = 0.210 (below the 0.5 decision threshold), consensus
α = −0.446, risk 5.81 → LOW.

**Strong, unanimous anomaly** — risk sample_id 509 (Train row 12,826), rank 1
of the committed alert report: prediction `anomaly`, P(anomaly) = 1.000,
consensus α = +0.943, risk 97.16 → CRITICAL → P1.

| Feature | Value | SHAP contribution | Direction |
|---|---|---|---|
| src_bytes | 0 | +0.095242 | increases_anomaly_score |
| dst_bytes | 0 | +0.045844 | increases_anomaly_score |
| same_srv_rate | 0.04 | +0.037703 | increases_anomaly_score |
| logged_in | 0 | +0.035877 | increases_anomaly_score |
| rerror_rate | 0.22 | +0.032997 | increases_anomaly_score |

### What SHAP evidence is — and is not

SHAP values describe **how much each feature's value moves this model's
P(anomaly) output for this record** away from the background baseline. They
do **not** prove causation, malicious intent, attack severity, or real-world
impact; they depend on this model and this background sample; and all
attributions come from the labelled training distribution
(`Test_data.csv` is never scored or explained).

## 5. Risk Assessment

### Formula (unchanged)

For each record, with `s` = the model's P(anomaly) and φ = the per-feature
SHAP contributions of that record's own explanation:

```
net  = Σ φ_f            mass = Σ |φ_f|
α (consensus_ratio) = net / mass ∈ [-1, 1]        (mass = 0 → α = 0)

risk = 100 · s · (1 + α) / 2                      ∈ [0, 100]
```

α is a **ratio, not a magnitude**: +1 = every attributed feature pushes in
the direction of the net effect, 0 = fully contested, −1 = unanimously
against. Fixed thresholds, anchored to the model's 0.5 operating point
(quantile thresholds were explicitly rejected):

| Level | Range | Meaning |
|---|---|---|
| LOW | < 20 | not prioritized |
| MEDIUM | 20–49.9 | elevated evidence, sub-threshold or weakly supported |
| HIGH | 50–79.9 | model-flagged with supportive evidence |
| CRITICAL | ≥ 80 | flagged with strong consensus |

### Why raw SHAP magnitude is not added to the score

Because additivity makes it redundant: `base + Σφ = P(anomaly)` exactly, so
an additive candidate (`0.7·score + 0.3·normalized mass`) is **rank-identical
to the score alone** (Spearman 1.000 on the 1,000-row sample) — adding
attribution mass double-counts the model score. The information attributions
add *beyond* the score is their **directional agreement α**, used as a
bounded multiplicative factor (it reorders flagged records, Spearman 0.879
vs the score-only order). The engine re-verifies additivity per record and
rejects inconsistent contribution lists.

### Distribution (1,000-row labelled sample, seed 42)

| Level | Count | | Priority | Count |
|---|---|---|---|---|
| LOW | 521 | | P4 | 521 |
| MEDIUM | 2 | | P3 | 2 |
| HIGH | 7 | | P2 | 7 |
| CRITICAL | 470 | | P1 | 470 |

Score distribution: min 0.0, max 97.16, mean 44.21, median 0.08; strongly
bimodal — histogram (bin width 10): `0–10: 521, 40–50: 2, 50–60: 2, 60–70: 2,
70–80: 3, 80–90: 105, 90–100: 365`. Figure:
`reports/figures/risk_distribution.png`.

Worked examples (all verified against the committed report):

| Case | s | α | risk | level |
|---|---|---|---|---|
| Clearly normal (min s, sid 0) | 0.000 | −0.998 | 0.0 | LOW |
| Clearly anomalous (max s, sid 2) | 1.000 | +0.847 | 92.3 | CRITICAL |
| Flagged, weakest consensus (sid 838) | 0.665 | +0.291 | 42.9 | MEDIUM |
| Sub-threshold, evidence against (sid 491) | 0.210 | −0.446 | 5.8 | LOW |
| Borderline (min \|s−0.5\|, sid 838) | 0.665 | +0.291 | 42.9 | MEDIUM |

**Interpretation:** the risk score is a **model-derived prioritization
score** evaluated on the labelled training distribution. It is **not** a
probability of attack or compromise, an incident-severity or business-impact
estimate, or causal risk. The level names are alert-prioritization
vocabulary, not attack severity.

## 6. Alert Prioritization

### Deterministic ordering (unchanged)

1. `risk_score` descending
2. `anomaly_score` descending
3. `risk_level` by explicit priority rank (CRITICAL > HIGH > MEDIUM > LOW)
4. `sample_id` ascending

Mapping: **CRITICAL → P1, HIGH → P2, MEDIUM → P3, LOW → P4 (review
backlog)**. P1 means **"highest analyst review priority"** — never attack
severity, danger level, or financial impact. No new score is computed; the
layer only orders the existing risk records, and true labels are dropped
from alert records so they structurally cannot influence the ordering.
`sample_id` is the record's position in the 1,000-row report sample (0–999);
`source_row_id` is its zero-based row in `Train_data.csv` (lineage metadata
only, never used for ordering).

### Distribution and verification

Counts above (P1 470 / P2 7 / P3 2 / P4 521). Figure:
`reports/figures/alert_priority_distribution.png`. Verification performed in
this stage: re-sorting all 1,000 risk records with the documented rules
reproduces the committed alert order **exactly (1,000/1000 positions)**, and
the priority counts equal the risk-level counts under the mapping.

### Representative alerts

| Rank | sample_id | Train row | P(anomaly) | α | risk | level → priority |
|---|---|---|---|---|---|---|
| 1 | 509 | 12,826 | 1.000 | +0.943 | 97.16 | CRITICAL → P1 |
| 478/479 | 406 | 10,738 | 0.675 | +0.299 | 43.85 | MEDIUM → P3 |
| 480+ (backlog) | 0 | 106 | 0.000 | −0.998 | 0.00 | LOW → P4 |

Rank-1 alert explanation: the model is fully flagged (P(anomaly) = 1.0) and
its evidence is nearly unanimous (α = +0.943): zero bytes in both directions,
almost no shared-service activity, no successful login, elevated retry rates
— every top feature pushes the anomaly score up.

## 7. End-to-End Consistency

Verified on 2026-09-29 against the live pipeline (saved artifact + domain
code) and the committed reports; **21/21 checks passed, no discrepancies
found** (none patched — there was nothing to patch):

| Check | Records | Result |
|---|---|---|
| prediction matches model output (recomputed) | clear-normal row 106, sid 406, sid 509 | PASS |
| SHAP → risk re-derivation matches committed anomaly score (|Δ| < 1e-12) | sid 406, 509 | PASS |
| risk score re-derivation matches committed report (|Δ| < 1e-9) | sid 406, 509 | PASS |
| risk level matches score under the fixed thresholds | sid 406, 509 | PASS |
| priority equals `PRIORITY_MAP[level]` | sid 406 (P3), 509 (P1) | PASS |
| lineage: `sample_id`/`source_row_id` match committed report; stored evidence values equal the actual source row in `Train_data.csv` | sid 406, 509 | PASS |
| committed alert order == deterministic re-sort of the risk records | all 1,000 | PASS |
| `sample_id` = position 0–999; id sets consistent across risk/alert reports | all 1,000 | PASS |
| priority counts == level counts under the mapping | all 1,000 | PASS |
| ad-hoc vector with no report match carries no fabricated ids (also enforced by the test suite) | API layer | PASS |

Chain verified end to end: **prediction → SHAP → risk → priority → alert
order**, for a normal record, a borderline record, and a top-priority alert.

## 8. Reproducibility

### Environment

Python 3.13.1; fully pinned `requirements.txt` (scikit-learn 1.8.0,
pandas 2.3.3, numpy 2.2.2, xgboost 3.4.1, shap 0.52.0, joblib 1.5.3,
matplotlib 3.11.2, fastapi 0.141.1). Clean-venv install from
`requirements.txt` verified in Stage 3.

### Determinism verification (this stage)

Each official pipeline was rerun and compared against the committed report:

| Command | Result |
|---|---|
| `python -m src.nids.evaluate` | metrics/folds/confusion matrix **identical**; only `timestamp_utc` + `provenance` differ; figure regenerated identically |
| `python -m src.nids.compare` | JSON **identical** modulo timestamp/provenance; CSV **byte-identical** |
| `python -m src.nids.explain` | all SHAP values, global importance, and sample explanations **identical**; only `git_revision` and runtime timings differ |
| `python -m src.nids.risk` | all risk scores, levels, counts, distribution **identical**; only `git_revision` differs |
| `python -m src.nids.prioritize` | full ordering and counts **identical**; only `git_revision` + upstream timestamp differ |

No regenerated report was committed: the committed artifacts are the
authoritative evidence, and regeneration was used purely as verification.

### Deterministic seeds and guards

- Data seeds: CV `random_state=42`; SHAP background seed 42; report samples
  seed 42 (1,000 rows, no replacement, sorted).
- Full-run bit-identity is additionally enforced by the opt-in slow tests
  (`RUN_SLOW_EVAL_TESTS=1`): two complete comparison runs produce identical
  fold metrics; the full CV reproduces the README metrics.
- Test suite: 241 tests (ML core, evaluation, comparison, explainability,
  risk, prioritization, API, dashboard, hardening) — all passing; CI
  (`.github/workflows/ci.yml`) runs the portable subset on Python 3.13 for
  every push/PR.
- Model artifact: `models/rf_ids_pipeline.joblib`, SHA-256
  `7c6097c8cdd3d8946818ae88b61a062fc9cb9e67950efc3699538308ee8a3c13`
  (re-verified after every pipeline rerun in this stage).

## 9. Performance

Measured in Stage 3 (30 repetitions, dev machine, Windows, Python 3.13.1,
`python run_api.py` on 127.0.0.1:8000); recorded for transparency, **not**
as real-time or production-scale guarantees:

| Endpoint | Median | p95 | Notes |
|---|---|---|---|
| `POST /predict` | ~96 ms | ~106 ms | one record |
| `POST /predict/batch` (10 rows) | ~95 ms | ~105 ms | ≈105 rows/s |
| `POST /analyze` | ~307 ms | ~318 ms | ≈4.5 s first request (one-time lazy SHAP context build) |
| `GET /alerts` | ~52 ms | ~65 ms | ≈923 KiB committed-report payload |

`/analyze` breakdown: prediction ≈ 89 ms + SHAP ≈ 212 ms + risk ≈ 0.1 ms —
SHAP dominates **by design** (the analyst explicitly requests the analysis;
`/predict` stays SHAP-free). No optimization was applied; none was warranted.

## 10. Limitations

- **Dataset.** Structured KDD/NSL-KDD-style network-flow records; any caveat
  of that dataset family (age, train/test redundancy, simulated traffic)
  applies to every number in this report.
- **Test partition.** `Test_data.csv` is unlabelled: **no accuracy, F1, or
  any performance number is reported for it**, and none can be. All metrics
  here are 5-fold CV metrics on the labelled training distribution.
- **Live detection.** The system does not capture packets from a network
  interface; it classifies structured flow records supplied to the API.
- **Temporal correlation.** The dataset provides no usable
  timestamps/sessions/endpoints, so event grouping or attack-chain
  reconstruction is not defensibly supported on this data.
- **Explainability.** SHAP provides model attribution, not causal
  explanation; attributions depend on the model and the 100-row background.
- **Risk.** The risk score is a model-derived prioritization score — not
  attack probability, incident severity, or business impact.
- **Priority.** P1–P4 are analyst review priorities, not incident severity.
- **Generalization.** Performance on this dataset must not be read as
  equivalent performance on arbitrary real-world traffic; error composition
  by attack family is not measurable with the binary label available
  (recorded as an open measurement gap in the selection framework).

## 11. Future Work

Explicitly **not implemented**; listed as technically justified directions:

- PCAP/flow extraction pipelines to produce structured records from raw
  captures.
- Live/streaming ingestion in front of the existing prediction API.
- Temporal/session correlation — requires data with timestamps, session, and
  endpoint identifiers.
- Attack-type (multi-class) classification — requires labelled attack-family
  data; would also close the FN-composition measurement gap.
- Drift monitoring for deployment-time input/score distribution tracking.
