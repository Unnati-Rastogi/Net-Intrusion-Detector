# Model Selection Framework (Neutral — No Winner Declared)

**Status:** framework only. No scores, weights, rankings, or winner are assigned
in this document. The production prediction pipeline remains **RandomForest**
until a selection decision is made through the process defined here.

## Purpose and scope

This document defines *how* the project will choose its production detection
model when that decision is made. It applies to the evaluated candidates
(`lr_baseline`, `rf_production`, `xgb_candidate`) and to any future candidate.
It deliberately does **not** make the choice — it makes the choice auditable
when it happens.

**Evidence base:** all numbers referenced here come from

- `reports/metrics.json` — single-model CV evaluation
- `reports/model_comparison.json` / `.csv` — three-way comparison

and are **out-of-fold 5-fold StratifiedKFold metrics on the labelled training
data (`data/Train_data.csv`) — the training distribution**. They are not
held-out-test or production performance estimates. The unlabelled
`data/Test_data.csv` is never scored for selection.

## Criteria

For a network intrusion detector, accuracy is a weak contract: the two error
types have very different operational costs, and accuracy collapses both into
one number. The criteria below are what the decision will weigh. Each lists
where it is (or will be) measured.

### 1. Anomaly recall (missed-attack rate)

Share of actual attacks flagged. The primary security metric: every false
negative is an intrusion that goes unanswered. Measured as
`recall_anomaly` and, concretely, as the false-negative cell of the
out-of-fold confusion matrix (`cm[1][0]`, row = actual anomaly).

### 2. Anomaly precision (false-alarm burden)

Share of flagged connections that are genuinely attacks. Every false positive
costs analyst triage time; in deployment, a detector that floods analysts gets
muted or ignored, which quietly destroys its value. Measured as
`precision_anomaly` and the false-positive cell (`cm[0][1]`).

### 3. Anomaly F1

Harmonic mean of the two above. Useful as a single summary of the
recall/precision tension, but it hides *which* side the errors fall on, so it
is never sufficient alone.

### 4. False negatives — count and composition

Beyond the aggregate count: *which kinds* of attacks are missed. The current
artifacts do not break errors down by attack family (the dataset provides a
single binary label), so this is recorded as an **open measurement gap**. A
candidate that misses 37 scattered probes is operationally different from one
that misses 37 instances of one critical service — the aggregate cannot see
that.

### 5. False positives — count and operational scaling

The raw count must be translated into an expected alert burden. A dataset-level
FP rate of *r* does not predict the production alert rate (real traffic
volumes, class prevalence, and attack mix differ), but it allows candidate-vs-
candidate comparison under identical conditions.

### 6. Macro F1

Equal-weight treatment of both classes. Guards against a candidate that trades
the minority-relevant error profile for headline accuracy.

### 7. Computational cost

Three independently measured dimensions:

- **Training/refit cost** — matters for retraining cadence.
- **Per-prediction latency** — single-sample API response time (the `/predict`
  path). To be benchmarked on held-out *training* rows or synthetic rows —
  the unlabelled `Test_data.csv` is not used for benchmarking.
- **Batch throughput and memory** — the `/predict/batch` path and the size of
  the serialized artifact (RF: 200 fully-grown trees; XGB: 200 boosted rounds
  at depth 6 — the artifacts differ materially in size and traversal cost).

### 8. Model explainability compatibility

A planned project layer (SHAP) requires per-prediction attributions. Both tree
candidates are compatible with `shap.TreeExplainer`; the linear baseline would
need `LinearExplainer`. A deeper consideration: XGBoost supports **monotonic
constraints** (e.g. "more connections/bytes cannot decrease the anomaly score"),
which can encode security priors directly into the model — RF has no equivalent
mechanism. Explainability quality of the resulting attributions (stability,
consistency across similar samples) is part of this criterion, not just
"does a library exist".

### 9. Reproducibility

Deterministic seeds (verified: full-run fold metrics and confusion matrices
reproduce bit-identically), pinned library versions, and SHA-256 provenance in
every artifact. Candidates must be equally reproducible to be compared at all;
a candidate whose scores drift between runs is disqualified from fair
comparison regardless of its metrics.

### 10. Deployment complexity

Dependency footprint (XGBoost adds a native library and, on Windows, GPU
wheels that are unused), artifact size, serving integration (both tree
candidates live inside the same sklearn `Pipeline` contract, so the API layer
needs no changes for either), and upgrade/maintenance path for a college-
maintained codebase.

## Why accuracy alone must not decide

1. **It prices the two errors identically.** Accuracy weights a missed
   intrusion the same as a wasted triage minute. Operationally they are not
   comparable, and the right trade-off is a security decision, not a
   statistical default.
2. **It cannot distinguish error profiles.** In the current comparison, the
   two tree candidates' accuracies differ by roughly one fold standard
   deviation (0.9975 vs 0.9979) — accuracy alone would call them equivalent —
   yet their out-of-fold false-negative counts differ materially
   (RF: 50, XGB: 37). Identical headline, different security posture. This is
   the concrete demonstration, from this project's own artifacts, of why the
   decision must look inside the confusion matrix.
3. **It is distribution-bound.** Cross-validation accuracy on the training
   distribution says nothing about how the model degrades on novel attack
   patterns in deployment; treating it as a production guarantee is the exact
   misreading this project's evaluation labels warn against.
4. **It ignores probability quality.** Alert thresholds, prioritization, and
   risk scoring (planned layers) consume predicted *probabilities*, not hard
   labels. Two models with equal accuracy can be very differently calibrated.

## Decision procedure (when the review happens)

1. **Pre-register** the decision rules *before* inspecting candidate metrics:
   which criteria are hard minimums (e.g. an anomaly-recall floor), which are
   tie-breakers, and what evidence settles a genuine trade-off.
2. Run `python -m src.nids.compare`; every candidate is scored once with its
   documented, seeded configuration — no per-candidate tuning on CV test folds.
3. Record the decision in a written selection record: chosen model, the
   criteria table, the trade-offs accepted, and the rationale.
4. Any production switch must: re-run the determinism certification
   (`RUN_SLOW_EVAL_TESTS=1`), pass the full test suite, retrain the production
   artifact deliberately (`python -m src.nids.train`), and update the README
   claims. The Logistic Regression baseline is retained regardless, as the
   reference point.
5. Until all of the above happens, RandomForest remains the production model —
   by design, not by omission.

## Current evidence snapshot (factual, no ranking)

From `reports/model_comparison.json` (5-fold stratified CV on the labelled
training data, out-of-fold):

| Candidate | F1 (anomaly) | Recall (anomaly) | FN | FP |
|---|---|---|---|---|
| LR baseline | 0.9723 ± 0.0028 | 0.9678 ± 0.0053 | 378 | 269 |
| RF (production) | 0.9973 ± 0.0014 | 0.9957 ± 0.0025 | 50 | 14 |
| XGBoost | 0.9977 ± 0.0011 | 0.9968 ± 0.0024 | 37 | 16 |

(FN = missed attacks `cm[1][0]`, FP = false alarms `cm[0][1]`.) These numbers
describe the training distribution under cross-validation only. This table is
evidence for the procedure above; it is not a verdict.
