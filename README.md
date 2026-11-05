# Net Intrusion Detector (NIDS MVP)

A Machine Learning-based Network Intrusion Detection System: a Random Forest
classifier trained on NSL-KDD-style network-flow records, served through a
FastAPI REST API and a dashboard for model comparison and labeled-data replay.

- **Model**: scikit-learn `RandomForestClassifier` (200 trees,
  `class_weight='balanced'`, `random_state=42`)
- **Evaluation**: 5-fold stratified **cross-validation on the training data only**
  (see [Evaluation methodology](#evaluation-methodology) below)
- **Model comparison**: Logistic Regression baseline vs Random Forest vs
  XGBoost on identical CV folds (see [Model comparison](#model-comparison) below)
- **Explainability**: SHAP TreeExplainer over the production Random Forest,
  reported in the original 41 feature names (see
  [Explainability](#explainability-shap) below)
- **Risk assessment (offline)**: transparent, model-derived prioritization
  score combining P(anomaly) with SHAP evidence consensus
  (see [Transparent risk scoring](#transparent-risk-scoring) and
  `docs/risk_scoring.md`)
- **Alert prioritization (offline)**: deterministic analyst-review ordering
  (P1–P4) over the risk-scored records — no new score, no labels
  (see `docs/alert_prioritization.md`)
- **API**: FastAPI (`/health`, `/predict`, `/predict/batch`, `/analyze`,
  `/alerts`, and benchmark `/evaluation/*` views) with strict request validation
  and `feature_errors` on bad input
- **Dashboard**: vanilla HTML/CSS/JS served at `/`, with NSL-KDD and
  CIC-IDS2017 model comparisons plus labeled benchmark replay; live capture
  is not implemented

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Requires **Python 3.13** (developed and tested on 3.13.1). `requirements.txt`
is fully pinned, so a clean `pip install` reproduces the exact dependency set
(verified from an empty virtual environment).

## Data (not committed)

Add the matching dataset files here for training, inference, and evaluation:

| File | Purpose |
|---|---|
| `data/Train_data.csv` | original labeled 20% training subset (25,192 rows × 42 cols) used by the saved model |
| `data/KDDTrain+_full_labeled.csv` | full labeled training split (125,973 rows), used for the larger model comparison |
| `data/Test_data.csv` | 22,544 feature rows from the NSL-KDD KDDTest+ split (labels omitted in this copy) |
| `data/KDDTest+_labeled.csv` | matching labeled benchmark rows used only by the holdout evaluator (download separately; not committed) |

The model expects the NSL-KDD 41-feature schema (including
`duration, protocol_type, service, flag, ...`). The training split is used for
fitting and cross-validation; the separate KDDTest+ split is used for
inference and, when its matching labels are available, holdout evaluation.
Other datasets need a compatible feature mapping and a separate evaluation;
they are not plug-in inputs to this model.

The datasets are gitignored, so tests that read them skip automatically in a
clean checkout (see [Tests](#tests)); add them locally for the full suite.

## Train

```bash
python -m src.nids.train
# -> models/rf_ids_pipeline.joblib (fitted on all 25,192 rows)
```

The trained model artifact is included in this repository (≈8.6 MB), so this
step is optional unless you retrain: delete `models/rf_ids_pipeline.joblib`
and rerun the command above.

## Evaluate

Reproduces the model-selection cross-validation and writes the report
artifacts (no model artifact is changed):

```bash
python -m src.nids.evaluate
# -> reports/metrics.json           metrics + full provenance (JSON)
# -> reports/confusion_matrix.png   out-of-fold confusion matrix
```

Compare the three model candidates on the same folds (writes new report
files; changes no model artifact):

```bash
python -m src.nids.compare
# -> reports/model_comparison.json   LR / RF / XGB comparison + provenance (JSON)
# -> reports/model_comparison.csv    flat mean ± std table
```

## Evaluation methodology

`src/nids/evaluate.py` re-runs the exact procedure used to approve the
current model — nothing new is introduced:

- **Split**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over
  the 25,192 labelled rows of `data/Train_data.csv` (the same split the
  original model-selection numbers were computed with).
- **Fitting**: a fresh clone of the full pipeline (one-hot encoding included)
  is fitted inside every fold on that fold's training rows only — no leakage.
- **Metrics** (per fold, then mean ± population std, `ddof=0`): accuracy,
  anomaly precision/recall/F1 (positive class `anomaly`), normal
  precision/recall (positive class `normal`), macro F1, plus one out-of-fold
  confusion matrix (rows = actual, columns = predicted).

Current results on this machine (also in `reports/metrics.json`): accuracy
**0.9975 ± 0.0013**, F1 (anomaly) **0.9973 ± 0.0014**, macro F1
**0.9974 ± 0.0013**.

### What these numbers are — and are not

- ✅ Performance of this pipeline under cross-validation **on the training
  distribution** (`Train_data.csv`), suitable for comparing model variants
  with each other.
- ❌ **Not** an estimate of held-out-test or production performance.
  `data/Test_data.csv` by itself is **unlabelled** — it has no ground-truth
  `class` column — so this cross-validation report does not use it. The
  separate [labeled benchmark evaluation](#labeled-benchmark-evaluation) uses
  the original labels from a matching KDDTest+ source file and verifies every
  feature row before scoring.
- Both CSVs are NSL-KDD-style records. Any caveat that applies to that
  dataset family (age, redundancy between train and test captures, etc.)
  applies to any number this project reports.
- These metrics also describe the model *before* it was refit on all
  25,192 rows; the saved artifact is that final full-data fit.

`reports/metrics.json` includes dataset SHA-256 hashes and shapes, the exact
model parameters, library versions (sklearn/pandas/numpy/joblib/Python), the
git revision, and a UTC timestamp, so a run can be reproduced or audited.

### Model comparison

`src/nids/compare.py` evaluates three candidates on **identical folds** — the
same `StratifiedKFold(5, shuffle=True, random_state=42)` split, preprocessing
refit inside every fold — through one shared CV routine (`run_cv_folds` in
`src/nids/evaluate.py`):

| Candidate | Classifier | Notes |
|---|---|---|
| `lr_baseline` | `LogisticRegression(max_iter=2000, class_weight='balanced')` | project baseline; a `StandardScaler` is added *inside* its pipeline (fit on fold-training rows only) because a linear model cannot compare unscaled byte counts |
| `rf_production` | `RandomForestClassifier(**RF_PARAMS)` | the current production model, evaluated unchanged |
| `xgb_candidate` | `XGBClassifier` (200 trees, `max_depth=6`, `learning_rate=0.1`, subsample/colsample 0.8, `tree_method='hist'`, seeded) | `scale_pos_weight = n_normal / n_anomaly` mirrors `class_weight='balanced'` |

All three share **one preprocessing implementation** — the `ColumnTransformer`
from `src/nids/pipeline.py::build_pipeline` — so there is no second feature
schema. No hyperparameter is tuned on the CV test folds: each candidate is
scored once with its documented, seeded configuration. **No winner is
declared** by the artifact: production remains Random Forest until model
selection is reviewed against security-relevant criteria (anomaly recall,
precision trade-off, calibration, explainability).

Current results (mean ± std over 5 folds, training distribution only — full
detail in `reports/model_comparison.json`):

| Model | Accuracy | F1 (anomaly) | Recall (anomaly) | Macro F1 |
|---|---|---|---|---|
| LR baseline | 0.9743 ± 0.0025 | 0.9723 ± 0.0028 | 0.9678 ± 0.0053 | 0.9742 ± 0.0026 |
| Random Forest (production) | 0.9975 ± 0.0013 | 0.9973 ± 0.0014 | 0.9957 ± 0.0025 | 0.9974 ± 0.0013 |
| XGBoost | 0.9979 ± 0.0010 | 0.9977 ± 0.0011 | 0.9968 ± 0.0024 | 0.9979 ± 0.0010 |

The "what these numbers are — and are not" caveats above apply verbatim:
this is a comparison **on the training distribution** via cross-validation.
The dashboard presents this comparison separately from the fixed, labeled
KDDTest+ holdout result. Both views are labeled with their data scope. The
holdout is not an independent final test after it has been used to compare
candidates.

## Labeled benchmark evaluation

The dashboard's main view replays the saved model against the **22,544-row
NSL-KDD KDDTest+ holdout** and compares each guess with the dataset's known
label. This gives the project an honest evaluation result: attacks found,
attacks missed, false alarms, and detection rate by attack type. It is a
replay of historical benchmark data, **not live network monitoring**.

The project already contains the matching 41-feature `data/Test_data.csv`,
but that local copy has no labels. The small public
[NSL-KDD dataset archive mirror](https://github.com/jmnwong/NSL-KDD-Dataset)
provides [`KDDTest+.TXT`](https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt)
with those features plus attack-type and difficulty
labels (the mirror's README says it recovered the archive from the Wayback
Machine). Download that file, place it at
`data/KDDTest+_labeled.csv`, and run:

```bash
python -m src.nids.holdout
# -> reports/holdout_metrics.json
```

The evaluator refuses to score if the 41 feature columns or row order differ
from `data/Test_data.csv`. It uses the existing saved model, performs no
training, and records SHA-256 hashes for the features, labeled source, and
model. The dashboard reads the saved report through `GET /evaluation/holdout`.

Current saved-model result on this dataset: **78.07% accuracy**, **63.51%
attack recall**, **4,683 missed attacks**, and **260 false alarms**. The low
attack recall is useful evidence that the current model needs improvement;
the near-perfect training cross-validation score alone hid that weakness.
These results are specific to NSL-KDD and do not predict performance on a
modern network or a different flow format.

### Compare models on the labeled test set

The dashboard also compares Logistic Regression, Random Forest, and XGBoost
on the same KDDTest+ rows. These candidates are fitted on the full 125,973-row
KDDTrain+ split; the table includes the current saved Random Forest, which
was fitted on the project's 25,192-row subset. It shows accuracy, attack
precision/recall/F1, and inference time for the 22,544 test rows.

Rebuild the comparison report with:

```bash
python -m src.nids.compare_holdout
# -> reports/holdout_model_comparison.json
```

Current result: XGBoost is the strongest of these candidates on this test
split at **79.8% accuracy, 96.9% attack precision, and 66.7% attack recall**.
None of the compared versions meets the requested 90% minimum for all three
metrics. The dashboard states this directly. Since the test split has now
been used to compare candidates, a fresh independent dataset is required for
an unbiased final score after selecting or tuning a model.

## CIC-IDS2017 day-based model comparison

The project also compares the same three model families on the corrected
CIC-IDS2017 flow release. The archive and extracted daily CSVs live in
`data/CIC-IDS2017/` and are excluded from Git because the expanded data is
about 1.15 GB. The correction authors document known flow/label issues and
provide guidance for the `Attempted` labels:
[corrected dataset and label guidance](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html).
The original dataset description is on the
[Canadian Institute for Cybersecurity page](https://www.unb.ca/cic/datasets/ids-2017.html).

Run a fresh comparison with:

```bash
python -m src.nids.compare_cic
# -> reports/cic_ids2017_model_comparison.json
```

Every run freshly fits Logistic Regression, Random Forest, and XGBoost. It
trains on Monday–Wednesday (1,190,343 flows), uses Thursday (362,076 flows)
to choose each model's threshold and the candidate with the strongest
validation F1, then evaluates all candidates on Friday (547,557 flows).
Friday does not choose a model or threshold. Threshold selection first seeks
the highest attack recall with at least 90% validation precision; when a
candidate cannot meet that precision floor, its best validation F1 is used.
The dashboard shows the latest report, including per-attack-family Friday
recall. Scores describe this controlled benchmark and do not establish live
network performance.

`BENIGN` and labels ending in `- Attempted` map to the benign class, following
the correction authors' guidance for binary detection. The classifier drops
flow IDs, addresses, timestamps, and `Attempted Category` so it cannot learn
identifiers, capture ordering, or label-derived metadata. CIC and NSL-KDD use
different feature schemas, so each comparison uses its own preprocessing.
An unlabeled dataset can be scored, but its labels are required to know which
model performed better.

Latest Friday results from the saved report, using thresholds chosen on
Thursday:

| Model | Accuracy | Attack precision | Attack recall | Attack F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 71.66% | 99.39% | 39.38% | 56.41% |
| Random Forest (selected on Thursday) | 92.48% | 97.48% | 86.08% | 91.42% |
| XGBoost | 84.95% | 75.58% | 100.00% | 86.09% |

No candidate reached 90% on accuracy, attack precision, and attack recall
simultaneously. Random Forest gave the strongest Friday F1 and precision;
XGBoost caught nearly every attack at the cost of more false alarms. Recall
also varied by family: on Friday, Random Forest found 5.98% of Botnet and
78.12% of Portscan flows, while XGBoost found 99.46% and 99.999%, respectively.
These are results from this day split, not a general guarantee.

### One-file intrusion scan

The application-facing workflow uses a saved CIC detector. A user uploads one
compatible CIC flow CSV; the detector labels every flow as `attack` or
`benign`, returns the number flagged, and produces a downloadable results
file. A `Label` column is optional: when present, the response also includes
evaluation metrics for that upload.

The default detector is the candidate selected by the day-based comparison:
Random Forest, fitted on Monday--Wednesday with its Thursday-selected
threshold. Build it after the comparison has completed:

```bash
python -m src.nids.cic_detector
# -> models/cic_ids2017_detector.joblib
```

With the API running, scan one file:

```bash
curl.exe -X POST "http://127.0.0.1:8000/detections/cic/scan" `
  -H "Content-Type: text/csv" --data-binary "@network-flows.csv"
```

`GET /detections/cic/status` reports whether this detector is ready.
`POST /detections/cic/explain` accepts one flow's CIC feature values and
returns the selected detector's SHAP evidence. The browser upload interface
will use these endpoints; users do not train a model for each upload.

### Uploaded CIC scenario profiles

The backend can also create a separate research profile from a user's labeled CIC
flow CSV. It trains all three candidates, chooses a candidate and threshold
from an internal validation split, and saves every candidate so a later
labeled scenario can be compared against the same trained models. Profiles
stay separate from the existing NSL-KDD checker. Uploads must use the corrected
CIC feature schema and include a `Label` column for training or comparison.
`BENIGN` and `- Attempted` labels follow the binary mapping described above.

The training endpoint accepts the CSV bytes directly, with
`Content-Type: text/csv`:

```bash
curl.exe -X POST "http://127.0.0.1:8000/profiles/cic/train?name=Scenario%20A" `
  -H "Content-Type: text/csv" --data-binary "@scenario-a.csv"
```

The response includes a job ID. Poll `GET /profiles/cic/jobs/{job_id}` until
training finishes, then use the returned profile ID:

- `GET /profiles/cic` lists saved profiles.
- `POST /profiles/cic/{profile_id}/evaluate` compares all saved candidates
  against a separate labeled scenario without retraining them on that file.
- `POST /profiles/cic/{profile_id}/predict` scans an unlabeled CSV in the same
  schema and returns a downloadable CSV with each flow's prediction and score.
- `POST /profiles/cic/{profile_id}/explain` accepts one feature-value JSON
  object and returns SHAP evidence for the profile's validation-selected model.

Uploads use raw CSV request bodies, so a future browser file control can send
the file directly. Saved profiles and prediction outputs are kept in ignored
`models/profiles/` and `reports/profiles/` folders.

## Explainability (SHAP)

`src/nids/explain.py` explains the **saved production pipeline** with SHAP's
`TreeExplainer` (feature_perturbation="interventional", deterministic 100-row
training background, seed 42). It is a read-only side layer: no retraining,
no changes to the model artifact or the prediction path, and it is **not** on
the API hot path.

```bash
python -m src.nids.explain
# -> reports/explanations.json   global importance + methodology + examples
```

How it works, in one paragraph: the pipeline's preprocessing turns each record
into 116 transformed columns (66+11+3 one-hot, 36 numeric passthrough; the two
constant columns are dropped). TreeExplainer attributes the model's
**P(anomaly)** for each transformed column, and the layer sums each original
feature's columns back together — so every explanation is reported in the
**original 41 feature names** (one-hot groups aggregated by summation; the
dropped constants contribute exactly 0). The baseline (base value) is the
model's mean P(anomaly) over the background sample, and base + summed
contributions reconstructs P(anomaly) exactly (verified to <1e-6 in tests).

The global report uses a deterministic sample of **1,000 labelled training
rows** (seed 42) — full-data explanation would take roughly an hour at
~0.12 s/sample; the 100-row background builds in milliseconds. Global
importance is the mean absolute SHAP value per original feature.

**What SHAP values are — and are not:** they are feature-attribution values
describing **this model's output** — how much each feature's value moves the
model's P(anomaly) for a given record away from the baseline. They are **not**
causal explanations of network behaviour, and global SHAP importance is not
causal or real-world feature importance. Everything describes the model on
the **training distribution** (`Train_data.csv`); `Test_data.csv` is never
scored or explained.

## Transparent risk scoring (offline)

`src/nids/risk.py` turns prediction + SHAP evidence into a bounded
prioritization score (full design and reasoning in `docs/risk_scoring.md`):

```
risk = 100 · P(anomaly) · (1 + consensus_ratio) / 2      ∈ [0, 100]
consensus_ratio = Σ(shap) / Σ(|shap|)                    ∈ [-1, 1]
```

The SHAP evidence enters only through its **directional consensus** —
adding attribution magnitude would double-count the model score, because
`base + Σ(shap) = P(anomaly)` exactly (an additive candidate was verified
rank-identical to the score alone and rejected). Fixed policy thresholds
(LOW < 20 ≤ MEDIUM < 50 ≤ HIGH < 80 ≤ CRITICAL) are anchored to the model's
0.5 operating point, not fitted to data.

```bash
python -m src.nids.risk
# -> reports/risk_assessment.json   per-record risk + aggregates + methodology
```

Risk scores are **model-derived prioritization scores** evaluated on the
labelled training distribution — **not** probabilities of attack or
compromise, not impact estimates, not causal risk, and the level names are
alert-prioritization vocabulary, not attack severity. The per-record API
workflow (`POST /analyze`) consumes this layer, and the committed report is
served by `GET /alerts`. Event grouping and incident correlation are later
stages.

## Alert prioritization (offline)

`src/nids/prioritize.py` orders the risk-scored records for analyst review
(full design in `docs/alert_prioritization.md`):

```
1. risk_score descending   2. anomaly_score descending
3. risk_level by explicit rank   4. sample_id ascending

CRITICAL -> P1   HIGH -> P2   MEDIUM -> P3   LOW -> P4 (review backlog)
```

No new score is computed and no SHAP/model work is redone — it is a pure,
deterministic re-ordering of the existing risk records (a 1,000-alert
report builds in ~0.2 s). True labels are dropped from the alert records:
they structurally cannot influence the operational ordering. P1 means
"highest analyst review priority", never "most dangerous attack". Every
report record carries `sample_id` (position within the 1,000-row report
sample) and `source_row_id` (its zero-based row in `Train_data.csv`); the
latter is lineage metadata only and never influences ordering.

```bash
python -m src.nids.prioritize
# -> reports/alert_prioritization.json   ordered alerts + counts + methodology
```

An offline deterministic prioritization of model-generated alerts, served
verbatim by `GET /alerts` (the committed report stays authoritative; nothing
is re-ranked at request time).

## Implementation status

**Currently implemented:**

- Logistic Regression baseline (evaluation candidate)
- Random Forest production model + training pipeline
- 5-fold stratified CV evaluation with provenance
  (`reports/metrics.json`, `reports/confusion_matrix.png`)
- FastAPI service + dashboard
- Automated tests

**New (previous iterations):**

- XGBoost comparison — evaluation artifact only; the production prediction
  pipeline is unchanged and still Random Forest

**New (this iteration):**

- SHAP explainability side-layer (`reports/explanations.json`); production
  model, prediction pipeline, and API are unchanged

**New (this stage — engineering hardening):**

- Non-finite rate values (NaN/±inf) are rejected as 422 validation errors
  instead of reaching the model
- Structured JSON error contracts (generic 500 handler; clear artifact
  errors on `/alerts`), verified by robustness and concurrency tests
- Clean-checkout portability: dataset-dependent tests skip when
  `data/*.csv` is absent
- GitHub Actions CI (portable test subset on Python 3.13)

**Planned (NOT implemented yet):**

- Optional lightweight event grouping
- Incident correlation

Live packet capture / real-time network monitoring is explicitly out of
scope for this project.

## Run API + dashboard

```bash
python run_api.py
# open http://127.0.0.1:8000  (dashboard)
# docs at http://127.0.0.1:8000/docs
```

The dashboard implements the analyst workflow on top of the API:
`POST /analyze` runs detection → SHAP explanation → risk → review priority
for one record (the user explicitly requests the analysis; `/predict` stays
fast and SHAP-free), and `GET /alerts` serves the committed prioritization
report verbatim — nothing is re-ranked in the browser. `anomaly_score` is
the model's P(anomaly) output (a model score, not an attack probability);
risk is a model-derived prioritization score, not attack severity; P1–P4
order analyst review, nothing more. Analysis operates on structured
network-flow records, not live traffic.

## API performance (measured)

Single development machine (Windows, Python 3.13.1, pinned requirements),
30 repetitions against `python run_api.py` on 127.0.0.1:8000 — recorded for
transparency, not as a capacity claim:

| Endpoint | Median | p95 | Notes |
|---|---|---|---|
| `POST /predict` | ~96 ms | ~106 ms | one record |
| `POST /predict/batch` (10 rows) | ~95 ms | ~105 ms | ≈105 rows/s |
| `POST /analyze` | ~307 ms | ~318 ms | first request ≈4.5 s (one-time lazy SHAP context build) |
| `GET /alerts` | ~52 ms | ~65 ms | ≈923 KiB report payload |

`/analyze` breakdown: prediction ≈ 89 ms + SHAP explanation ≈ 212 ms + risk
≈ 0.1 ms — SHAP dominates by design: the analyst explicitly requests the
analysis, and `/predict` stays SHAP-free. No real-time or production-scale
performance is implied.

## Tests

```bash
python -m unittest discover -s tests -v
```

241 tests across the ML core, evaluation, comparison, explainability, risk,
prioritization, API, dashboard, and hardening suites.

- **Slow/opt-in tests** (full-dataset determinism and CV-reproduction
  checks, several minutes):
  `RUN_SLOW_EVAL_TESTS=1 python -m unittest discover -s tests -v`.
  Skipped by default.
- **Without `data/*.csv`**, the 41 dataset-dependent tests skip
  automatically, leaving a portable subset — this is what CI runs. Add the
  datasets locally for the full suite.
- **CI**: `.github/workflows/ci.yml` installs the pinned requirements on
  Python 3.13 and runs the portable test subset on every push/PR.

## Project layout

```
src/nids/        config, data loading, pipeline, training, evaluation, model comparison, explainability, risk assessment, alert prioritization, prediction, API
static/          dashboard (index.html, app.js, app.css)
tests/           unittest suite (ML core + evaluation + comparison + API + dashboard)
models/          saved pipeline artifact
reports/         generated evaluation outputs (metrics.json, confusion_matrix.png, model_comparison.json/.csv)
data/            Train_data.csv / Test_data.csv (not committed)
```
