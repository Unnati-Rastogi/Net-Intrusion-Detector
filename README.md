# Net Intrusion Detector (NIDS MVP)

A Machine Learning-based Network Intrusion Detection System: a Random Forest
classifier trained on NSL-KDD-style network-flow records, served through a
FastAPI REST API with a minimal built-in dashboard.

- **Model**: scikit-learn `RandomForestClassifier` (200 trees,
  `class_weight='balanced'`, `random_state=42`)
- **Evaluation**: 5-fold stratified **cross-validation on the training data only**
  (see [Evaluation methodology](#evaluation-methodology) below)
- **Model comparison**: Logistic Regression baseline vs Random Forest vs
  XGBoost on identical CV folds (see [Model comparison](#model-comparison) below)
- **Explainability**: SHAP TreeExplainer over the production Random Forest,
  reported in the original 41 feature names (see
  [Explainability](#explainability-shap) below)
- **API**: FastAPI (`/health`, `/predict`, `/predict/batch`) with strict
  request validation and `feature_errors` on bad input
- **Dashboard**: vanilla HTML/JS served at `/` (sample loader, 41-feature
  form with dropdowns, batch test-data demo, recent-predictions table)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Data (not committed)

Place the two dataset files here before training:

| File | Purpose |
|---|---|
| `data/Train_data.csv` | labelled training data (25,192 rows × 42 cols incl. `class`: `normal`/`anomaly`) |
| `data/Test_data.csv` | unlabelled inference data (22,544 rows × 41 cols, **no** `class` column) |

Any NSL-KDD-style split with the standard 41 feature columns +
`duration, protocol_type, service, flag, ...` schema works. The split is
never merged: train is used for fitting/CV only, test only for inference.

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
  `data/Test_data.csv` is **unlabelled** — it has no ground-truth `class`
  column — so it is **never evaluated**, merged, or used here; the report
  only records its SHA-256 hash and shape to document that exclusion.
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
`data/Test_data.csv` is unlabelled and is evaluated by **no** candidate.

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

**Planned (NOT implemented yet):**

- Risk scoring
- Alert prioritization
- Optional lightweight event grouping
- Dashboard explanation integration

Live packet capture / real-time network monitoring is explicitly out of
scope for this project.

## Run API + dashboard

```bash
python run_api.py
# open http://127.0.0.1:8000  (dashboard)
# docs at http://127.0.0.1:8000/docs
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project layout

```
src/nids/        config, data loading, pipeline, training, evaluation, model comparison, explainability, prediction, API
static/          dashboard (index.html, app.js, app.css)
tests/           unittest suite (ML core + evaluation + comparison + API + dashboard)
models/          saved pipeline artifact
reports/         generated evaluation outputs (metrics.json, confusion_matrix.png, model_comparison.json/.csv)
data/            Train_data.csv / Test_data.csv (not committed)
```
