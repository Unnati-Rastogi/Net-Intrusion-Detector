# Net Intrusion Detector (NIDS MVP)

A Machine Learning-based Network Intrusion Detection System: a Random Forest
classifier trained on NSL-KDD-style network-flow records, served through a
FastAPI REST API with a minimal built-in dashboard.

- **Model**: scikit-learn `RandomForestClassifier` (200 trees,
  `class_weight='balanced'`, `random_state=42`)
- **Evaluation**: 5-fold stratified **cross-validation on the training data only**
  (see [Evaluation methodology](#evaluation-methodology) below)
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
src/nids/        config, data loading, pipeline, training, evaluation, prediction, API
static/          dashboard (index.html, app.js, app.css)
tests/           unittest suite (ML core + evaluation + API + dashboard)
models/          saved pipeline artifact
reports/         generated evaluation outputs (metrics.json, confusion_matrix.png)
data/            Train_data.csv / Test_data.csv (not committed)
```
