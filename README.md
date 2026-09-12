# Net Intrusion Detector (NIDS MVP)

A Machine Learning-based Network Intrusion Detection System: a Random Forest
classifier trained on NSL-KDD-style network-flow records, served through a
FastAPI REST API with a minimal built-in dashboard.

- **Model**: scikit-learn `RandomForestClassifier` (200 trees,
  `class_weight='balanced'`, `random_state=42`)
- **Validation**: 5-fold stratified CV on the training data —
  accuracy **0.9975 ± 0.0013**, F1 (anomaly) **0.9973 ± 0.0014**
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
# -> models/rf_ids_pipeline.joblib (fitted pipeline; 5-fold CV metrics:
#    accuracy 0.9975 ± 0.0013, F1(anomaly) 0.9973 ± 0.0014)
```

The trained model artifact is included in this repository (≈8.6 MB), so this
step is optional unless you retrain: delete `models/rf_ids_pipeline.joblib`
and rerun the command above.

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
src/nids/        config, data loading, pipeline, training, prediction, API
static/          dashboard (index.html, app.js, app.css)
tests/           unittest suite (ML core + API + dashboard)
models/          saved pipeline artifact
data/            Train_data.csv / Test_data.csv (not committed)
```
