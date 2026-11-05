"""Compare LR, Random Forest, and XGBoost on the fixed NSL-KDD holdout.

Each candidate is fitted only on the full KDDTrain+ split and evaluated once
on KDDTest+. This is a benchmark comparison, not production monitoring; do
not tune a model using these holdout results and then report them as an
independent final test.

Run: ``python -m src.nids.compare_holdout``
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, compare, holdout
from .pipeline import build_pipeline
from .predict import load_model

FULL_TRAIN_PATH = config.DATA_DIR / "KDDTrain+_full_labeled.csv"
REPORT_PATH = config.PROJECT_ROOT / "reports" / "holdout_model_comparison.json"


def load_full_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load full NSL-KDD train labels into the app's binary 41-feature schema."""
    if not FULL_TRAIN_PATH.exists():
        raise FileNotFoundError(f"full labeled training file not found: {FULL_TRAIN_PATH}")
    source = pd.read_csv(FULL_TRAIN_PATH, header=None, names=holdout.SOURCE_COLUMNS)
    if len(source) != 125_973:
        raise ValueError(f"expected 125,973 training records, found {len(source)}")
    X = source[config.FEATURE_COLUMNS]
    attack_type = source["attack_type"].astype("string").str.strip().str.lower()
    if attack_type.isna().any() or attack_type.eq("").any():
        raise ValueError("full training data contains invalid attack labels")
    y = (attack_type != "normal").astype(int)
    return X, y


def candidate_models(y: pd.Series) -> dict[str, tuple[str, Any]]:
    """Build the exact comparison candidates, weighting XGBoost from full data."""
    shared_preprocessing = build_pipeline().named_steps["preprocessing"]
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    xgb_pipeline = Pipeline(
        [
            ("preprocessing", shared_preprocessing),
            (
                "classifier",
                XGBClassifier(
                    **{
                        **config.XGB_PARAMS,
                        "scale_pos_weight": positive_weight,
                        "n_jobs": -1,
                    }
                ),
            ),
        ]
    )
    registry = {
        ("rf_full_training" if name == "rf_production" else name): (
            str(spec["description"]), spec["factory"]()
        )
        for name, spec in compare.candidate_registry().items()
        if name != "xgb_candidate"
    }
    registry["xgb_candidate"] = (
        "XGBoost; class weighting computed from the full training split",
        xgb_pipeline,
    )
    return registry


def compare_candidates() -> dict[str, Any]:
    """Train each candidate on KDDTrain+, score KDDTest+, and record timing."""
    X_train, y_train = load_full_training_data()
    X_test, y_test, attack_type = holdout.load_labeled_holdout()
    results: dict[str, dict[str, Any]] = {}

    for name, (description, model) in candidate_models(y_train).items():
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        prediction = pd.Series(model.predict(X_test).astype(int), index=X_test.index)
        predict_seconds = time.perf_counter() - predict_started

        attack_recall_by_type = {}
        for kind, indexes in attack_type.groupby(attack_type, sort=True).groups.items():
            if kind == "normal":
                continue
            attack_recall_by_type[str(kind)] = round(
                float(prediction.loc[indexes].mean()), 6
            )

        results[name] = {
            "description": description,
            "training_records": int(len(X_train)),
            "metrics": {
                "accuracy": round(float(accuracy_score(y_test, prediction)), 6),
                "precision_anomaly": round(float(precision_score(y_test, prediction, zero_division=0)), 6),
                "recall_anomaly": round(float(recall_score(y_test, prediction, zero_division=0)), 6),
                "f1_anomaly": round(float(f1_score(y_test, prediction, zero_division=0)), 6),
            },
            "counts": {
                "false_alarms": int(((y_test == 0) & (prediction == 1)).sum()),
                "missed_attacks": int(((y_test == 1) & (prediction == 0)).sum()),
                "correctly_detected_attacks": int(((y_test == 1) & (prediction == 1)).sum()),
                "correctly_cleared_normal": int(((y_test == 0) & (prediction == 0)).sum()),
            },
            "attack_recall_by_type": attack_recall_by_type,
            "timing_seconds": {
                "fit": round(fit_seconds, 3),
                "predict_all_test_rows": round(predict_seconds, 3),
                "predict_per_1000_rows": round(predict_seconds * 1000 / len(X_test), 4),
            },
        }

    # Include the model the dashboard currently serves, trained on the
    # project's original 20% KDDTrain+ subset, for a direct reference row.
    load_started = time.perf_counter()
    saved_model = load_model()
    load_seconds = time.perf_counter() - load_started
    predict_started = time.perf_counter()
    saved_prediction = pd.Series(saved_model.predict(X_test).astype(int), index=X_test.index)
    predict_seconds = time.perf_counter() - predict_started
    saved_recall_by_type = {}
    for kind, indexes in attack_type.groupby(attack_type, sort=True).groups.items():
        if kind == "normal":
            continue
        saved_recall_by_type[str(kind)] = round(
            float(saved_prediction.loc[indexes].mean()), 6
        )
    results["rf_saved_model"] = {
        "description": "Existing saved Random Forest, fitted on the project's original 20% training subset",
        "training_records": 25_192,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, saved_prediction)), 6),
            "precision_anomaly": round(float(precision_score(y_test, saved_prediction, zero_division=0)), 6),
            "recall_anomaly": round(float(recall_score(y_test, saved_prediction, zero_division=0)), 6),
            "f1_anomaly": round(float(f1_score(y_test, saved_prediction, zero_division=0)), 6),
        },
        "counts": {
            "false_alarms": int(((y_test == 0) & (saved_prediction == 1)).sum()),
            "missed_attacks": int(((y_test == 1) & (saved_prediction == 0)).sum()),
            "correctly_detected_attacks": int(((y_test == 1) & (saved_prediction == 1)).sum()),
            "correctly_cleared_normal": int(((y_test == 0) & (saved_prediction == 0)).sum()),
        },
        "attack_recall_by_type": saved_recall_by_type,
        "timing_seconds": {
            "fit": None,
            "load_saved_model": round(load_seconds, 3),
            "predict_all_test_rows": round(predict_seconds, 3),
            "predict_per_1000_rows": round(predict_seconds * 1000 / len(X_test), 4),
        },
    }

    return {
        "report": "nids_full_train_holdout_model_comparison",
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": {
            "kind": "fixed_train_test_benchmark_comparison",
            "training_records": int(len(X_train)),
            "testing_records": int(len(X_test)),
            "features": int(X_train.shape[1]),
            "positive_class": "attack",
            "note": (
                "LR, RF, and XGBoost candidates were fitted only on the full NSL-KDD "
                "KDDTrain+ split; the current saved RF uses the project's original "
                "20% training subset. All rows were scored on the same separate "
                "KDDTest+ split. These are benchmark-specific "
                "results, not proof of performance on current or unrelated networks. "
                "Do not tune candidates on this test set and present these same scores "
                "as an independent final evaluation."
            ),
        },
        "training_dataset": {
            "path": "data/KDDTrain+_full_labeled.csv",
            "sha256": holdout.sha256_file(FULL_TRAIN_PATH),
            "normal_records": int((y_train == 0).sum()),
            "attack_records": int((y_train == 1).sum()),
        },
        "testing_dataset": {
            "name": "NSL-KDD KDDTest+",
            "features_path": "data/Test_data.csv",
            "labels_path": "data/KDDTest+_labeled.csv",
            "features_sha256": holdout.sha256_file(config.TEST_CSV),
            "labels_sha256": holdout.sha256_file(holdout.LABELED_TEST_PATH),
            "normal_records": int((y_test == 0).sum()),
            "attack_records": int((y_test == 1).sum()),
        },
        "provenance": {
            "saved_model_sha256": holdout.sha256_file(config.MODEL_PATH),
        },
        "models": results,
    }


def main() -> dict[str, Any]:
    report = compare_candidates()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Full-train / labeled-test model comparison")
    for name, result in report["models"].items():
        m = result["metrics"]
        t = result["timing_seconds"]
        print(
        f"{name}: accuracy={m['accuracy']:.1%}, precision={m['precision_anomaly']:.1%}, "
            f"recall={m['recall_anomaly']:.1%}, F1={m['f1_anomaly']:.1%}, "
            f"fit={t['fit']}s, test-inference={t['predict_all_test_rows']}s"
        )
    print(f"saved report to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
