"""Train and compare the NIDS candidates on corrected CIC-IDS2017 flows.

Every invocation fits Logistic Regression, Random Forest, and XGBoost from
scratch. Monday–Wednesday are training data, Thursday selects the model and
operating threshold, and Friday is held back for the final evaluation. No
candidate is assumed to be the winner.

Run from the project root with ``python -m src.nids.compare_cic``.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "CIC-IDS2017"
REPORT_PATH = PROJECT_ROOT / "reports" / "cic_ids2017_model_comparison.json"
DAY_FILES = {
    "monday": "monday.csv",
    "tuesday": "tuesday.csv",
    "wednesday": "wednesday.csv",
    "thursday": "thursday.csv",
    "friday": "friday.csv",
}
TRAIN_DAYS = ("monday", "tuesday", "wednesday")
VALIDATION_DAY = "thursday"
TEST_DAY = "friday"

# These values identify a flow, its capture time, or a label-derived field.
# They must not be allowed to help the classifier memorize hosts, ordering,
# timestamps, or ground truth metadata.
EXCLUDED_COLUMNS = (
    "id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Attempted Category",
)
LABEL_COLUMN = "Label"
CATEGORICAL_COLUMNS = ("Protocol",)
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    paths = {day: data_dir / name for day, name in DAY_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CIC-IDS2017 daily CSVs are missing: " + ", ".join(missing)
        )
    return paths


def _schema(paths: dict[str, Path]) -> tuple[list[str], list[str]]:
    expected_header: list[str] | None = None
    for day, path in paths.items():
        header = list(pd.read_csv(path, nrows=0).columns)
        if len(header) != len(set(header)):
            raise ValueError(f"{day} CSV contains duplicate column names")
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise ValueError(f"{day} CSV does not match the other daily schemas")

    assert expected_header is not None
    required = {LABEL_COLUMN, *EXCLUDED_COLUMNS, *CATEGORICAL_COLUMNS}
    missing = sorted(required.difference(expected_header))
    if missing:
        raise ValueError("CIC-IDS2017 schema is missing columns: " + ", ".join(missing))
    feature_columns = [
        name for name in expected_header
        if name not in EXCLUDED_COLUMNS and name != LABEL_COLUMN
    ]
    if not feature_columns:
        raise ValueError("CIC-IDS2017 schema has no model features")
    return expected_header, feature_columns


def _load_day(path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    use_columns = [*feature_columns, LABEL_COLUMN]
    dtypes: dict[str, Any] = {name: np.float32 for name in feature_columns}
    dtypes.update({name: "string" for name in CATEGORICAL_COLUMNS})
    dtypes[LABEL_COLUMN] = "string"
    frame = pd.read_csv(path, usecols=use_columns, dtype=dtypes, low_memory=False)
    frame = frame.loc[:, use_columns]

    raw_labels = frame.pop(LABEL_COLUMN).str.strip()
    if raw_labels.isna().any() or raw_labels.eq("").any():
        raise ValueError(f"{path.name} contains missing labels")

    # The corrected release documents “Attempted” records as flows that do
    # not show malicious behavior. Keep those in the benign class for this
    # binary detector; retain their source labels for provenance and counts.
    normalized = raw_labels.str.casefold()
    benign = normalized.eq("benign") | normalized.str.endswith(" - attempted")
    targets = (~benign).astype(np.uint8)

    numeric_columns = [name for name in feature_columns if name not in CATEGORICAL_COLUMNS]
    for name in numeric_columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame, targets, raw_labels


def _preprocessor(feature_columns: list[str], *, scale: bool) -> ColumnTransformer:
    numeric_columns = [name for name in feature_columns if name not in CATEGORICAL_COLUMNS]
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def _candidate_models(
    feature_columns: list[str], positive_weight: float
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for name, classifier, scaled in (
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=positive_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    ):
        models[name] = Pipeline(
            [
                ("preprocessing", _preprocessor(feature_columns, scale=scaled)),
                ("classifier", classifier),
            ]
        )
    return models


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    )
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, predicted)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predicted)), 6),
        "precision_attack": round(float(precision_score(y_true, predicted, zero_division=0)), 6),
        "recall_attack": round(float(recall_score(y_true, predicted, zero_division=0)), 6),
        "f1_attack": round(float(f1_score(y_true, predicted, zero_division=0)), 6),
        "average_precision": round(float(average_precision_score(y_true, scores)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "counts": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def _select_threshold(
    y_true: np.ndarray, scores: np.ndarray, precision_floor: float = 0.90
) -> tuple[float, dict[str, Any], str]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        threshold = 0.5
        policy = "default_threshold_no_candidates"
    else:
        candidate_precision = precision[:-1]
        candidate_recall = recall[:-1]
        f1_values = 2 * candidate_precision * candidate_recall / np.maximum(
            precision[:-1] + recall[:-1], np.finfo(float).eps
        )
        eligible = np.flatnonzero(candidate_precision >= precision_floor)
        if len(eligible):
            best_recall = candidate_recall[eligible].max()
            eligible = eligible[np.isclose(candidate_recall[eligible], best_recall)]
            best_f1 = f1_values[eligible].max()
            eligible = eligible[np.isclose(f1_values[eligible], best_f1)]
            # Prefer the higher threshold on a tie to reduce alert volume.
            best_index = int(eligible[np.argmax(thresholds[eligible])])
            policy = f"highest_recall_at_validation_precision_at_least_{precision_floor:.0%}"
        else:
            best_value = f1_values.max()
            best_index = int(np.flatnonzero(np.isclose(f1_values, best_value))[-1])
            policy = "maximum_validation_f1_no_threshold_met_precision_floor"
        threshold = float(thresholds[best_index])
    return threshold, _metrics(y_true, scores, threshold), policy


def _attack_family_metrics(
    labels: pd.Series, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, int | float]]:
    predicted = scores >= threshold
    normalized = labels.str.casefold()
    attack_mask = ~normalized.eq("benign") & ~normalized.str.endswith(" - attempted")
    result: dict[str, dict[str, int | float]] = {}
    for family in sorted(labels[attack_mask].unique(), key=str.casefold):
        indexes = labels.eq(family).to_numpy()
        count = int(indexes.sum())
        detected = int(predicted[indexes].sum())
        result[str(family)] = {
            "records": count,
            "detected": detected,
            "missed": count - detected,
            "recall": round(detected / count, 6) if count else 0.0,
        }
    return result


def compare_models(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Fit every candidate, select on Thursday, and report Friday once."""
    paths = _input_paths(data_dir)
    _, feature_columns = _schema(paths)
    print("Loading CIC-IDS2017 daily flows…", flush=True)

    day_data: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
    day_counts: dict[str, Any] = {}
    label_counts: dict[str, dict[str, int]] = {}
    for day, path in paths.items():
        X, y, labels = _load_day(path, feature_columns)
        day_data[day] = (X, y, labels)
        day_counts[day] = {
            "records": int(len(y)),
            "benign": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
        }
        label_counts[day] = {
            str(label): int(count)
            for label, count in labels.value_counts().sort_index().items()
        }
        print(
            f"  {day.title()}: {len(y):,} rows, "
            f"{day_counts[day]['attack']:,} attacks after label mapping",
            flush=True,
        )

    X_train = pd.concat([day_data[day][0] for day in TRAIN_DAYS], ignore_index=True)
    y_train = pd.concat([day_data[day][1] for day in TRAIN_DAYS], ignore_index=True).to_numpy()
    X_validation, y_validation, _ = day_data[VALIDATION_DAY]
    X_test, y_test, test_labels = day_data[TEST_DAY]
    positive_weight = float(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1))
    models = _candidate_models(feature_columns, positive_weight)

    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"Training {name.replace('_', ' ')} on {len(y_train):,} rows…", flush=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        validation_started = time.perf_counter()
        validation_scores = model.predict_proba(X_validation)[:, 1]
        validation_seconds = time.perf_counter() - validation_started
        threshold, validation_metrics, threshold_policy = _select_threshold(
            y_validation, validation_scores
        )

        test_started = time.perf_counter()
        test_scores = model.predict_proba(X_test)[:, 1]
        test_seconds = time.perf_counter() - test_started
        test_metrics = _metrics(y_test.to_numpy(), test_scores, threshold)
        test_metrics_at_default = _metrics(y_test.to_numpy(), test_scores, 0.5)

        results[name] = {
            "training_records": int(len(y_train)),
            "validation": validation_metrics,
            "threshold_selection_policy": threshold_policy,
            "test": test_metrics,
            "test_at_default_threshold": test_metrics_at_default,
            "attack_family_test": _attack_family_metrics(test_labels, test_scores, threshold),
            "timing_seconds": {
                "fit": round(fit_seconds, 3),
                "validation_inference": round(validation_seconds, 3),
                "test_inference": round(test_seconds, 3),
                "test_inference_per_1000_rows": round(test_seconds * 1000 / len(y_test), 4),
            },
            "warnings": [str(item.message) for item in fit_warnings],
        }
        vm = results[name]["validation"]
        tm = results[name]["test"]
        print(
            f"  validation F1={vm['f1_attack']:.3f}, recall={vm['recall_attack']:.3f}; "
            f"Friday F1={tm['f1_attack']:.3f}, recall={tm['recall_attack']:.3f}, "
            f"precision={tm['precision_attack']:.3f}",
            flush=True,
        )

    selected_name = max(
        results,
        key=lambda name: (
            results[name]["validation"]["f1_attack"],
            results[name]["validation"]["recall_attack"],
            results[name]["validation"]["precision_attack"],
        ),
    )
    report = {
        "report": "nids_cic_ids2017_day_split_model_comparison",
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Corrected CIC-IDS2017",
            "source_archive": "data/CIC-IDS2017/CICIDS2017_improved.zip",
            "archive_sha256": sha256_file(data_dir / "CICIDS2017_improved.zip"),
            "daily_files": {
                day: {"path": f"data/CIC-IDS2017/{path.name}", "sha256": sha256_file(path)}
                for day, path in paths.items()
            },
            "records_by_day": day_counts,
            "source_labels_by_day": label_counts,
            "features": len(feature_columns),
            "feature_columns": feature_columns,
            "excluded_columns": list(EXCLUDED_COLUMNS),
        },
        "evaluation": {
            "kind": "chronological_day_holdout",
            "training_days": list(TRAIN_DAYS),
            "validation_day": VALIDATION_DAY,
            "test_day": TEST_DAY,
            "training_records": int(len(y_train)),
            "validation_records": int(len(y_validation)),
            "test_records": int(len(y_test)),
            "positive_class": "attack",
            "label_policy": (
                "BENIGN and labels ending in ' - Attempted' are mapped to benign; "
                "all other source labels are mapped to attack. Attempted Category "
                "is excluded from model features because it is label-derived metadata."
            ),
            "model_selection": (
                "All candidates are freshly fitted on every run. Each candidate's "
                "threshold is independently selected on Thursday to maximize attack "
                "recall while keeping validation precision at or above 90%; if no "
                "threshold meets that precision floor, maximum validation F1 is used. "
                "The candidate with the highest Thursday attack F1 at its selected "
                "threshold is chosen. Friday is not used for model or threshold selection."
            ),
            "note": (
                "Friday results estimate performance on a later capture day from this "
                "controlled benchmark. They do not establish performance on live or "
                "unrelated network traffic. A new unlabeled dataset cannot identify "
                "which model is more accurate without reference labels."
            ),
        },
        "selected_model_by_validation": selected_name,
        "models": results,
    }
    return report


def main() -> dict[str, Any]:
    report = compare_models()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = REPORT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(REPORT_PATH)
    print(f"Selected by Thursday validation: {report['selected_model_by_validation']}")
    print(f"Saved comparison to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
