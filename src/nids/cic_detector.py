"""Default CIC-IDS2017 detector used for one-file intrusion scans.

The candidate comparison is performed separately by :mod:`compare_cic`.
This module takes its validation-selected candidate, fits it on the fixed
training days, and exposes safe same-schema CSV scanning for the application.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import compare_cic, profiles

MODEL_PATH = compare_cic.PROJECT_ROOT / "models" / "cic_ids2017_detector.joblib"
METADATA_PATH = compare_cic.PROJECT_ROOT / "reports" / "cic_ids2017_detector.json"
DATASET_ALERT_RATE = 0.01


def _comparison_report() -> dict[str, Any]:
    if not compare_cic.REPORT_PATH.is_file():
        raise FileNotFoundError("CIC comparison report was not found; run the comparison first")
    report = json.loads(compare_cic.REPORT_PATH.read_text(encoding="utf-8"))
    selected = report.get("selected_model_by_validation")
    if selected not in report.get("models", {}):
        raise ValueError("CIC comparison report has no valid selected model")
    return report


def build_default_detector() -> dict[str, Any]:
    """Fit the validation-selected model on Monday--Wednesday CIC data."""
    report = _comparison_report()
    selected = report["selected_model_by_validation"]
    paths = compare_cic._input_paths()
    _, feature_columns = compare_cic._schema(paths)
    frames, targets = [], []
    for day in compare_cic.TRAIN_DAYS:
        X, y, _ = compare_cic._load_day(paths[day], feature_columns)
        frames.append(X)
        targets.append(y)
    X_train = pd.concat(frames, ignore_index=True)
    y_train = pd.concat(targets, ignore_index=True).to_numpy()
    positive_weight = float(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1))
    model = compare_cic._candidate_models(feature_columns, positive_weight)[selected]
    model.fit(X_train, y_train)
    threshold = float(report["models"][selected]["validation"]["threshold"])
    bundle = {
        "model": model,
        "model_name": selected,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "training_days": list(compare_cic.TRAIN_DAYS),
        "validation_day": compare_cic.VALIDATION_DAY,
        "comparison_generated_at_utc": report.get("generated_at_utc"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    metadata = {
        "detector": "cic_ids2017_default_flow_detector",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": selected,
        "threshold": threshold,
        "training_days": list(compare_cic.TRAIN_DAYS),
        "validation_day": compare_cic.VALIDATION_DAY,
        "selected_from": str(compare_cic.REPORT_PATH.relative_to(compare_cic.PROJECT_ROOT)),
        "validation_metrics": report["models"][selected]["validation"],
        "held_out_test_metrics": report["models"][selected]["test"],
        "model_artifact": str(MODEL_PATH.relative_to(compare_cic.PROJECT_ROOT)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_default_detector() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("default CIC detector is not ready")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def scan_csv(csv_path: Path, output_path: Path) -> dict[str, Any]:
    """Scan one uploaded CIC-schema file. Labels are optional."""
    bundle, metadata = load_default_detector()
    X, y, labels = profiles._load_cic_csv(csv_path, require_labels=False)
    if list(X.columns) != bundle["feature_columns"]:
        raise ValueError("CSV feature schema does not match the default CIC detector")
    scores = bundle["model"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]
    predicted = scores >= threshold
    output = pd.DataFrame({
        "record": np.arange(1, len(X) + 1),
        "prediction": np.where(predicted, "attack", "benign"),
        "anomaly_score": scores,
        "threshold": threshold,
    })
    if labels is not None:
        output.insert(1, "source_label", labels.reset_index(drop=True))
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    context_columns = [
        column for column in ("Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp")
        if column in header
    ]
    if context_columns:
        context = pd.read_csv(csv_path, usecols=context_columns, dtype="string", low_memory=False)
        output = pd.concat([context.reset_index(drop=True), output], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    # Keep only a small, ranked preview in the response. Full results remain
    # downloadable; the preview lets the browser immediately show what an
    # analyst should inspect first and request a SHAP explanation for it.
    top_indexes = np.argsort(scores)[::-1][: min(10, len(scores))]
    top_alerts = []
    for index in top_indexes:
        feature_values = {
            column: (
                value.item() if isinstance(value, np.generic) else value
            )
            for column, value in X.iloc[int(index)].items()
        }
        alert: dict[str, Any] = {
            "record": int(index) + 1,
            "prediction": "attack" if bool(predicted[index]) else "benign",
            "anomaly_score": float(scores[index]),
            "features": feature_values,
        }
        if labels is not None:
            alert["source_label"] = str(labels.iloc[int(index)])
        if context_columns:
            for column, value in context.iloc[int(index)].items():
                alert[column] = None if pd.isna(value) else str(value)
        top_alerts.append(alert)
    flagged_rate = float(predicted.mean()) if len(predicted) else 0.0
    result: dict[str, Any] = {
        "detector": metadata["detector"],
        "model": bundle["model_name"],
        "threshold": threshold,
        "records_scanned": int(len(X)),
        "attacks_flagged": int(predicted.sum()),
        "normal_records": int(len(X) - predicted.sum()),
        "flagged_rate": flagged_rate,
        "dataset_assessment": (
            "suspicious_activity_detected"
            if flagged_rate >= DATASET_ALERT_RATE
            else "no_significant_suspicious_activity"
        ),
        "dataset_assessment_policy": (
            f"At least {DATASET_ALERT_RATE:.0%} of flows must cross the review threshold "
            "for a dataset-level suspicious-activity conclusion. Individual flagged "
            "flows remain available for review below that rate."
        ),
        "results_file": str(output_path),
        "top_alerts": top_alerts,
    }
    if y is not None:
        if y.nunique() == 2:
            result["labeled_evaluation"] = {
                **compare_cic._metrics(y.to_numpy(), scores, threshold),
                "attack_metrics_available": True,
            }
        else:
            # A clean baseline has no true attacks, so attack precision and
            # recall are undefined. Accuracy and false-alarm count remain
            # useful and are returned without inventing a recall value.
            result["labeled_evaluation"] = {
                "accuracy": float(np.mean(predicted == y.to_numpy())),
                "attack_metrics_available": False,
                "label_note": "Only one class is present in the supplied labels; attack precision and recall are not applicable.",
                "counts": {"false_alarms": int(predicted.sum())},
            }
    return result


def explain_record(features: dict[str, Any]) -> dict[str, Any]:
    """Explain one flow under the default detector with SHAP."""
    import shap

    bundle, metadata = load_default_detector()
    columns = bundle["feature_columns"]
    missing = [column for column in columns if column not in features]
    extra = [column for column in features if column not in columns]
    if missing or extra:
        raise ValueError("feature fields do not match the CIC detector")
    X = pd.DataFrame([{name: features[name] for name in columns}], columns=columns)
    pipeline = bundle["model"]
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())
    model_name = bundle["model_name"]
    if model_name == "random_forest":
        values = np.asarray(shap.TreeExplainer(classifier).shap_values(transformed))
        values = values[0, :, 1] if values.ndim == 3 else values.reshape(-1)
        scale = "attack probability"
    elif model_name == "xgboost":
        values = np.asarray(shap.TreeExplainer(classifier).shap_values(transformed)).reshape(-1)
        scale = "log-odds"
    else:
        values = np.asarray(shap.LinearExplainer(classifier, transformed).shap_values(transformed)).reshape(-1)
        scale = "log-odds"
    grouped: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        source = name.removeprefix("numeric__").removeprefix("categorical__")
        for categorical in compare_cic.CATEGORICAL_COLUMNS:
            if source.startswith(f"{categorical}_"):
                source = categorical
        grouped[source] = grouped.get(source, 0.0) + float(value)
    score = float(pipeline.predict_proba(X)[0, 1])
    evidence = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    return {
        "detector": metadata["detector"], "model": model_name,
        "prediction": "attack" if score >= bundle["threshold"] else "benign",
        "anomaly_score": score, "threshold": bundle["threshold"],
        "explanation_scale": scale,
        "evidence": [{"feature": name, "value": features[name], "shap_value": value,
                      "direction": "raises attack score" if value > 0 else "lowers attack score"}
                     for name, value in evidence],
    }


def main() -> dict[str, Any]:
    metadata = build_default_detector()
    print(f"Built {metadata['model']} CIC detector at {MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    main()
