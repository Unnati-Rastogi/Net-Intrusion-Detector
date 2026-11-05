"""Dataset-specific CIC flow model profiles.

A profile trains all candidate models on one labeled upload, chooses a
candidate and threshold from an internal validation split, and retains every
candidate so later labeled scenarios can be compared without retraining.
Unlabeled uploads can be scanned with the validation-selected candidate.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from . import compare_cic

PROFILE_DIR = compare_cic.PROJECT_ROOT / "models" / "profiles"
REPORT_DIR = compare_cic.PROJECT_ROOT / "reports" / "profiles"
SCHEMA_PATH = Path(__file__).with_name("cic_ids2017_schema.json")
CIC_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
BENIGN_LABELS = {"benign", "normal", "0", "false", "non-attack", "non_attack"}


def _profile_paths(profile_id: str) -> tuple[Path, Path]:
    if not profile_id or any(ch not in "0123456789abcdef-" for ch in profile_id.lower()):
        raise ValueError("invalid profile id")
    return PROFILE_DIR / f"{profile_id}.joblib", REPORT_DIR / f"{profile_id}.json"


def _load_cic_csv(path: Path, *, require_labels: bool) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    header = list(pd.read_csv(path, nrows=0).columns)
    if len(header) != len(set(header)):
        raise ValueError("CSV contains duplicate column names")
    expected = CIC_SCHEMA["feature_columns"]
    missing = [name for name in expected if name not in header]
    if missing:
        raise ValueError("CSV does not match the supported CIC flow schema; missing fields: " + ", ".join(missing))
    if require_labels and compare_cic.LABEL_COLUMN not in header:
        raise ValueError(f"labeled CSV must include a '{compare_cic.LABEL_COLUMN}' column")
    feature_columns = [
        name for name in header
        if name not in CIC_SCHEMA["excluded_columns"] and name != compare_cic.LABEL_COLUMN
    ]
    # Profiles share one stable feature contract so a saved model cannot be
    # accidentally applied to an unrelated or differently ordered schema.
    if feature_columns != expected:
        missing_features = [name for name in expected if name not in feature_columns]
        extras = [name for name in feature_columns if name not in expected]
        details = []
        if missing_features:
            details.append("missing: " + ", ".join(missing_features))
        if extras:
            details.append("unsupported: " + ", ".join(extras))
        if not details:
            details.append("feature columns are in a different order")
        raise ValueError("CSV does not match the CIC flow schema (" + "; ".join(details) + ")")

    selected = [*feature_columns]
    if compare_cic.LABEL_COLUMN in header:
        selected.append(compare_cic.LABEL_COLUMN)
    dtypes: dict[str, Any] = {name: np.float32 for name in feature_columns}
    dtypes.update({name: "string" for name in compare_cic.CATEGORICAL_COLUMNS})
    if compare_cic.LABEL_COLUMN in selected:
        dtypes[compare_cic.LABEL_COLUMN] = "string"
    frame = pd.read_csv(path, usecols=selected, dtype=dtypes, low_memory=False)
    frame = frame.loc[:, selected]
    if compare_cic.LABEL_COLUMN not in frame:
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame, None, None

    labels = frame.pop(compare_cic.LABEL_COLUMN).str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("CSV contains missing labels")
    normalized = labels.str.casefold().str.strip()
    benign = normalized.isin(BENIGN_LABELS) | normalized.str.endswith(" - attempted")
    y = (~benign).astype(np.uint8)
    if require_labels and y.nunique() != 2:
        raise ValueError("labeled CSV must contain both benign and attack records")
    numeric = [name for name in feature_columns if name not in compare_cic.CATEGORICAL_COLUMNS]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame, y, labels


def _profile_attack_family_metrics(
    labels: pd.Series, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, int | float]]:
    normalized = labels.str.casefold().str.strip()
    attack_mask = ~normalized.isin(BENIGN_LABELS) & ~normalized.str.endswith(" - attempted")
    predicted = scores >= threshold
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


def train_profile(csv_path: Path, name: str) -> dict[str, Any]:
    """Fit all models on one labeled upload and save the validation winner."""
    X, y, labels = _load_cic_csv(csv_path, require_labels=True)
    assert y is not None and labels is not None
    if len(y) < 100:
        raise ValueError("upload at least 100 labeled flows to create a model profile")
    indexes = np.arange(len(y))
    train_idx, validation_idx = train_test_split(
        indexes,
        test_size=0.2,
        random_state=compare_cic.RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
    y_train, y_validation = y.iloc[train_idx].to_numpy(), y.iloc[validation_idx].to_numpy()
    pos_weight = float(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1))
    models = compare_cic._candidate_models(list(X.columns), pos_weight)

    results: dict[str, Any] = {}
    for model_name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        scores = model.predict_proba(X_validation)[:, 1]
        threshold, metrics, policy = compare_cic._select_threshold(y_validation, scores)
        results[model_name] = {
            "validation": metrics,
            "threshold_policy": policy,
            "fit_seconds": round(fit_seconds, 3),
        }

    selected = max(
        results,
        key=lambda key: (
            results[key]["validation"]["f1_attack"],
            results[key]["validation"]["recall_attack"],
            results[key]["validation"]["precision_attack"],
        ),
    )
    profile_id = uuid.uuid4().hex
    model_path, report_path = _profile_paths(profile_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Save a small representative background sample for later SHAP reports.
    background = X_train.sample(n=min(100, len(X_train)), random_state=compare_cic.RANDOM_STATE)
    bundle = {
        "profile_id": profile_id,
        "name": name.strip()[:80] or "Uploaded CIC dataset",
        "feature_columns": list(X.columns),
        "models": models,
        "thresholds": {
            key: float(results[key]["validation"]["threshold"]) for key in results
        },
        "selected_model": selected,
        "background": background,
    }
    joblib.dump(bundle, model_path, compress=3)
    report = {
        "profile_id": profile_id,
        "name": bundle["name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": compare_cic.sha256_file(csv_path),
        "schema": "corrected CIC-IDS2017 flow features",
        "training_records": int(len(train_idx)),
        "validation_records": int(len(validation_idx)),
        "source_records": int(len(y)),
        "source_label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "positive_class": "attack",
        "split": "stratified 80/20 row split; use a separate scenario upload for cross-scenario evaluation",
        "selected_model": selected,
        "selection_rule": "highest validation attack F1 after selecting each threshold on validation only",
        "models": results,
        "model_artifact": str(model_path.relative_to(compare_cic.PROJECT_ROOT)),
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def load_profile(profile_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path, report_path = _profile_paths(profile_id)
    if not model_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("dataset profile was not found")
    return joblib.load(model_path), json.loads(report_path.read_text(encoding="utf-8"))


def evaluate_profile(profile_id: str, csv_path: Path) -> dict[str, Any]:
    """Compare a saved profile's candidates on a separate labeled scenario."""
    bundle, metadata = load_profile(profile_id)
    X, y, labels = _load_cic_csv(csv_path, require_labels=True)
    assert y is not None and labels is not None
    if list(X.columns) != bundle["feature_columns"]:
        raise ValueError("evaluation CSV feature schema does not match the trained profile")
    results: dict[str, Any] = {}
    for name, model in bundle["models"].items():
        scores = model.predict_proba(X)[:, 1]
        threshold = bundle["thresholds"][name]
        metrics = compare_cic._metrics(y.to_numpy(), scores, threshold)
        family = _profile_attack_family_metrics(labels, scores, threshold)
        results[name] = {"metrics": metrics, "attack_family_recall": family}
    return {
        "profile_id": profile_id,
        "profile_name": metadata["name"],
        "evaluation_records": int(len(y)),
        "evaluation_label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "models": results,
        "note": "This is a separate labeled scenario comparison. If you choose a model based on these results, do not present this same scenario as an untouched final test.",
    }


def predict_profile(profile_id: str, csv_path: Path, output_path: Path) -> dict[str, Any]:
    """Scan a same-schema unlabeled upload with the validation-selected model."""
    bundle, metadata = load_profile(profile_id)
    X, _, _ = _load_cic_csv(csv_path, require_labels=False)
    model_name = bundle["selected_model"]
    model = bundle["models"][model_name]
    scores = model.predict_proba(X)[:, 1]
    threshold = bundle["thresholds"][model_name]
    predictions = scores >= threshold
    output = pd.DataFrame(
        {
            "record": np.arange(1, len(X) + 1),
            "prediction": np.where(predictions, "attack", "benign"),
            "anomaly_score": scores,
            "threshold": threshold,
        }
    )
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    context_columns = [
        column for column in ("Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp")
        if column in header
    ]
    if context_columns:
        context = pd.read_csv(csv_path, usecols=context_columns, dtype="string", low_memory=False)
        output = pd.concat([context.reset_index(drop=True), output], axis=1)
    output.to_csv(output_path, index=False)
    return {
        "profile_id": profile_id,
        "profile_name": metadata["name"],
        "model": model_name,
        "threshold": threshold,
        "records_scanned": int(len(X)),
        "attacks_flagged": int(predictions.sum()),
        "normal_records": int(len(X) - predictions.sum()),
        "results_file": str(output_path),
    }


def explain_profile(profile_id: str, features: dict[str, Any]) -> dict[str, Any]:
    """Return SHAP contributions for one record under the selected profile."""
    import shap

    bundle, metadata = load_profile(profile_id)
    columns = bundle["feature_columns"]
    missing = [column for column in columns if column not in features]
    extra = [column for column in features if column not in columns]
    if missing or extra:
        raise ValueError(
            "feature fields do not match this profile; "
            + ("missing: " + ", ".join(missing) if missing else "")
            + ("; " if missing and extra else "")
            + ("unsupported: " + ", ".join(extra) if extra else "")
        )
    X = pd.DataFrame([{key: features[key] for key in columns}], columns=columns)
    model_name = bundle["selected_model"]
    pipeline = bundle["models"][model_name]
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    background = bundle["background"]
    transformed_background = preprocessor.transform(background)
    transformed_record = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    if model_name == "random_forest":
        explainer = shap.TreeExplainer(
            classifier,
            data=transformed_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        shap_values = explainer.shap_values(transformed_record)
        explanation_scale = "attack probability"
    elif model_name == "xgboost":
        # SHAP's interventional XGBoost path currently rejects categorical
        # splits in some supported XGBoost releases. Tree-path-dependent mode
        # works with those models and explains the raw margin (log-odds).
        explainer = shap.TreeExplainer(
            classifier,
            feature_perturbation="tree_path_dependent",
        )
        shap_values = explainer.shap_values(transformed_record)
        explanation_scale = "log-odds"
    else:
        explainer = shap.LinearExplainer(classifier, transformed_background)
        shap_values = explainer.shap_values(transformed_record)
        explanation_scale = "log-odds"

    if model_name in {"random_forest", "xgboost"}:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, 1]
        else:
            values = values.reshape(-1)
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).reshape(-1)[-1]
        base_value = float(base)
    else:
        values = np.asarray(shap_values).reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])

    contributions: dict[str, float] = {}
    for feature_name, value in zip(feature_names, values, strict=True):
        if feature_name.startswith("numeric__"):
            source_name = feature_name.removeprefix("numeric__")
        else:
            encoded_name = feature_name.removeprefix("categorical__")
            source_name = next(
                (
                    column for column in compare_cic.CATEGORICAL_COLUMNS
                    if encoded_name == column or encoded_name.startswith(f"{column}_")
                ),
                encoded_name,
            )
        contributions[source_name] = contributions.get(source_name, 0.0) + float(value)
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    score = float(pipeline.predict_proba(X)[0, 1])
    threshold = bundle["thresholds"][model_name]
    return {
        "profile_id": profile_id,
        "profile_name": metadata["name"],
        "model": model_name,
        "prediction": "attack" if score >= threshold else "benign",
        "anomaly_score": score,
        "threshold": threshold,
        "base_value": base_value,
        "explanation_scale": explanation_scale,
        "evidence": [
            {
                "feature": name,
                "value": features.get(name),
                "shap_value": contribution,
                "direction": "raises attack score" if contribution > 0 else "lowers attack score",
            }
            for name, contribution in ranked[:10]
        ],
        "interpretation_note": "SHAP describes feature contributions to this model's output; it does not establish the cause of the network activity.",
    }
