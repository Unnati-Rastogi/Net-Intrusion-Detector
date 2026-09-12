"""Prediction for the NIDS model (single sample and batch).

Loads the saved pipeline and exposes predict_connection() /
predict_connections() for raw connection records (the 41 dataset feature
columns as dicts, no 'class'). Later API/dashboard layers should call this
module rather than touching sklearn directly.
"""

from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config


class ModelNotTrainedError(RuntimeError):
    """Raised when prediction is requested before the model artifact exists."""


def load_model(path=config.MODEL_PATH):
    """Load the fitted pipeline; raise a clear error if it is missing."""
    if not path.exists():
        raise ModelNotTrainedError(
            f"model artifact not found at {path}. "
            "Run 'python -m src.nids.train' first."
        )
    return joblib.load(path)


def validate_sample(sample: dict[str, Any], expected_columns: list[str]) -> None:
    """Ensure a raw sample has exactly the expected feature columns.

    Raises ValueError listing the missing and/or unexpected keys.
    """
    missing = [c for c in expected_columns if c not in sample]
    extra = [k for k in sample if k not in expected_columns]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing feature(s): {missing}")
        if extra:
            parts.append(f"unexpected feature(s): {sorted(extra)}")
        raise ValueError("; ".join(parts))


def _class_probabilities(model, proba_row: np.ndarray) -> tuple[float, float]:
    """Map one predict_proba row onto (p_normal, p_anomaly) via model.classes_."""
    classes = np.asarray(model.classes_)
    p_normal = float(proba_row[int(np.where(classes == 0)[0][0])])
    p_anomaly = float(proba_row[int(np.where(classes == 1)[0][0])])
    return p_normal, p_anomaly


def _format_result(p_normal: float, p_anomaly: float) -> dict[str, Any]:
    is_anomaly = p_anomaly >= 0.5
    return {
        "prediction": "anomaly" if is_anomaly else "normal",
        "label": int(is_anomaly),
        "probability_normal": p_normal,
        "probability_anomaly": p_anomaly,
        "confidence": p_anomaly if is_anomaly else p_normal,
    }


def predict_connection(sample: dict[str, Any], model=None) -> dict[str, Any]:
    """Predict normal/anomaly for one network-flow record.

    `sample` must contain the 41 raw feature columns (the dataset columns
    minus 'class'; the two constant columns are dropped inside the pipeline).

    Returns a dict with:
        prediction           : 'normal' or 'anomaly'
        label                : 0 = normal, 1 = anomaly
        probability_normal   : model probability for 'normal'
        probability_anomaly  : model probability for 'anomaly'
        confidence           : probability of the predicted class
    """
    if model is None:
        model = load_model()

    expected_columns = list(model.feature_names_in_)
    validate_sample(sample, expected_columns)

    row = pd.DataFrame([sample], columns=expected_columns)
    p_normal, p_anomaly = _class_probabilities(model, model.predict_proba(row)[0])
    return _format_result(p_normal, p_anomaly)


def predict_connections(
    samples: list[dict[str, Any]], model=None
) -> list[dict[str, Any]]:
    """Predict for many records with one vectorized predict_proba pass.

    Every sample is validated first; a ValueError names the offending index.
    """
    if model is None:
        model = load_model()

    expected_columns = list(model.feature_names_in_)
    for i, sample in enumerate(samples):
        try:
            validate_sample(sample, expected_columns)
        except ValueError as exc:
            raise ValueError(f"sample at index {i}: {exc}") from exc

    frame = pd.DataFrame(list(samples), columns=expected_columns)
    proba = model.predict_proba(frame)
    classes = np.asarray(model.classes_)
    i_normal = int(np.where(classes == 0)[0][0])
    i_anomaly = int(np.where(classes == 1)[0][0])
    return [
        _format_result(float(row[i_normal]), float(row[i_anomaly]))
        for row in proba
    ]
