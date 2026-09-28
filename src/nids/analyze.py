"""Analyst analysis orchestrator for the API layer (Stage 2 integration).

This module adds NO new ML functionality. It composes the existing offline
domain layers around one network-flow record:

    prediction      -> predict.predict_connection      (saved RF pipeline)
    explanation     -> explain.explain_samples          (TreeExplainer, once)
    risk            -> risk.calculate_risk_for_explanation (consumes that
                       explanation verbatim; never recomputed separately)
    priority        -> prioritize.PRIORITY_MAP[risk_level] (the domain layer's
                       own deterministic mapping; P1 = highest review
                       priority, P4 = review backlog for unflagged records)

Terminology contract (enforced here and in the schemas):
    anomaly_score  is the model's P(anomaly) output -- a model score, NOT an
                   attack probability.
    risk_score     is the model-derived prioritization score -- NOT attack
                   severity, probability of compromise, or impact.
    priority       is the analyst review priority -- NOT attack danger.

Lineage: `sample_id` / `source_row_id` come from the committed Stage-1
reports (reports/risk_assessment.json), whose records carry both fields
explicitly. They are looked up -- never re-derived by re-running the
sampler and never fabricated. An ad-hoc feature vector gets null ids
unless it exactly and uniquely matches one stored report sample (matched
against that sample's actual source row in Train_data.csv).

Performance: the explanation context (saved model + SHAP background) and
the report-lineage snapshot are built lazily once per process and reused;
SHAP runs exactly once per /analyze request because the user explicitly
requested analyst-level analysis. /predict stays SHAP-free.
"""

import json
import threading
from typing import Any

import pandas as pd

from . import config, explain, prioritize, risk
from .data import load_training_data
from .predict import load_model, predict_connection

RISK_REPORT_PATH = explain.REPORTS_DIR / "risk_assessment.json"

METHODOLOGY_NOTE = {
    "anomaly_score": (
        "the model's P(anomaly) output for this record (a model score, "
        "not an attack probability)"
    ),
    "risk_score": (
        "model-derived prioritization score from src/nids/risk.py "
        "(not attack severity, probability of compromise, or impact)"
    ),
    "priority": (
        "analyst review priority from the deterministic P1-P4 mapping "
        "(P1 = highest analyst review priority, P4 = review backlog)"
    ),
    "explanation": (
        "SHAP feature attributions of the model's P(anomaly) output "
        "(model attribution, not causality)"
    ),
}


# --- Lazily shared, per-process singletons (built once, reused) --------------

_context_lock = threading.Lock()
_context: explain.ExplanationContext | None = None

_lineage_lock = threading.Lock()
_lineage: dict[str, Any] | None = None


def get_explanation_context() -> explain.ExplanationContext:
    """The explanation context for the saved production pipeline (cached)."""
    global _context
    if _context is None:
        with _context_lock:
            if _context is None:
                _context = explain.build_context()
    return _context


def get_report_lineage() -> dict[int, dict[str, Any]]:
    """sample_id -> {source_row_id, features} from the committed risk report.

    Reads the Stage-1 report (which persists both ids explicitly) and joins
    each record to its actual source row in Train_data.csv via
    source_row_id. Used to (a) resolve /analyze?sample_id=... requests and
    (b) recognize ad-hoc vectors that exactly match a stored report sample.
    The sampler is never re-run and no ids are invented.
    """
    global _lineage
    if _lineage is None:
        with _lineage_lock:
            if _lineage is None:
                report = json.loads(RISK_REPORT_PATH.read_text(encoding="utf-8"))
                X, _y = load_training_data(config.TRAIN_CSV)
                table: dict[int, dict[str, Any]] = {}
                for record in report["samples"]:
                    row = int(record["source_row_id"])
                    table[int(record["sample_id"])] = {
                        "source_row_id": row,
                        "features": {
                            # JSON-safe scalars (pandas/numpy values are not
                            # directly JSON-serializable in responses).
                            column: explain._json_value(X.iloc[row][column])
                            for column in config.FEATURE_COLUMNS
                        },
                    }
                _lineage = table
    return _lineage


def reset_caches() -> None:
    """Drop the cached context/lineage (used by tests)."""
    global _context, _lineage
    with _context_lock:
        _context = None
    with _lineage_lock:
        _lineage = None


# --- Core orchestration -------------------------------------------------------

def _match_report_sample(
    features: dict[str, Any], lineage: dict[int, dict[str, Any]]
) -> int | None:
    """The sample_id whose stored source row equals `features`, uniquely.

    Exact comparison only (no tolerance): an ad-hoc vector is recognized as
    a known report sample only when every feature value equals that sample's
    actual source row. Ambiguous matches are treated as no match so ids are
    never attached on weak evidence.
    """
    matches = []
    for sample_id, entry in lineage.items():
        # Compare over the stored sample's columns (the full feature set in
        # production; partial keys are supported so tests can use mini rows).
        for column, stored_value in entry["features"].items():
            value = features[column]
            if isinstance(stored_value, str):
                if str(value) != stored_value:
                    break
            elif float(value) != float(stored_value):
                break
        else:
            matches.append(sample_id)
    return matches[0] if len(matches) == 1 else None


def analyze_record(
    features: dict[str, Any],
    model=None,
    context: explain.ExplanationContext | None = None,
) -> dict[str, Any]:
    """Run prediction -> explanation -> risk -> priority for ONE record.

    Pure orchestration of existing domain functions; returns a plain dict
    matching the AnalyzeResponse schema. `source_row_id`/`sample_id` are
    filled only when the vector exactly matches one stored report sample;
    otherwise they stay None (never fabricated).
    """
    if model is None:
        model = load_model()
    if context is None:
        context = get_explanation_context()

    # 1. Detection (existing prediction path, identical to POST /predict).
    prediction = predict_connection(features, model=model)

    # 2. Explanation: exactly ONE explain_samples() call (SHAP once).
    frame = pd.DataFrame([features], columns=list(model.feature_names_in_))
    explanation_row = explain.explain_samples(context, frame, top_k=None)[0]

    # 3. Risk: consumes the SAME explanation record; nothing recomputed.
    risk_result = risk.calculate_risk_for_explanation(
        explanation_row, n_estimators=context.classifier.n_estimators
    )

    # 4. Priority: the domain layer's own deterministic mapping. Every risk
    # level maps to a review priority (LOW -> P4, the review backlog), so
    # priority is always defined for the analyst workflow.
    priority = prioritize.PRIORITY_MAP[risk_result["risk_level"]]

    # 5. Lineage: only from the stored report sample, never fabricated.
    sample_id = _match_report_sample(features, get_report_lineage())
    source_row_id = (
        get_report_lineage()[sample_id]["source_row_id"]
        if sample_id is not None
        else None
    )

    return {
        "prediction": prediction["prediction"],
        "label": prediction["label"],
        "anomaly_score": prediction["probability_anomaly"],
        "probability_normal": prediction["probability_normal"],
        "probability_anomaly": prediction["probability_anomaly"],
        "confidence": prediction["confidence"],
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "priority": priority,
        "sample_id": sample_id,
        "source_row_id": source_row_id,
        "explanation": [
            {
                "feature": contribution["feature"],
                "value": contribution["value"],
                "shap_value": contribution["shap_contribution"],
                "direction": contribution["direction"],
                "abs_contribution": contribution["abs_contribution"],
            }
            for contribution in explanation_row["top_features"]
        ],
        "methodology": dict(METHODOLOGY_NOTE),
    }


def analyze_report_sample(sample_id: int) -> dict[str, Any]:
    """Analyze a stored report sample by its sample_id (lineage preserved)."""
    lineage = get_report_lineage()
    if sample_id not in lineage:
        raise KeyError(sample_id)
    entry = lineage[sample_id]
    result = analyze_record(
        {column: entry["features"][column] for column in config.FEATURE_COLUMNS}
    )
    # The ids are known upfront for this mode; attach them authoritatively.
    result["sample_id"] = sample_id
    result["source_row_id"] = entry["source_row_id"]
    return result
