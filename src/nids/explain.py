"""SHAP explainability side-layer for the NIDS (Random Forest only).

Explains the SAVED production pipeline (models/rf_ids_pipeline.joblib) via
SHAP's TreeExplainer. This module never trains, retrains, or modifies the
model; it is a read-only side report and is NOT part of the API prediction
path.

Architecture (matches the fitted pipeline, no retraining):

    saved RF Pipeline (load_model)
        -> preprocessing (ColumnTransformer: const-drop + one-hot + passthrough)
        -> transformed 116-feature representation Z
        -> RandomForest classifier (explained, untouched)
        -> shap.TreeExplainer(clf, data=background_Z, feature_perturbation=
                              "interventional")
        -> SHAP values (n_samples, 116, 2)   [features x classes]
        -> roll one-hot groups back together by SUMMING members
        -> per-original-feature attributions (41 original feature names)

What is explained, precisely:
    - Output: the RF classifier's P(anomaly) for class index 1
      (model.classes_ = [0, 1]; verified empirically because the class-1 SHAP
      values reconstruct predict_proba[:, 1] by additivity to <1e-6).
    - Values: SHAP feature attributions describe how much each feature's value
      moves THIS model's output for THIS sample away from the baseline. They
      are model attributions, NOT causal explanations of network behaviour.
    - Baseline (base value): the model's mean P(anomaly) over the background
      sample -- for the interventional strategy, explainer.expected_value[1]
      equals mean(predict_proba(background)[:, 1]).
    - Perturbation strategy: "interventional" with a fixed background sample
      (100 training rows, numpy default_rng(42), no replacement, sorted).
      Rationale: the baseline becomes an interpretable average anomaly rate
      over realistic records; feature attributions are computed against that
      explicit reference distribution, which is stable and easy to defend in
      review. The alternative "tree_path_dependent" uses the training node
      statistics (baseline ~= the structural 0.5 prior) and is faster, but its
      baseline is harder to reason about for a security audience.

Usage (from the project root):
    python -m src.nids.explain      # writes reports/explanations.json

    Programmatic: build_context(...) once, then explain_samples(context, X).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier

from . import config, evaluate
from .data import load_training_data
from .predict import load_model

# --- Output locations / identity --------------------------------------------
REPORTS_DIR = evaluate.REPORTS_DIR
EXPLANATIONS_JSON_PATH = REPORTS_DIR / "explanations.json"
SCHEMA_VERSION = "1.0"
REPORT_NAME = "nids_shap_explanations"

# Class index for 'anomaly' in the SHAP value array's class axis.
# The binary RF orders classes as [0, 1]; index 1 is verified at explanation
# time by the additivity check against predict_proba[:, 1].
ANOMALY_CLASS_INDEX = 1

# Deterministic background for the interventional explainer.
BACKGROUND_SIZE = 100
BACKGROUND_SEED = config.RANDOM_STATE

# Deterministic sample size for the global report (full data would take
# roughly 50+ minutes at ~120 ms/sample; see README for the trade-off).
REPORT_SAMPLE_SIZE = 1000

TOP_K_DEFAULT = 10

ADDITIVITY_TOLERANCE = 1e-6

ADDITIVITY_NOTE = (
    "A handful of records can carry a feature value exactly equal to a "
    "trained split threshold; SHAP's tree traversal and sklearn resolve "
    "that tie on opposite sides, so the SHAP reconstruction differs from "
    "predict_proba by exactly one tree vote (1/n_estimators). Only this "
    "exact signature is tolerated, and each affected sample is flagged with "
    "additivity_note 'one_vote_split_threshold_tie'. Any larger mismatch "
    "is a hard error."
)

METHODOLOGY_NOTE = (
    "Global importance uses TreeExplainer SHAP values computed with the "
    "interventional feature-perturbation strategy against a fixed, "
    "deterministic background of 100 labelled training rows. Per sample and "
    "per original feature, the SHAP values of that feature's transformed "
    "columns (one-hot groups summed) are aggregated into one net "
    "contribution; global importance is the mean of the absolute net "
    "contributions over the report sample. SHAP values are feature-"
    "attribution values describing the model's output: how much each feature "
    "moves this model's P(anomaly) for a given record away from the baseline "
    "(the model's mean P(anomaly) over the background). They are NOT causal "
    "explanations of network behaviour."
)

LIMITATIONS = (
    [
        "SHAP attributions explain the MODEL's output, not the underlying "
        "network phenomenon: they are not causal explanations.",
        "Metrics and attributions come from the labelled training "
        "distribution only (Train_data.csv); the unlabelled Test_data.csv is "
        "never scored or explained.",
        "The interventional baseline depends on the 100-row background "
        "sample; a different background changes individual attributions "
        "(the baseline is the model's mean P(anomaly) over that sample).",
        "TreeExplainer attributions are exact for the tree ensemble given the "
        "background, but they describe a surrogate view of feature effects; "
        "correlated features split credit between them.",
        "Global importance is a mean over the report sample; rare attack "
        "patterns may be under-represented and their drivers under-weighted.",
        "Per-sample contributions are about the trained model as-is; they do "
        "not measure what would happen if the feature were different in the "
        "world (no intervention was performed on the network).",
    ]
)


# --- Transformed -> original feature mapping ---------------------------------
def transformed_feature_names(preprocessing) -> list[str]:
    """The 116 transformed column names, in pipeline output order."""
    return list(preprocessing.get_feature_names_out())


def build_feature_mapping(preprocessing) -> dict[str, list[int]]:
    """Deterministic map: original feature -> transformed column indices.

    A transformed column belongs to original feature F iff it IS F (numeric
    passthrough, names are preserved by verbose_feature_names_out=False) or is
    its one-hot column (named f"{F}_{category}"). The prefix match is anchored
    on the underscore-separated original name so no feature can claim another
    feature's column.

    The two constant columns (config.CONSTANT_COLUMNS) are DROPPED by the
    pipeline, so they map to zero transformed features; they are kept in the
    mapping explicitly (their SHAP contribution is exactly 0 -- the model
    never sees them). Any OTHER original feature owning zero columns is a
    mapping bug and raises. Validates that every transformed index is claimed
    by exactly one original feature.
    """
    transformed = transformed_feature_names(preprocessing)
    originals = list(preprocessing.feature_names_in_)
    mapping: dict[str, list[int]] = {name: [] for name in originals}

    for idx, col_name in enumerate(transformed):
        owners = [
            name
            for name in originals
            if col_name == name or col_name.startswith(f"{name}_")
        ]
        if len(owners) != 1:
            raise ValueError(
                f"transformed column {col_name!r} matched {owners!r}; "
                "expected exactly one original feature"
            )
        mapping[owners[0]].append(idx)

    claimed = sorted(i for idxs in mapping.values() for i in idxs)
    if claimed != list(range(len(transformed))):
        missing = sorted(set(range(len(transformed))) - set(claimed))
        raise ValueError(f"unmapped transformed indices: {missing}")

    unexpected_dropped = [
        name for name, idxs in mapping.items()
        if not idxs and name not in config.CONSTANT_COLUMNS
    ]
    if unexpected_dropped:
        raise ValueError(
            f"original features with no transformed column that are not "
            f"known constants: {unexpected_dropped}"
        )
    return mapping


def _mapping_block(preprocessing, mapping: dict[str, list[int]]) -> dict[str, Any]:
    """JSON-friendly mapping block: transformed names per original feature."""
    names = transformed_feature_names(preprocessing)
    dropped = sorted(name for name, idxs in mapping.items() if not idxs)
    return {
        "transformed_feature_count": len(names),
        "original_feature_count": len(mapping),
        "explained_feature_count": len(mapping) - len(dropped),
        "dropped_constant_features": dropped,
        "dropped_note": (
            "dropped inside the pipeline as constant-in-training (zero "
            "information); their SHAP contribution is exactly 0"
        ),
        "groups": {
            original: {
                "transformed_count": len(idxs),
                "transformed_features": [names[i] for i in idxs],
            }
            for original, idxs in mapping.items()
        },
    }


# --- Explainer context --------------------------------------------------------
@dataclass
class ExplanationContext:
    """Everything needed to explain the saved (or provided) RF pipeline."""

    pipeline: Any
    preprocessing: Any
    classifier: RandomForestClassifier
    mapping: dict[str, list[int]]
    explainer: Any
    background_indices: np.ndarray
    background_Z: np.ndarray
    model_hash: str | None
    shap_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


def background_sample_indices(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Deterministic background row positions (sorted, no replacement)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(size, n_rows), replace=False))


def build_context(
    model=None,
    training_data: tuple[pd.DataFrame, pd.Series] | None = None,
    background_size: int = BACKGROUND_SIZE,
    seed: int = BACKGROUND_SEED,
) -> ExplanationContext:
    """Assemble the explanation context WITHOUT retraining anything.

    `model` defaults to the saved production artifact (load_model). For tests,
    pass an already-fitted pipeline plus `training_data=(X, y)`; no fitting
    happens here either way. The background is drawn from the training rows
    only, transformed by the (already fitted) pipeline's preprocessing.
    """
    if model is None:
        model = load_model()
        model_hash = (
            evaluate.sha256_file(config.MODEL_PATH)
            if config.MODEL_PATH.exists()
            else None
        )
        source = "saved artifact (not retrained)"
    else:
        model_hash = None
        source = "provided fitted pipeline (tests)"

    if training_data is None:
        X, _y = load_training_data(config.TRAIN_CSV)
    else:
        X, _y = training_data

    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]
    mapping = build_feature_mapping(preprocessing)

    idx = background_sample_indices(len(X), background_size, seed)
    background_Z = preprocessing.transform(X.iloc[idx])
    explainer = shap.TreeExplainer(
        classifier, data=background_Z, feature_perturbation="interventional"
    )
    return ExplanationContext(
        pipeline=model,
        preprocessing=preprocessing,
        classifier=classifier,
        mapping=mapping,
        explainer=explainer,
        background_indices=idx,
        background_Z=background_Z,
        model_hash=model_hash,
        shap_version=shap.__version__,
        metadata={"model_source": source, "background_seed": seed},
    )


# --- Per-sample explanation ---------------------------------------------------
def verify_additivity(
    reconstructed: np.ndarray,
    scores: np.ndarray,
    n_estimators: int,
) -> tuple[np.ndarray, list[str | None]]:
    """Verify SHAP reconstructs P(anomaly); allow the exact one-vote tie.

    For each sample: base + sum(contributions) must equal the model's
    P(anomaly) within ADDITIVITY_TOLERANCE, OR differ by exactly one tree
    vote (1/n_estimators, the split-threshold-tie signature described in
    ADDITIVITY_NOTE; the band around one vote also absorbs the ~1e-8
    floating-point residue of summing the per-tree reconstructions). Returns
    (per-sample errors, per-sample notes) and raises on anything outside
    those two signatures — a two-vote mismatch is still a hard error.
    """
    errors = np.abs(reconstructed - scores)
    vote_quantum = 1.0 / n_estimators
    notes: list[str | None] = []
    for i, err in enumerate(errors):
        if err <= ADDITIVITY_TOLERANCE:
            notes.append(None)
        elif abs(float(err) - vote_quantum) <= ADDITIVITY_TOLERANCE:
            notes.append("one_vote_split_threshold_tie")
        else:
            raise ValueError(
                f"additivity check failed for sample {i} (error "
                f"{err:.3e} exceeds both the exact tolerance and the "
                "one-vote split-threshold-tie signature)"
            )
    return errors, notes


def explain_samples(
    context: ExplanationContext,
    samples: pd.DataFrame,
    top_k: int | None = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Explain raw (41-column) samples with the prepared context.

    Returns one dict per sample (original row order):
        sample_id          position of the record within `samples` (0-based);
                           preserved meaning: the report sample position
        source_row_id      zero-based row position of the record in the
                           SOURCE dataframe it was drawn from (not the
                           position within `samples`) — explicit data
                           lineage, carried through risk/prioritization
        prediction         'anomaly' | 'normal' (threshold 0.5, as in predict.py)
        anomaly_score      model P(anomaly) for the sample
        base_value         baseline P(anomaly) = background mean (class 1)
        top_features       contributions sorted by absolute magnitude:
                           original feature name, raw feature value, SHAP net
                           contribution, direction, absolute contribution
        additivity_error   |base + sum(contributions) - anomaly_score|

    top_k=None returns all 41 original features per sample. The raw
    116-dimensional SHAP array is never returned.
    """
    if len(samples) == 0:
        return []

    Z = context.preprocessing.transform(samples)
    # shap's internal additivity check compares against sklearn semantics
    # that break on split-threshold ties; the meaningful verification is the
    # explicit one below (against predict_proba, one-vote allowance).
    values = np.asarray(
        context.explainer.shap_values(Z, check_additivity=False)
    )
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(
            f"unexpected SHAP output shape {values.shape}; "
            "expected (n_samples, n_transformed, 2) for the binary classifier"
        )
    if not np.isfinite(values).all():
        raise ValueError("SHAP produced non-finite values")

    a = ANOMALY_CLASS_INDEX
    ev = np.asarray(context.explainer.expected_value, dtype=float)
    anomaly_values = values[:, :, a]                     # (n, 116)
    scores = context.classifier.predict_proba(Z)[:, 1]   # P(anomaly)

    # Guard the class axis: class-1 SHAP values must reconstruct P(anomaly).
    reconstructed = ev[a] + anomaly_values.sum(axis=1)
    errors, notes = verify_additivity(
        reconstructed, scores, context.classifier.n_estimators
    )

    # Per-original-feature net contributions: sum the group's members.
    nets = {
        original: anomaly_values[:, idxs].sum(axis=1)
        for original, idxs in context.mapping.items()
    }
    base_value = float(ev[a])
    originals = list(context.mapping)

    results: list[dict[str, Any]] = []
    for i in range(len(samples)):
        contributions = [
            {
                "feature": original,
                "value": _json_value(samples.iloc[i][original]),
                "shap_contribution": float(nets[original][i]),
                "direction": (
                    "increases_anomaly_score"
                    if nets[original][i] >= 0
                    else "decreases_anomaly_score"
                ),
                "abs_contribution": float(abs(nets[original][i])),
            }
            for original in originals
        ]
        contributions.sort(key=lambda c: c["abs_contribution"], reverse=True)
        results.append(
            {
                # sample_id stays the POSITION within the sampled frame;
                # source_row_id is the record's row in the source dataframe
                # (the frame's index label), so report samples built as
                # X.iloc[...] (e.g. _report_sample_frame) carry their real
                # Train_data row identity explicitly.
                "index": i,
                "source_row_id": int(samples.index[i]),
                "prediction": "anomaly" if scores[i] >= 0.5 else "normal",
                "anomaly_score": float(scores[i]),
                "base_value": base_value,
                "top_features": contributions if top_k is None else contributions[:top_k],
                "additivity_error": float(errors[i]),
                "additivity_note": notes[i],
            }
        )
    return results


def _json_value(value: Any) -> Any:
    """Native, JSON-safe scalar for a raw feature value."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


# --- Global importance ----------------------------------------------------------
def global_importance(
    context: ExplanationContext, anomaly_values: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Mean |SHAP| per original feature, plus per-class signed means.

    `anomaly_values` is the (n, 116) class-1 SHAP block. Aggregation order:
    per sample, sum each original feature's transformed members into one net
    contribution, take the absolute value, then average over samples. The
    companion per-class block returns mean signed contributions for both
    classes (anomaly = class 1, normal = class 0).
    """
    nets = {
        original: anomaly_values[:, idxs].sum(axis=1)
        for original, idxs in context.mapping.items()
    }
    importance = sorted(
        (
            {"feature": original, "mean_abs_shap": float(np.abs(v).mean())}
            for original, v in nets.items()
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )
    total = sum(row["mean_abs_shap"] for row in importance) or 1.0
    for row in importance:
        row["share_of_total"] = float(row["mean_abs_shap"] / total)
    return importance, nets


def _report_sample_frame(
    X: pd.DataFrame, size: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministic report sample (sorted positions, no replacement)."""
    idx = background_sample_indices(len(X), size, seed)
    return X.iloc[idx], idx


def build_explanation_report(
    context: ExplanationContext,
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int = REPORT_SAMPLE_SIZE,
    sample_seed: int = BACKGROUND_SEED,
    top_k: int = TOP_K_DEFAULT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the report sample through the explainer and assemble the report."""
    sample_X, sample_idx = _report_sample_frame(X, sample_size, sample_seed)
    t0 = time.perf_counter()
    Z = context.preprocessing.transform(sample_X)
    values = np.asarray(context.explainer.shap_values(Z))
    elapsed = time.perf_counter() - t0

    a = ANOMALY_CLASS_INDEX
    anomaly_values = values[:, :, a]
    scores = context.classifier.predict_proba(Z)[:, 1]
    reconstructed = np.asarray(context.explainer.expected_value, dtype=float)[a] \
        + anomaly_values.sum(axis=1)
    errors, notes = verify_additivity(
        reconstructed, scores, context.classifier.n_estimators
    )
    tie_mask = [n is not None for n in notes]
    tie_count = int(sum(tie_mask))
    strict_errors = errors[~np.array(tie_mask, dtype=bool)] if tie_count else errors
    additivity_err = float(strict_errors.max()) if len(strict_errors) else 0.0
    importance, nets = global_importance(context, anomaly_values)

    # Per-class mean |SHAP| per original feature (anomaly vs normal class).
    per_class: dict[str, dict[str, float]] = {}
    for label, class_idx in (("anomaly", 1), ("normal", 0)):
        per_class[label] = {
            original: float(
                np.abs(values[:, idxs, class_idx].sum(axis=1)).mean()
            )
            for original, idxs in context.mapping.items()
        }

    report: dict[str, Any] = {
        "report": REPORT_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_population": (
            f"{min(sample_size, len(X))} deterministically sampled labelled "
            f"training rows (seed {sample_seed}) from data/Train_data.csv "
            "(25,192 rows); each embedded record carries sample_id (position "
            "within the report sample) and source_row_id (its zero-based row "
            "in Train_data.csv); SHAP importance describes the MODEL on the "
            "training distribution, not real-world or causal feature effects"
        ),
        "model": {
            "path": "models/rf_ids_pipeline.joblib",
            "sha256": context.model_hash,
            "classifier": type(context.classifier).__name__,
            "n_estimators": context.classifier.n_estimators,
            "class_weight": context.classifier.class_weight,
            "random_state": context.classifier.random_state,
            "retrained": False,
            "model_source": context.metadata["model_source"],
        },
        "dataset": {
            **evaluate.dataset_provenance(config.TRAIN_CSV),
            "target_column": config.TARGET_COLUMN,
            "class_counts": {
                "normal": int((y == 0).sum()),
                "anomaly": int((y == 1).sum()),
            },
        },
        "excluded_unlabelled_dataset":
            evaluate.excluded_unlabelled_dataset_provenance(),
        "shap": {
            "version": context.shap_version,
            "explainer": "shap.TreeExplainer",
            "feature_perturbation": "interventional",
            "explained_output": "P(anomaly) for class index 1 (model.classes_ = [0, 1])",
            "class_selection": (
                "binary RF with classes [0=normal, 1=anomaly]; the anomaly "
                "axis (index 1) is verified at explanation time by the "
                "additivity check against predict_proba[:, 1]"
            ),
            "base_value": float(np.asarray(context.explainer.expected_value)[a]),
            "baseline_definition": (
                "model's mean P(anomaly) over the 100-row background sample "
                "(interventional expected value for class 1)"
            ),
            "background": {
                "size": int(len(context.background_indices)),
                "seed": context.metadata["background_seed"],
                "selection": (
                    "deterministic random sample (numpy default_rng(seed), "
                    "no replacement, sorted) of labelled training rows; "
                    "fixed so every run of this report is reproducible"
                ),
                "rows_are_training_only": True,
            },
            "values_meaning": (
                "per-feature attributions of THIS model's P(anomaly) output "
                "for each record, relative to the baseline; not causal "
                "explanations"
            ),
            "additivity_max_error": additivity_err,
            "additivity_tie_samples": tie_count,
            "additivity_note_definition": ADDITIVITY_NOTE,
            "deterministic": True,
        },
        "feature_mapping": _mapping_block(context.preprocessing, context.mapping),
        "global_importance": importance,
        "per_class_mean_abs_shap": per_class,
        "explanation_methodology": METHODOLOGY_NOTE,
        "limitations": LIMITATIONS,
        "environment": {
            **evaluate.environment_info(),
            "shap": context.shap_version,
        },
        "provenance": {
            "git_revision": evaluate.git_revision(),
            "git_dirty": evaluate.git_dirty(),
            "command": "python -m src.nids.explain",
        },
        "_runtime": {
            "report_sample_size": int(len(sample_X)),
            "explainer_class": "TreeExplainer(interventional)",
            "elapsed_seconds": round(elapsed, 2),
            "per_sample_seconds_approx": round(elapsed / max(len(sample_X), 1), 4),
        },
    }
    per_sample = explain_samples(context, sample_X.head(5), top_k=top_k)
    report["sample_explanations"] = per_sample
    return report, importance


# --- Entry point ----------------------------------------------------------------
def main(
    background_size: int = BACKGROUND_SIZE,
    sample_size: int = REPORT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Build the explanation context from the saved model and write the report."""
    print(f"loading saved pipeline from {config.MODEL_PATH}")
    context = build_context(background_size=background_size)
    print(
        f"background: {len(context.background_indices)} training rows "
        f"(seed {BACKGROUND_SEED}), interventional TreeExplainer"
    )
    X, y = load_training_data(config.TRAIN_CSV)
    print(f"report sample: {min(sample_size, len(X))} training rows (seed {BACKGROUND_SEED})")

    report, importance = build_explanation_report(
        context, X, y, sample_size=sample_size
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLANATIONS_JSON_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nglobal feature importance (mean |SHAP| on P(anomaly), top 15):")
    for row in importance[:15]:
        print(f"  {row['feature']:<28}{row['mean_abs_shap']:.4f}  "
              f"({row['share_of_total']:.1%})")
    runtime = report["_runtime"]
    print(f"\nexplained {runtime['report_sample_size']} samples in "
          f"{runtime['elapsed_seconds']}s "
          f"(~{runtime['per_sample_seconds_approx']}s/sample)")
    print(f"saved explanation report to {EXPLANATIONS_JSON_PATH}")
    return report


if __name__ == "__main__":
    main()
