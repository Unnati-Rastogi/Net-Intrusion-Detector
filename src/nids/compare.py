"""Model comparison: Logistic Regression vs Random Forest vs XGBoost.

All three candidates share ONE preprocessing implementation — the ColumnTransformer
from pipeline.build_pipeline() — and ONE fold assignment: run_cv_folds() from
evaluate.py (StratifiedKFold(5, shuffle=True, random_state=42), fresh pipeline
clone fitted per fold, preprocessing refit on fold-training rows only).

Candidates:
    lr_baseline       LogisticRegression(max_iter=2000, class_weight='balanced',
                                        random_state=42) on StandardScaler-scaled
                                        features (a linear model cannot compare raw
                                        byte counts spanning 0..~1e9; scaling is
                                        applied INSIDE the pipeline, after the shared
                                        preprocessing, so it is fit on fold-train only
                                        and does not leak).
    rf_production     The existing RandomForest pipeline — the current production
                                        configuration, evaluated unchanged.
    xgb_candidate     XGBClassifier with scale_pos_weight mirroring RF's
                                        class_weight='balanced' (n_normal/n_anomaly),
                                        deterministic on CPU.

No model-selection tuning happens on the CV test folds: every candidate is
evaluated once with its documented configuration. No winner is declared here —
model selection is an evidence-based decision made after reviewing this report.

IMPORTANT: metrics are 5-fold StratifiedKFold on the labelled training data
(data/Train_data.csv) — the training distribution, NOT held-out-test or
production performance. The unlabelled data/Test_data.csv is never evaluated.

Usage (from the project root):
    python -m src.nids.compare

Outputs:
    reports/model_comparison.json   machine-readable report + provenance
    reports/model_comparison.csv    flat mean±std table (one row per model)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config, evaluate
from . import pipeline as pipeline_module
from .data import load_training_data

# --- Output locations -------------------------------------------------------
REPORTS_DIR = evaluate.REPORTS_DIR
COMPARISON_JSON_PATH = REPORTS_DIR / "model_comparison.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"

# --- Report identity --------------------------------------------------------
SCHEMA_VERSION = "1.0"
COMPARISON_REPORT_NAME = "nids_model_comparison"
EVALUATION_POPULATION = "5-fold StratifiedKFold on labelled training data"
EVALUATION_NOTE = (
    "All metrics are out-of-fold 5-fold stratified cross-validation on the "
    "labelled training dataset (data/Train_data.csv) only, i.e. the training "
    "distribution. The unlabelled data/Test_data.csv has no ground truth and "
    "is NOT evaluated by any candidate. These numbers are not held-out-test or "
    "production performance estimates."
)
MODEL_SELECTION_NOTE = (
    "No winner is declared by this artifact. Final model selection will be an "
    "evidence-based decision weighing security-relevant metrics (anomaly "
    "recall, precision trade-off, calibration, explainability) — the production "
    "prediction pipeline remains RandomForest until that review."
)


# --- Candidate factories (single shared preprocessing) -----------------------
def build_lr_pipeline() -> Pipeline:
    """LR baseline: shared preprocessing + StandardScaler + LogisticRegression."""
    preprocessing = pipeline_module.build_pipeline().named_steps["preprocessing"]
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("scaling", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=config.RANDOM_STATE)),
        ]
    )


def build_rf_pipeline() -> Pipeline:
    """Production RF pipeline, evaluated unchanged."""
    return pipeline_module.build_pipeline()


def scale_pos_weight() -> float:
    """n_normal / n_anomaly, mirroring RF's class_weight='balanced'.

    Read from config.TRAIN_CSV at call time (not import time) so the loader
    always sees the currently configured dataset.
    """
    _, y = load_training_data(config.TRAIN_CSV)
    return int((y == 0).sum()) / int((y == 1).sum())


def build_xgb_pipeline() -> Pipeline:
    """XGBoost candidate: shared preprocessing + XGBClassifier."""
    spw = scale_pos_weight()
    preprocessing = pipeline_module.build_pipeline().named_steps["preprocessing"]
    params = {**config.XGB_PARAMS, "scale_pos_weight": spw, "n_jobs": -1}
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", XGBClassifier(**params)),
        ]
    )


def candidate_registry() -> dict[str, dict[str, Any]]:
    """Ordered candidate spec used for evaluation AND artifact generation."""
    return {
        "lr_baseline": {
            "factory": build_lr_pipeline,
            "description": "LogisticRegression (project baseline, CV F1(anomaly) ~= 0.9691)",
        },
        "rf_production": {
            "factory": build_rf_pipeline,
            "description": "RandomForest — current production model, evaluated unchanged",
        },
        "xgb_candidate": {
            "factory": build_xgb_pipeline,
            "description": "XGBoost — comparison candidate, not production",
        },
    }


# --- Evaluation --------------------------------------------------------------
def compare_models(
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
    candidates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate every candidate on IDENTICAL folds; return per-candidate results.

    run_cv_folds uses positional .iloc indexing from the splitter, so calling it
    repeatedly for the same (X, y) yields identical fold assignment for every
    model — a fair, leak-free comparison. Preprocessing is refit inside each
    fold for every candidate.
    """
    X, y = load_training_data(config.TRAIN_CSV)
    registry = candidates if candidates is not None else candidate_registry()
    results: dict[str, dict[str, Any]] = {}
    for name, spec in registry.items():
        print(f"evaluating {name} ...")
        folds, cm = evaluate.run_cv_folds(
            spec["factory"], X, y, n_splits=n_splits, random_state=random_state
        )
        results[name] = {
            "description": spec["description"],
            "configuration": configuration_of(name, spec["factory"]()),
            "fold_metrics": {k: [round(float(v), 6) for v in vs] for k, vs in folds.items()},
            "summary": evaluate.summarize_folds(folds),
            "confusion_matrix": {
                "rows": "actual",
                "columns": "predicted",
                "labels": [config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
                "out_of_fold_counts": cm.tolist(),
            },
        }
    return results


def configuration_of(model_name: str, pipeline: Pipeline) -> dict[str, Any]:
    """Serializable configuration block for a fitted-shape pipeline."""
    preprocessing = pipeline.named_steps["preprocessing"]
    cfg: dict[str, Any] = {
        "preprocessing": {
            "source": "src/nids/pipeline.py::build_pipeline (shared, single implementation)",
            "steps": [name for name, _ in pipeline.steps if name != "classifier"],
            "const_drop": list(config.CONSTANT_COLUMNS),
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        }
    }
    if "scaling" in pipeline.named_steps:
        cfg["scaling"] = "StandardScaler (fit inside fold-train only)"
    clf = pipeline.named_steps["classifier"]
    cfg["classifier"] = type(clf).__name__
    if model_name == "lr_baseline":
        cfg["classifier_params"] = {
            "max_iter": clf.max_iter, "class_weight": clf.class_weight,
            "random_state": clf.random_state, "solver": clf.solver, "C": clf.C,
        }
    elif model_name == "rf_production":
        cfg["classifier_params"] = {
            "n_estimators": clf.n_estimators, "class_weight": clf.class_weight,
            "random_state": clf.random_state, "n_jobs": clf.n_jobs,
        }
    elif model_name == "xgb_candidate":
        cfg["classifier_params"] = {
            "n_estimators": clf.n_estimators, "learning_rate": clf.learning_rate,
            "max_depth": clf.max_depth, "subsample": clf.subsample,
            "colsample_bytree": clf.colsample_bytree, "reg_lambda": clf.reg_lambda,
            "tree_method": clf.tree_method, "eval_metric": clf.eval_metric,
            "random_state": clf.random_state, "n_jobs": clf.n_jobs,
            "scale_pos_weight": round(float(clf.scale_pos_weight), 4),
        }
    return cfg


# --- Report + CSV ------------------------------------------------------------
def build_report(
    results: dict[str, dict[str, Any]],
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """Assemble the machine-readable model-comparison report."""
    return {
        "report": COMPARISON_REPORT_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": {
            "kind": "model_comparison_cross_validation_on_training_distribution",
            "evaluation_population": EVALUATION_POPULATION,
            "note": EVALUATION_NOTE,
            "cross_validation": {
                "splitter": "sklearn.model_selection.StratifiedKFold",
                "n_splits": n_splits,
                "shuffle": True,
                "random_state": random_state,
                "out_of_fold": True,
                "preprocessing_refit_per_fold": True,
                "identical_folds_for_all_candidates": True,
            },
            "positive_class": {"label": 1, "name": config.POSITIVE_CLASS},
            "negative_class": {"label": 0, "name": config.NEGATIVE_CLASS},
        },
        "dataset": {
            **evaluate.dataset_provenance(config.TRAIN_CSV),
            "target_column": config.TARGET_COLUMN,
            "class_counts": _class_counts(),
        },
        "excluded_unlabelled_dataset": evaluate.excluded_unlabelled_dataset_provenance(),
        "models": results,
        "model_selection": {
            "decision": "not_declared",
            "note": MODEL_SELECTION_NOTE,
        },
        "environment": {**evaluate.environment_info(), "xgboost": _xgb_version()},
        "provenance": {
            "git_revision": evaluate.git_revision(),
            "git_dirty": evaluate.git_dirty(),
            "command": "python -m src.nids.compare",
        },
    }


def _class_counts() -> dict[str, int]:
    _, y = load_training_data(config.TRAIN_CSV)
    return {"normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())}


def _xgb_version() -> str:
    import xgboost
    return xgboost.__version__


def comparison_csv(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flat one-row-per-model table of mean±std for every metric."""
    rows: list[dict[str, Any]] = []
    for name, res in results.items():
        row: dict[str, Any] = {"model": name}
        for metric in evaluate.METRIC_KEYS:
            s = res["summary"][metric]
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
        row["confusion_matrix"] = str(res["confusion_matrix"]["out_of_fold_counts"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(n_splits: int = 5) -> dict[str, Any]:
    """Run the three-way comparison and write both artifacts."""
    print(EVALUATION_NOTE)
    results = compare_models(n_splits=n_splits)

    print(f"\nmodel comparison (mean +/- std over {n_splits} folds):")
    header = ["model"] + list(evaluate.METRIC_KEYS)
    print("  " + "  ".join(f"{h:>18}" for h in header))
    for name, res in results.items():
        cells = [f"{res['summary'][m]['mean']:.4f} +/- {res['summary'][m]['std']:.4f}"
                 for m in evaluate.METRIC_KEYS]
        print("  " + "  ".join([f"{name:>18}"] + [f"{c:>18}" for c in cells]))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(results, n_splits=n_splits, random_state=config.RANDOM_STATE)
    COMPARISON_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    comparison_csv(results).to_csv(COMPARISON_CSV_PATH, index=False)

    print(f"\nsaved comparison report to {COMPARISON_JSON_PATH}")
    print(f"saved comparison table to {COMPARISON_CSV_PATH}")
    return report


if __name__ == "__main__":
    main()
