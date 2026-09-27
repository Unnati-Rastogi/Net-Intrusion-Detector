"""Cross-validation evaluation for the NIDS model.

Reproduces the model-selection methodology already used to approve the
current model, on the labelled training dataset only:

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fresh clone of build_pipeline() fitted per fold, so preprocessing is
    refit inside every fold on that fold's training rows only

Reported per fold and aggregated (mean +/- population std, ddof=0, the
convention behind the numbers quoted in the README):

    accuracy
    anomaly precision / recall / F1   (positive class = 'anomaly' = 1)
    normal precision / recall         (positive class = 'normal' = 0)
    macro F1

plus one out-of-fold confusion matrix (rows = actual, columns = predicted).

IMPORTANT - what this report is NOT:
These are cross-validation metrics on the TRAINING DISTRIBUTION
(data/Train_data.csv). data/Test_data.csv is unlabelled (no ground-truth
'class' column) and is never evaluated, merged, or used here. Nothing in
this report measures held-out-test or production performance.

Usage (from the project root):
    python -m src.nids.evaluate

Outputs:
    reports/metrics.json           machine-readable report + provenance
    reports/confusion_matrix.png   out-of-fold confusion matrix figure
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config
from .data import load_training_data
from .pipeline import build_pipeline

# --- Output locations -------------------------------------------------------
REPORTS_DIR = config.PROJECT_ROOT / "reports"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_PNG_PATH = REPORTS_DIR / "confusion_matrix.png"

# --- Report identity --------------------------------------------------------
SCHEMA_VERSION = "1.0"
EVALUATION_KIND = "cross_validation_on_training_distribution"
EVALUATION_NOTE = (
    "All metrics are out-of-fold 5-fold stratified cross-validation on the "
    "labelled training dataset (data/Train_data.csv) only, i.e. the training "
    "distribution. The unlabelled data/Test_data.csv has no ground truth and "
    "is NOT evaluated. These numbers are not held-out-test or production "
    "performance estimates."
)

# Metric names exactly as they appear in reports/metrics.json.
METRIC_KEYS = (
    "accuracy",
    "precision_anomaly",
    "recall_anomaly",
    "f1_anomaly",
    "precision_normal",
    "recall_normal",
    "f1_macro",
)


# --- Provenance helpers -----------------------------------------------------
def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    """Current git commit hash, or 'unknown' outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes (None if unknown)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def environment_info() -> dict[str, str]:
    """Library/runtime versions relevant to reproducing this evaluation."""
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def _display_path(path: Path) -> str:
    """POSIX-style path relative to the project root, or absolute otherwise."""
    path = path.resolve()
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:  # dataset outside the project (e.g. tests)
        return path.as_posix()


def dataset_provenance(path: Path) -> dict[str, Any]:
    """Path, SHA-256 and shape of a dataset file (loaded to get its shape)."""
    df = pd.read_csv(path)
    return {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "shape": {"rows": int(df.shape[0]), "columns_including_target": int(df.shape[1])},
    }


def excluded_unlabelled_dataset_provenance() -> dict[str, Any] | None:
    """Provenance for Test_data.csv, recorded to document that it is NOT used.

    The file is read only to hash/measure it and verify it has no 'class'
    column; its rows are never scored or evaluated as though labelled.
    """
    if not config.TEST_CSV.exists():
        return None
    df = pd.read_csv(config.TEST_CSV)
    info: dict[str, Any] = {
        "path": _display_path(config.TEST_CSV),
        "sha256": sha256_file(config.TEST_CSV),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "has_class_column": bool(config.TARGET_COLUMN in df.columns),
        "reason_excluded": (
            "unlabelled (no ground-truth 'class' column); never evaluated, "
            "merged, or used to tune anything in this report"
        ),
    }
    if info["has_class_column"]:
        info["warning"] = (
            "test file unexpectedly contains a 'class' column; it is still "
            "NOT evaluated in this report"
        )
    return info


def model_provenance() -> dict[str, Any]:
    """Model configuration as used during evaluation (unchanged from training)."""
    model = build_pipeline()
    clf = model.named_steps["classifier"]
    return {
        "pipeline_class": f"{type(model).__module__}.{type(model).__name__}",
        "steps": [name for name, _ in model.steps],
        "classifier": type(clf).__name__,
        "classifier_params": {
            "n_estimators": clf.n_estimators,
            "class_weight": clf.class_weight,
            "random_state": clf.random_state,
            "n_jobs": clf.n_jobs,
        },
        "preprocessing_refit_per_fold": True,
    }


# --- Cross-validation -------------------------------------------------------
def run_cv_folds(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, list[float]], np.ndarray]:
    """Run stratified k-fold CV for any model factory; reusable across models.

    The methodology is the one used to approve the current model: a shuffled,
    stratified k-fold split (random_state=42) with a fresh model (preprocessing
    included) fitted on each fold's training rows only. Rows are positional
    indices from the splitter, applied with .iloc to keep the fold assignment
    reproducible and leak-free. Every candidate in the model comparison uses
    this same function, so all models are scored on identical folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: dict[str, list[float]] = {name: [] for name in METRIC_KEYS}
    cm_total = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        folds["accuracy"].append(float(accuracy_score(y_test, pred)))
        folds["precision_anomaly"].append(
            float(precision_score(y_test, pred, pos_label=1, zero_division=0))
        )
        folds["recall_anomaly"].append(
            float(recall_score(y_test, pred, pos_label=1, zero_division=0))
        )
        folds["f1_anomaly"].append(
            float(f1_score(y_test, pred, pos_label=1, zero_division=0))
        )
        folds["precision_normal"].append(
            float(precision_score(y_test, pred, pos_label=0, zero_division=0))
        )
        folds["recall_normal"].append(
            float(recall_score(y_test, pred, pos_label=0, zero_division=0))
        )
        folds["f1_macro"].append(
            float(f1_score(y_test, pred, average="macro", zero_division=0))
        )

        # labels=[0, 1] -> rows/cols ordered (normal, anomaly), no surprises
        # if a fold happens to score perfectly.
        cm_total += confusion_matrix(y_test, pred, labels=[0, 1])

    return folds, cm_total


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, list[float]], np.ndarray]:
    """CV evaluation of the production pipeline (thin, stable wrapper).

    Delegates to run_cv_folds with the production build_pipeline factory;
    signature and behavior are unchanged from the original implementation.
    """
    return run_cv_folds(
        build_pipeline, X, y, n_splits=n_splits, random_state=random_state
    )


def summarize_folds(folds: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-fold metric values into mean / std / folds blocks.

    std is the population standard deviation (ddof=0), the convention behind
    the +/- numbers quoted in the README.
    """
    summary: dict[str, dict[str, Any]] = {}
    for name, values in folds.items():
        rounded = [round(float(v), 6) for v in values]
        arr = np.asarray(rounded, dtype=float)
        summary[name] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=0)), 6),
            "folds": rounded,
        }
    return summary


# --- Report + figure --------------------------------------------------------
def build_report(
    X: pd.DataFrame,
    y: pd.Series,
    metric_summary: dict[str, dict[str, Any]],
    cm: np.ndarray,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """Assemble the full machine-readable evaluation report."""
    return {
        "report": "nids_cross_validation_evaluation",
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": {
            "kind": EVALUATION_KIND,
            "note": EVALUATION_NOTE,
            "cross_validation": {
                "splitter": "sklearn.model_selection.StratifiedKFold",
                "n_splits": n_splits,
                "shuffle": True,
                "random_state": random_state,
                "out_of_fold": True,
                "preprocessing_refit_per_fold": True,
            },
            "positive_class": {"label": 1, "name": config.POSITIVE_CLASS},
            "negative_class": {"label": 0, "name": config.NEGATIVE_CLASS},
        },
        "dataset": {
            **dataset_provenance(config.TRAIN_CSV),
            "target_column": config.TARGET_COLUMN,
            "feature_columns": int(X.shape[1]),
            "class_counts": {
                "normal": int((y == 0).sum()),
                "anomaly": int((y == 1).sum()),
            },
        },
        "excluded_unlabelled_dataset": excluded_unlabelled_dataset_provenance(),
        "model": model_provenance(),
        "metrics": metric_summary,
        "confusion_matrix": {
            "rows": "actual",
            "columns": "predicted",
            "labels": [config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
            "out_of_fold_counts": cm.tolist(),
            "figure_path": "reports/confusion_matrix.png",
        },
        "environment": environment_info(),
        "provenance": {
            "git_revision": git_revision(),
            "git_dirty": git_dirty(),
            "command": "python -m src.nids.evaluate",
        },
    }


def save_confusion_matrix_figure(cm: np.ndarray, path: Path) -> None:
    """Render the out-of-fold confusion matrix to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="connections")

    labels = [f"{config.NEGATIVE_CLASS} (0)", f"{config.POSITIVE_CLASS} (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(
        "Out-of-fold confusion matrix\n"
        "5-fold stratified cross-validation on training data only",
        fontsize=11,
    )

    threshold = (cm.max() / 2) if cm.max() else 1
    for i in range(2):
        row_total = int(cm[i].sum())
        for j in range(2):
            count = int(cm[i, j])
            share = count / row_total if row_total else 0.0
            ax.text(
                j,
                i,
                f"{count:,}\n{share:.2%} of actual",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm[i, j] > threshold else "#1a1a1a",
            )

    clf = build_pipeline().named_steps["classifier"]
    fig.text(
        0.99,
        0.01,
        (
            f"RandomForest(n_estimators={clf.n_estimators}, "
            f"class_weight={clf.class_weight!r}, random_state={clf.random_state}) "
            "on data/Train_data.csv only; unlabelled Test_data.csv not evaluated"
        ),
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Entry point ------------------------------------------------------------
def main(n_splits: int = 5) -> dict[str, Any]:
    """Run the CV evaluation and write reports/metrics.json + the PNG."""
    X, y = load_training_data()
    print(f"loaded training data: {X.shape[0]} rows x {X.shape[1]} feature columns")
    print(f"class balance: normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}")
    print(
        f"methodology: {n_splits}-fold stratified CV (shuffle=True, "
        f"random_state={config.RANDOM_STATE}); preprocessing refit per fold"
    )
    print(EVALUATION_NOTE)

    folds, cm = evaluate_cv(X, y, n_splits=n_splits)
    summary = summarize_folds(folds)

    print(f"\ncross-validation metrics (mean +/- std over {n_splits} folds):")
    for name in METRIC_KEYS:
        print(f"  {name:<18}{summary[name]['mean']:.4f} +/- {summary[name]['std']:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_figure(cm, CONFUSION_PNG_PATH)
    report = build_report(X, y, summary, cm, n_splits=n_splits,
                          random_state=config.RANDOM_STATE)
    METRICS_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nsaved metrics report to {METRICS_JSON_PATH}")
    print(f"saved confusion matrix figure to {CONFUSION_PNG_PATH}")
    return report


if __name__ == "__main__":
    main()
