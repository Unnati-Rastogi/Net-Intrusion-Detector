"""Evaluate the saved NIDS model on the labeled NSL-KDD benchmark holdout.

The project's existing Test_data.csv is the 41-feature KDDTest+ test split,
but its labels were removed. This module uses the public labeled KDDTest+
reference file only to restore the ground-truth labels after verifying every
feature row matches exactly. It never fits or changes the model.

Run from the project root with ``python -m src.nids.holdout``.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from . import config
from .data import load_inference_data
from .predict import load_model

LABELED_TEST_PATH = config.DATA_DIR / "KDDTest+_labeled.csv"
REPORT_PATH = config.PROJECT_ROOT / "reports" / "holdout_metrics.json"
SOURCE_URL = "https://github.com/jmnwong/NSL-KDD-Dataset"
SOURCE_FILE = "KDDTest+.TXT"
SOURCE_COLUMNS = config.FEATURE_COLUMNS + ["attack_type", "difficulty"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
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


def load_labeled_holdout() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load labels and prove they align with the project's inference rows."""
    if not LABELED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"labeled benchmark file not found at {LABELED_TEST_PATH}; "
            "see the README section 'Labeled benchmark evaluation'"
        )

    source = pd.read_csv(LABELED_TEST_PATH, header=None, names=SOURCE_COLUMNS)
    features = load_inference_data()
    expected = source[config.FEATURE_COLUMNS].reset_index(drop=True)
    actual = features[config.FEATURE_COLUMNS].reset_index(drop=True)
    if expected.shape != actual.shape or not expected.equals(actual):
        raise ValueError(
            "labeled KDDTest+ rows do not exactly match data/Test_data.csv; "
            "refusing to attach labels to a different record order"
        )

    label = source["attack_type"].astype("string").str.strip().str.lower()
    if label.isna().any() or label.eq("").any():
        raise ValueError("benchmark contains invalid ground-truth labels")
    y = (label != "normal").astype(int)
    return features, y, label


def build_report(model=None) -> dict[str, Any]:
    """Score all labeled holdout records and summarize misses by attack type."""
    X, y, attack_type = load_labeled_holdout()
    model = model or load_model()
    if list(model.feature_names_in_) != list(config.FEATURE_COLUMNS):
        raise ValueError("saved model feature columns do not match the benchmark")

    predicted = pd.Series(model.predict(X).astype(int), index=X.index)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    anomaly_recall = float(recall_score(y, predicted, pos_label=1, zero_division=0))
    attack_types = []
    for name, indexes in attack_type.groupby(attack_type, sort=True).groups.items():
        if name == "normal":
            continue
        actual_count = int(len(indexes))
        detected = int(predicted.loc[indexes].sum())
        attack_types.append({
            "name": str(name),
            "records": actual_count,
            "detected": detected,
            "missed": actual_count - detected,
            "detection_rate": round(detected / actual_count, 6) if actual_count else 0.0,
        })
    attack_types.sort(key=lambda item: (-item["missed"], item["name"]))

    return {
        "report": "nids_labeled_holdout_evaluation",
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": {
            "name": "NSL-KDD KDDTest+ labeled holdout",
            "kind": "fixed_benchmark_holdout",
            "records_scored": int(len(X)),
            "features": int(X.shape[1]),
            "ground_truth_source": SOURCE_URL,
            "source_file": SOURCE_FILE,
            "label_definition": "normal is normal; every named attack type is anomaly",
            "scope_note": (
                "This replays historical, labeled benchmark records through the saved model. "
                "It is not a live network feed or a measure of current network activity."
            ),
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(y, predicted)), 6),
            "precision_anomaly": round(float(precision_score(y, predicted, pos_label=1, zero_division=0)), 6),
            "recall_anomaly": round(anomaly_recall, 6),
            "f1_anomaly": round(float(f1_score(y, predicted, pos_label=1, zero_division=0)), 6),
            "recall_normal": round(float(recall_score(y, predicted, pos_label=0, zero_division=0)), 6),
            "false_alarms": fp,
            "missed_attacks": fn,
            "correctly_detected_attacks": tp,
            "correctly_cleared_normal": tn,
            "confusion_matrix": {
                "labels": ["normal", "attack"],
                "rows": "actual",
                "columns": "predicted",
                "counts": [[tn, fp], [fn, tp]],
            },
        },
        "class_counts": {
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
            "predicted_normal": int((predicted == 0).sum()),
            "predicted_attack": int((predicted == 1).sum()),
        },
        "attack_types": attack_types,
        "provenance": {
            "features_path": "data/Test_data.csv",
            "labels_path": LABELED_TEST_PATH.relative_to(config.PROJECT_ROOT).as_posix(),
            "features_sha256": sha256_file(config.TEST_CSV),
            "labeled_source_sha256": sha256_file(LABELED_TEST_PATH),
            "model_sha256": sha256_file(config.MODEL_PATH),
            "git_revision": _git_revision(),
        },
    }


def main() -> dict[str, Any]:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    print(f"scored {report['evaluation']['records_scored']:,} labeled holdout records")
    print(f"accuracy: {metrics['accuracy']:.1%}")
    print(f"attack detection (recall): {metrics['recall_anomaly']:.1%}")
    print(f"missed attacks: {metrics['missed_attacks']:,}")
    print(f"false alarms: {metrics['false_alarms']:,}")
    print(f"saved report to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
