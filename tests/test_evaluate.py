"""Tests for the NIDS cross-validation evaluation outputs.

These test the schema and internal consistency of reports/metrics.json and
the confusion-matrix figure produced by src.nids.evaluate. The full CV run
is too heavy for the unit suite, so the fast tests use a tiny synthetic
dataset, and one opt-in test runs the real evaluation when
RUN_SLOW_EVAL_TESTS=1 is set in the environment.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nids import config, evaluate


def _synthetic_training_csv(n_rows: int = 400) -> str:
    """Tiny NSL-KDD-style CSV (full 41-feature schema) in a temp dir."""
    rng = np.random.default_rng(7)
    half = n_rows // 2

    # Separable numeric signal on a handful of columns; the two training-data
    # constant columns stay constant, as in the real dataset.
    normal = pd.DataFrame(
        {
            "duration": rng.integers(0, 10, half),
            "src_bytes": rng.integers(0, 500, half),
            "dst_bytes": rng.integers(0, 5000, half),
            "count": rng.integers(1, 5, half),
            "srv_count": rng.integers(1, 5, half),
            "same_srv_rate": rng.uniform(0.5, 1.0, half),
        }
    )
    anomaly = pd.DataFrame(
        {
            "duration": rng.integers(10, 60, half),
            "src_bytes": rng.integers(5000, 9000, half),
            "dst_bytes": rng.integers(5000, 9000, half),
            "count": rng.integers(50, 100, half),
            "srv_count": rng.integers(50, 100, half),
            "same_srv_rate": rng.uniform(0.0, 0.4, half),
        }
    )

    # Fill in the remaining dataset columns (categoricals, flags, rates...)
    # so the real pipeline can be used unchanged, exactly as in production.
    services = ["http", "private", "ftp", "smtp"]
    flags = ["SF", "S0", "REJ"]
    for part in (normal, anomaly):
        for column in config.FEATURE_COLUMNS:
            if column in part.columns:
                continue
            if column == "protocol_type":
                part[column] = rng.choice(["tcp", "udp", "icmp"], len(part))
            elif column == "service":
                part[column] = rng.choice(services, len(part))
            elif column == "flag":
                part[column] = rng.choice(flags, len(part))
            elif column in config.CONSTANT_COLUMNS:
                part[column] = 0
            elif column in ("land", "logged_in", "root_shell", "is_guest_login"):
                part[column] = rng.integers(0, 2, len(part))
            elif column.startswith(("serror", "rerror", "srv_serror", "srv_rerror",
                                    "dst_host_serror", "dst_host_rerror",
                                    "dst_host_srv_serror", "dst_host_srv_rerror",
                                    "same_srv", "diff_srv", "srv_diff_host",
                                    "dst_host_same_srv", "dst_host_diff_srv",
                                    "dst_host_srv_diff", "dst_host_same_src")):
                part[column] = rng.uniform(0.0, 1.0, len(part)).round(2)
            else:
                part[column] = rng.integers(0, 10, len(part))

    df = pd.concat([normal, anomaly], ignore_index=True)
    labels = ["normal"] * half + ["anomaly"] * half
    df.insert(0, "class", labels)
    df = df[["class"] + config.FEATURE_COLUMNS]  # canonical dataset order
    assert list(df.columns) == ["class"] + config.FEATURE_COLUMNS

    tmp = tempfile.mkdtemp(prefix="nids_eval_test_")
    path = os.path.join(tmp, "Train_data.csv")
    df.to_csv(path, index=False)
    return path


def _run_synthetic_evaluation() -> tuple[dict, Path]:
    """Evaluate the synthetic dataset in a temp dir; returns (report, tmpdir)."""
    csv_path = Path(_synthetic_training_csv())
    tmpdir = csv_path.parent
    # evaluate.main() derives report paths from config.PROJECT_ROOT, so patch
    # both the paths and the loaders to point at the temp dataset.
    train_csv_backup = config.TRAIN_CSV
    reports_backup = evaluate.REPORTS_DIR
    json_backup = evaluate.METRICS_JSON_PATH
    png_backup = evaluate.CONFUSION_PNG_PATH
    config.TRAIN_CSV = csv_path
    evaluate.REPORTS_DIR = tmpdir
    evaluate.METRICS_JSON_PATH = tmpdir / "metrics.json"
    evaluate.CONFUSION_PNG_PATH = tmpdir / "confusion_matrix.png"
    try:
        from src.nids.data import load_training_data

        X, y = load_training_data(csv_path)
        folds, cm = evaluate.evaluate_cv(X, y)
        summary = evaluate.summarize_folds(folds)
        report = evaluate.build_report(X, y, summary, cm, n_splits=5,
                                       random_state=config.RANDOM_STATE)
        (tmpdir / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        evaluate.save_confusion_matrix_figure(cm, tmpdir / "confusion_matrix.png")
    finally:
        config.TRAIN_CSV = train_csv_backup
        evaluate.REPORTS_DIR = reports_backup
        evaluate.METRICS_JSON_PATH = json_backup
        evaluate.CONFUSION_PNG_PATH = png_backup
    return report, tmpdir


class TestEvaluateOutputs(unittest.TestCase):
    """Schema and consistency checks on a full synthetic evaluation run."""

    @classmethod
    def setUpClass(cls):
        cls.report, cls.tmpdir = _run_synthetic_evaluation()

    def test_report_identity_and_caveat(self):
        self.assertEqual(self.report["report"], "nids_cross_validation_evaluation")
        self.assertEqual(self.report["schema_version"], "1.0")
        self.assertIn("timestamp_utc", self.report)
        evaluation = self.report["evaluation"]
        self.assertEqual(evaluation["kind"], "cross_validation_on_training_distribution")
        note = evaluation["note"]
        self.assertIn("training", note)
        self.assertIn("Test_data.csv", note)
        self.assertIn("NOT evaluated", note)

    def test_report_declares_cross_validation(self):
        cv = self.report["evaluation"]["cross_validation"]
        self.assertEqual(cv["splitter"], "sklearn.model_selection.StratifiedKFold")
        self.assertEqual(cv["n_splits"], 5)
        self.assertTrue(cv["shuffle"])
        self.assertEqual(cv["random_state"], 42)
        self.assertTrue(cv["out_of_fold"])
        self.assertTrue(cv["preprocessing_refit_per_fold"])

    def test_metrics_schema_and_fold_counts(self):
        expected = {
            "accuracy",
            "precision_anomaly",
            "recall_anomaly",
            "f1_anomaly",
            "precision_normal",
            "recall_normal",
            "f1_macro",
        }
        metrics = self.report["metrics"]
        self.assertEqual(set(metrics), expected)
        for name, block in metrics.items():
            self.assertEqual(set(block), {"mean", "std", "folds"})
            self.assertEqual(len(block["folds"]), 5)
            for value in block["folds"]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(block["mean"], min(block["folds"]))
            self.assertLessEqual(block["mean"], max(block["folds"]))
            self.assertGreaterEqual(block["std"], 0.0)
        # The synthetic data is separable -> near-perfect scores.
        self.assertGreaterEqual(metrics["accuracy"]["mean"], 0.99)
        self.assertGreaterEqual(metrics["f1_anomaly"]["mean"], 0.99)
        self.assertGreaterEqual(metrics["f1_macro"]["mean"], 0.99)

    def test_metrics_match_readme_convention_after_real_run(self):
        """Opt-in: full CV on the real dataset reproduces README numbers.

        Run with RUN_SLOW_EVAL_TESTS=1 (a few minutes; fits 5 forests).
        """
        if os.environ.get("RUN_SLOW_EVAL_TESTS") != "1":
            self.skipTest("set RUN_SLOW_EVAL_TESTS=1 to run the full CV check")
        report = evaluate.main()
        acc = report["metrics"]["accuracy"]
        f1 = report["metrics"]["f1_anomaly"]
        self.assertAlmostEqual(acc["mean"], 0.9975, places=3)
        self.assertAlmostEqual(acc["std"], 0.0013, places=3)
        self.assertAlmostEqual(f1["mean"], 0.9973, places=3)
        self.assertAlmostEqual(f1["std"], 0.0014, places=3)

    def test_dataset_provenance_includes_hash_shape_and_class_counts(self):
        dataset = self.report["dataset"]
        self.assertTrue(dataset["path"].endswith("Train_data.csv"))
        self.assertRegex(dataset["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(dataset["shape"]["rows"], 400)
        self.assertEqual(dataset["class_counts"], {"normal": 200, "anomaly": 200})

    @unittest.skipUnless(
        config.TEST_CSV.exists(), "requires data/Test_data.csv (gitignored)"
    )
    def test_unlabelled_test_dataset_recorded_as_excluded(self):
        excluded = self.report.get("excluded_unlabelled_dataset")
        self.assertIsNotNone(excluded)
        self.assertTrue(excluded["path"].endswith("Test_data.csv"))
        self.assertRegex(excluded["sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(excluded["has_class_column"])
        self.assertIn("unlabelled", excluded["reason_excluded"])

    def test_model_provenance_matches_approved_configuration(self):
        model = self.report["model"]
        self.assertEqual(model["classifier"], "RandomForestClassifier")
        self.assertEqual(model["classifier_params"]["n_estimators"], 200)
        self.assertEqual(model["classifier_params"]["class_weight"], "balanced")
        self.assertEqual(model["classifier_params"]["random_state"], 42)
        self.assertTrue(model["preprocessing_refit_per_fold"])

    def test_provenance_includes_environment_and_git(self):
        provenance = self.report["provenance"]
        self.assertRegex(provenance["git_revision"], r"^[0-9a-f]{40}$|unknown")
        self.assertIn("git_dirty", provenance)
        environment = self.report["environment"]
        self.assertIn("sklearn", environment)
        self.assertRegex(environment["sklearn"], r"^\d+\.\d+")

    def test_confusion_matrix_block_consistent(self):
        cm_block = self.report["confusion_matrix"]
        self.assertEqual(cm_block["rows"], "actual")
        self.assertEqual(cm_block["columns"], "predicted")
        self.assertEqual(cm_block["labels"], ["normal", "anomaly"])
        cm = np.asarray(cm_block["out_of_fold_counts"])
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(int(cm.sum()), 400)  # every row scored exactly once
        accuracy_from_cm = float(np.trace(cm)) / cm.sum()
        self.assertAlmostEqual(
            accuracy_from_cm, self.report["metrics"]["accuracy"]["mean"], delta=0.005
        )
        self.assertEqual(cm_block["figure_path"], "reports/confusion_matrix.png")


class TestEvaluationArtifacts(unittest.TestCase):
    """The real reports/ artifacts, when they have been generated."""

    @classmethod
    def setUpClass(cls):
        cls.metrics_path = evaluate.METRICS_JSON_PATH
        cls.png_path = evaluate.CONFUSION_PNG_PATH

    def test_generated_metrics_json_exists_and_is_valid(self):
        if not self.metrics_path.exists():
            self.skipTest("run 'python -m src.nids.evaluate' to generate reports/")
        report = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(report["evaluation"]["kind"],
                         "cross_validation_on_training_distribution")
        self.assertEqual(sorted(report["metrics"]),
                         sorted(evaluate.METRIC_KEYS))
        for block in report["metrics"].values():
            self.assertEqual(len(block["folds"]), report["evaluation"]
                             ["cross_validation"]["n_splits"])

    def test_generated_confusion_matrix_png_exists(self):
        if not self.png_path.exists():
            self.skipTest("run 'python -m src.nids.evaluate' to generate reports/")
        header = self.png_path.read_bytes()[:8]
        self.assertEqual(header, b"\x89PNG\r\n\x1a\n")  # PNG magic bytes


if __name__ == "__main__":
    unittest.main()
