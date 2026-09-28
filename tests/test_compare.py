"""Tests for the LR / RF / XGBoost model comparison (src.nids.compare).

Fast tests build the candidate pipelines and run the full CV comparison on a
tiny synthetic dataset (same full 41-feature schema as the real data). Opt-in
tests check the real reports/ artifacts and full-run determinism — enable
with RUN_SLOW_EVAL_TESTS=1.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.nids import compare, config, evaluate
from src.nids.data import load_inference_data, load_training_data

# One dataset generator shared with the evaluate tests (no duplicate schema).
try:  # project-root invocation: python -m unittest tests.test_compare
    from tests.test_evaluate import _synthetic_training_csv
except ImportError:  # discover invocation: python -m unittest discover -s tests
    from test_evaluate import _synthetic_training_csv


class TestCandidatePipelines(unittest.TestCase):
    """XGBoost/LR construction and preprocessing compatibility."""

    def test_registry_has_all_three_candidates(self):
        registry = compare.candidate_registry()
        self.assertEqual(
            list(registry), ["lr_baseline", "rf_production", "xgb_candidate"]
        )
        for spec in registry.values():
            self.assertIn("factory", spec)
            self.assertIn("description", spec)

    @unittest.skipUnless(
        config.TRAIN_CSV.exists(), "requires data/Train_data.csv (gitignored)"
    )
    def test_all_candidates_use_the_shared_preprocessing_implementation(self):
        """Every candidate's preprocessing must be the build_pipeline() one.

        Candidates build their own fresh ColumnTransformer instance, so
        identity cannot hold — but the construction path is shared, which
        means an identical configuration (same repr for the same build).
        """
        reference = repr(
            compare.pipeline_module.build_pipeline().named_steps["preprocessing"]
        )
        for name, spec in compare.candidate_registry().items():
            pipeline = spec["factory"]()
            preprocessing = pipeline.named_steps["preprocessing"]
            self.assertEqual(type(preprocessing).__name__, "ColumnTransformer", name)
            self.assertEqual(repr(preprocessing), reference, name)
            self.assertEqual(preprocessing.remainder, "passthrough", name)

    @unittest.skipUnless(
        config.TRAIN_CSV.exists(), "requires data/Train_data.csv (gitignored)"
    )
    def test_xgb_construction(self):
        import xgboost

        pipeline = compare.build_xgb_pipeline()
        self.assertEqual([name for name, _ in pipeline.steps],
                         ["preprocessing", "classifier"])
        clf = pipeline.named_steps["classifier"]
        self.assertIsInstance(clf, xgboost.XGBClassifier)
        self.assertEqual(clf.n_estimators, config.XGB_PARAMS["n_estimators"])
        self.assertEqual(clf.learning_rate, config.XGB_PARAMS["learning_rate"])
        self.assertEqual(clf.max_depth, config.XGB_PARAMS["max_depth"])
        self.assertEqual(clf.random_state, config.RANDOM_STATE)
        self.assertEqual(clf.tree_method, "hist")
        # Mirrors RandomForest's class_weight='balanced' (n_normal > n_anomaly).
        self.assertGreater(clf.scale_pos_weight, 1.0)

    def test_lr_construction(self):
        pipeline = compare.build_lr_pipeline()
        self.assertEqual([name for name, _ in pipeline.steps],
                         ["preprocessing", "scaling", "classifier"])
        clf = pipeline.named_steps["classifier"]
        self.assertEqual(clf.max_iter, 2000)
        self.assertEqual(clf.class_weight, "balanced")
        self.assertEqual(clf.random_state, config.RANDOM_STATE)

    def test_rf_production_uses_approved_configuration(self):
        pipeline = compare.build_rf_pipeline()
        clf = pipeline.named_steps["classifier"]
        self.assertEqual(clf.n_estimators, config.RF_PARAMS["n_estimators"])
        self.assertEqual(clf.class_weight, config.RF_PARAMS["class_weight"])
        self.assertEqual(clf.random_state, config.RANDOM_STATE)

    @unittest.skipUnless(
        config.TRAIN_CSV.exists(), "requires data/Train_data.csv (gitignored)"
    )
    def test_candidates_fit_and_predict_on_training_schema(self):
        """Preprocessing compatibility: all three fit/predict the real schema."""
        X, y = load_training_data(config.TRAIN_CSV)
        idx = list(y[y == 0].index[:25]) + list(y[y == 1].index[:25])
        sample, sample_y = X.loc[idx], y.loc[idx]
        for name, spec in compare.candidate_registry().items():
            pipeline = spec["factory"]()
            pipeline.fit(sample, sample_y)
            preds = np.asarray(pipeline.predict(sample), dtype=int)
            self.assertEqual(len(preds), 50, name)
            self.assertTrue(set(preds) <= {0, 1}, name)


class TestCompareOnSyntheticData(unittest.TestCase):
    """Full comparison run on the synthetic dataset (paths patched)."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = Path(_synthetic_training_csv())
        with mock.patch.object(config, "TRAIN_CSV", cls.csv_path):
            cls.results = compare.compare_models()
            cls.report = compare.build_report(
                cls.results, n_splits=5, random_state=config.RANDOM_STATE
            )

    def test_all_three_models_evaluated(self):
        self.assertEqual(list(self.results),
                         ["lr_baseline", "rf_production", "xgb_candidate"])
        for res in self.results.values():
            for metric in evaluate.METRIC_KEYS:
                self.assertEqual(len(res["fold_metrics"][metric]), 5)

    def test_report_schema(self):
        self.assertEqual(self.report["report"], "nids_model_comparison")
        self.assertEqual(self.report["schema_version"], "1.0")
        evaluation = self.report["evaluation"]
        self.assertEqual(
            evaluation["evaluation_population"],
            "5-fold StratifiedKFold on labelled training data",
        )
        self.assertIn("Test_data.csv", evaluation["note"])
        self.assertIn("NOT evaluated", evaluation["note"])
        cv = evaluation["cross_validation"]
        self.assertEqual(cv["n_splits"], 5)
        self.assertTrue(cv["shuffle"])
        self.assertEqual(cv["random_state"], 42)
        self.assertTrue(cv["identical_folds_for_all_candidates"])

    def test_report_declares_no_winner(self):
        selection = self.report["model_selection"]
        self.assertEqual(selection["decision"], "not_declared")
        self.assertIn("RandomForest", selection["note"])

    def test_each_model_has_configuration_and_mean_std(self):
        for name, res in self.results.items():
            self.assertIn("configuration", res, name)
            self.assertIn("classifier", res["configuration"])
            self.assertIn("preprocessing", res["configuration"])
            for metric in evaluate.METRIC_KEYS:
                block = res["summary"][metric]
                self.assertEqual(set(block), {"mean", "std", "folds"})
                self.assertEqual(len(block["folds"]), 5)
                for value in block["folds"]:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                self.assertGreaterEqual(block["mean"], min(block["folds"]))
                self.assertLessEqual(block["mean"], max(block["folds"]))

    def test_confusion_matrices_are_consistent(self):
        total_rows = self.report["dataset"]["shape"]["rows"]
        for name, res in self.results.items():
            cm = np.asarray(res["confusion_matrix"]["out_of_fold_counts"])
            self.assertEqual(cm.shape, (2, 2), name)
            self.assertEqual(int(cm.sum()), total_rows, name)
            acc_from_cm = float(np.trace(cm)) / cm.sum()
            self.assertAlmostEqual(
                acc_from_cm, res["summary"]["accuracy"]["mean"], delta=0.005, msg=name
            )

    def test_synthetic_scores_are_high_for_separable_data(self):
        for name in ("rf_production", "xgb_candidate"):
            self.assertGreaterEqual(
                self.results[name]["summary"]["f1_anomaly"]["mean"], 0.95, name
            )

    def test_xgb_configuration_recorded_in_report(self):
        cfg = self.results["xgb_candidate"]["configuration"]["classifier_params"]
        self.assertEqual(cfg["n_estimators"], 200)
        self.assertEqual(cfg["random_state"], 42)
        self.assertIn("scale_pos_weight", cfg)

    @unittest.skipUnless(
        config.TRAIN_CSV.exists(),
        "requires data/Train_data.csv (xgb scale_pos_weight)",
    )
    def test_deterministic_fold_metrics_on_synthetic_data(self):
        """Same data + seed twice -> identical fold metrics (RF and XGB)."""
        X, y = load_training_data(self.csv_path)
        for name in ("rf_production", "xgb_candidate"):
            factory = compare.candidate_registry()[name]["factory"]
            folds_a, _ = evaluate.run_cv_folds(factory, X, y)
            folds_b, _ = evaluate.run_cv_folds(factory, X, y)
            self.assertEqual(folds_a, folds_b, name)

    def test_csv_export_shape(self):
        frame = compare.comparison_csv(self.results)
        self.assertEqual(list(frame["model"]),
                         ["lr_baseline", "rf_production", "xgb_candidate"])
        for metric in evaluate.METRIC_KEYS:
            self.assertIn(f"{metric}_mean", frame.columns)
            self.assertIn(f"{metric}_std", frame.columns)


class TestNoTestDatasetLeakage(unittest.TestCase):
    """The unlabelled Test_data.csv must never be evaluated as labelled."""

    def test_loader_rejects_class_column(self):
        """load_inference_data must refuse any file that carries labels."""
        df = pd.DataFrame(
            {"duration": [0, 1], "protocol_type": ["tcp", "udp"], "class": ["normal", "anomaly"]}
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            path = fh.name
        try:
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_inference_data(Path(path))
        finally:
            os.unlink(path)

    def test_compare_never_reads_test_csv(self):
        """compare_models must not read Test_data.csv at all."""
        csv_path = Path(_synthetic_training_csv())
        real_read_csv = pd.read_csv

        def guarded_read_csv(path, *args, **kwargs):
            if "Test_data" in str(path):
                raise AssertionError(f"Test_data.csv was read: {path}")
            return real_read_csv(path, *args, **kwargs)

        with mock.patch.object(config, "TRAIN_CSV", csv_path), \
                mock.patch("src.nids.data.pd.read_csv", side_effect=guarded_read_csv):
            results = compare.compare_models()
        self.assertEqual(list(results),
                         ["lr_baseline", "rf_production", "xgb_candidate"])


class TestGeneratedComparisonArtifacts(unittest.TestCase):
    """The real reports/ artifacts, when they have been generated."""

    @classmethod
    def setUpClass(cls):
        cls.json_path = compare.COMPARISON_JSON_PATH
        cls.csv_path = compare.COMPARISON_CSV_PATH

    def test_generated_comparison_json_is_valid_and_labelled(self):
        if not self.json_path.exists():
            self.skipTest("run 'python -m src.nids.compare' to generate reports/")
        report = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(report["report"], "nids_model_comparison")
        self.assertEqual(
            report["evaluation"]["evaluation_population"],
            "5-fold StratifiedKFold on labelled training data",
        )
        self.assertEqual(list(report["models"]),
                         ["lr_baseline", "rf_production", "xgb_candidate"])
        self.assertEqual(report["model_selection"]["decision"], "not_declared")
        self.assertIn("xgboost", report["environment"])
        self.assertIsNotNone(report["excluded_unlabelled_dataset"])
        self.assertFalse(report["excluded_unlabelled_dataset"]["has_class_column"])

    def test_generated_comparison_csv_rows(self):
        if not self.csv_path.exists():
            self.skipTest("run 'python -m src.nids.compare' to generate reports/")
        frame = pd.read_csv(self.csv_path)
        self.assertEqual(list(frame["model"]),
                         ["lr_baseline", "rf_production", "xgb_candidate"])

    def test_full_run_determinism_against_real_dataset(self):
        """Opt-in: two full comparison runs must produce identical metrics."""
        if os.environ.get("RUN_SLOW_EVAL_TESTS") != "1":
            self.skipTest("set RUN_SLOW_EVAL_TESTS=1 to run the full determinism check")
        results_a = compare.compare_models()
        results_b = compare.compare_models()
        for name in results_a:
            self.assertEqual(results_a[name]["fold_metrics"],
                             results_b[name]["fold_metrics"], name)


if __name__ == "__main__":
    unittest.main()
