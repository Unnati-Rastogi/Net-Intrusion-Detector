"""Tests for the SHAP explanation layer (src.nids.explain).

Fast tests run the full explanation flow on a tiny synthetic dataset (same
41-feature schema as the real data) with a small background sample. Tests
that need the REAL saved artifact (116-feature mapping, artifact integrity)
load it without computing any SHAP values. The opt-in slow test regenerates
the real explanation report behind RUN_SLOW_EVAL_TESTS=1.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.nids import config, explain, evaluate
from src.nids.data import load_training_data
from src.nids.pipeline import build_pipeline
from src.nids.predict import load_model

try:  # project-root invocation: python -m unittest tests.test_explain
    from tests.test_evaluate import _synthetic_training_csv
except ImportError:  # discover invocation: python -m unittest discover -s tests
    from test_evaluate import _synthetic_training_csv

SMALL_BACKGROUND = 40


def _fitted_synthetic_context(background_size: int = SMALL_BACKGROUND):
    """Fit the real pipeline on synthetic data and build an explanation context.

    No retraining of the production model happens: the fit here creates a
    separate in-memory pipeline for tests only; the saved artifact is never
    touched (verified separately).
    """
    csv_path = Path(_synthetic_training_csv())
    X, y = load_training_data(csv_path)
    fitted = build_pipeline().fit(X, y)
    context = explain.build_context(
        model=fitted, training_data=(X, y), background_size=background_size
    )
    return context, X, y, csv_path


class TestShapDependency(unittest.TestCase):
    """SHAP must be importable and recorded in the context."""

    def test_shap_importable_with_version(self):
        self.assertIsInstance(shap.__version__, str)
        self.assertGreaterEqual(
            tuple(int(p) for p in shap.__version__.split(".")[:2]), (0, 45)
        )

    def test_explainer_construction(self):
        context, _, _, _ = _fitted_synthetic_context()
        self.assertIsInstance(context.explainer, shap.TreeExplainer)
        self.assertEqual(context.explainer.feature_perturbation, "interventional")
        self.assertIsNotNone(context.explainer.expected_value)


class TestFeatureMapping(unittest.TestCase):
    """The 116 -> 41 mapping: complete, unique, correct aggregation groups."""

    @classmethod
    def setUpClass(cls):
        # Real saved artifact, preprocessing only (no SHAP computation).
        cls.real_pre = load_model().named_steps["preprocessing"]
        cls.real_mapping = explain.build_feature_mapping(cls.real_pre)
        cls.real_names = explain.transformed_feature_names(cls.real_pre)

    def test_transformed_feature_count_is_116_for_current_pipeline(self):
        self.assertEqual(len(self.real_names), 116)
        one_hot = {
            column: sum(
                1 for n in self.real_names if n.startswith(f"{column}_")
            )
            for column in config.CATEGORICAL_COLUMNS
        }
        self.assertEqual(one_hot["protocol_type"], 3)
        self.assertEqual(one_hot["service"], 66)
        self.assertEqual(one_hot["flag"], 11)
        numeric = len(self.real_names) - sum(one_hot.values())
        self.assertEqual(numeric, 36)  # 41 - 3 categorical - 2 dropped constants

    def test_complete_mapping_all_transformed_accounted(self):
        claimed = sorted(i for idxs in self.real_mapping.values() for i in idxs)
        self.assertEqual(claimed, list(range(116)))

    def test_every_transformed_maps_to_exactly_one_original(self):
        for original, idxs in self.real_mapping.items():
            for idx in idxs:
                name = self.real_names[idx]
                is_owner = name == original or name.startswith(f"{original}_")
                self.assertTrue(is_owner, f"{name} mapped under {original}")

    def test_no_duplicate_or_missing_original_features(self):
        self.assertEqual(len(self.real_mapping), 41)
        self.assertEqual(set(self.real_mapping), set(config.FEATURE_COLUMNS))
        # Dict keys cannot duplicate; ensure no original feature is missing.
        self.assertEqual(
            sorted(self.real_mapping), sorted(config.FEATURE_COLUMNS)
        )

    def test_one_hot_groups_are_correctly_aggregated(self):
        for column in config.CATEGORICAL_COLUMNS:
            expected = sorted(
                i for i, n in enumerate(self.real_names)
                if n.startswith(f"{column}_")
            )
            self.assertEqual(sorted(self.real_mapping[column]), expected,
                             column)
            self.assertGreater(len(self.real_mapping[column]), 1, column)

    def test_passthrough_features_map_one_to_one(self):
        numeric = [
            f for f in config.FEATURE_COLUMNS
            if f not in config.CATEGORICAL_COLUMNS
            and f not in config.CONSTANT_COLUMNS
        ]
        self.assertEqual(len(numeric), 36)
        for feature in numeric:
            self.assertEqual(len(self.real_mapping[feature]), 1, feature)
            self.assertEqual(
                self.real_names[self.real_mapping[feature][0]], feature
            )

    def test_dropped_constants_map_to_nothing(self):
        self.assertEqual(
            sorted(self.real_mapping[c] for c in config.CONSTANT_COLUMNS),
            [[], []],
        )


class TestExplainSamples(unittest.TestCase):
    """Schema, top_k, determinism, and additivity on the synthetic dataset."""

    @classmethod
    def setUpClass(cls):
        cls.context, cls.X, cls.y, _ = _fitted_synthetic_context()
        cls.samples = pd.concat(
            [cls.X[cls.y == 0].head(3), cls.X[cls.y == 1].head(3)]
        )

    def test_output_schema(self):
        results = explain.explain_samples(self.context, self.samples)
        self.assertEqual(len(results), 6)
        for row in results:
            self.assertEqual(
                set(row),
                {"index", "source_row_id", "prediction", "anomaly_score",
                 "base_value", "top_features", "additivity_error",
                 "additivity_note"},
            )
            self.assertIn(row["prediction"], {"anomaly", "normal"})
            for contribution in row["top_features"]:
                self.assertEqual(
                    set(contribution),
                    {"feature", "value", "shap_contribution", "direction",
                     "abs_contribution"},
                )

    def test_top_k_default_is_10(self):
        results = explain.explain_samples(self.context, self.samples)
        self.assertEqual(len(results[0]["top_features"]), 10)

    def test_top_k_configurable_and_none_returns_all(self):
        three = explain.explain_samples(self.context, self.samples, top_k=3)
        self.assertEqual(len(three[0]["top_features"]), 3)
        everything = explain.explain_samples(self.context, self.samples, top_k=None)
        self.assertEqual(len(everything[0]["top_features"]), 41)

    def test_sorted_by_absolute_magnitude(self):
        results = explain.explain_samples(self.context, self.samples)
        for row in results:
            absolutes = [c["abs_contribution"] for c in row["top_features"]]
            self.assertEqual(absolutes, sorted(absolutes, reverse=True))

    def test_additivity_and_anomaly_class_attribution(self):
        results = explain.explain_samples(self.context, self.samples)
        full = explain.explain_samples(self.context, self.samples, top_k=None)
        scores = self.context.pipeline.predict_proba(self.samples)[:, 1]
        for i, row in enumerate(results):
            self.assertLess(row["additivity_error"], 1e-6)
            self.assertAlmostEqual(row["anomaly_score"], float(scores[i]),
                                   places=9)
            # Additivity holds over ALL 41 features (the top_k view is a
            # truncation, so only the full feature set reconstructs the score).
            total = full[i]["base_value"] + sum(
                c["shap_contribution"] for c in full[i]["top_features"]
            )
            self.assertAlmostEqual(total, row["anomaly_score"], places=6)
            # The explained axis is P(anomaly): prediction derives from it.
            self.assertEqual(
                row["prediction"], "anomaly" if scores[i] >= 0.5 else "normal"
            )

    def test_direction_labels_consistent(self):
        results = explain.explain_samples(self.context, self.samples, top_k=None)
        for row in results:
            for c in row["top_features"]:
                expected = ("increases_anomaly_score"
                            if c["shap_contribution"] >= 0
                            else "decreases_anomaly_score")
                self.assertEqual(c["direction"], expected)
                self.assertAlmostEqual(
                    c["abs_contribution"], abs(c["shap_contribution"]), places=12
                )

    def test_one_hot_aggregation_sum_matches_raw_shap_values(self):
        raw = np.asarray(
            self.context.explainer.shap_values(
                self.context.preprocessing.transform(self.samples)
            )
        )[:, :, 1]
        for column in config.CATEGORICAL_COLUMNS:
            idxs = self.context.mapping[column]
            expected = raw[:, idxs].sum(axis=1)
            results = explain.explain_samples(
                self.context, self.samples, top_k=None
            )
            for i, row in enumerate(results):
                contribution = next(
                    c for c in row["top_features"] if c["feature"] == column
                )
                self.assertAlmostEqual(
                    contribution["shap_contribution"], float(expected[i]),
                    places=9, msg=column,
                )

    def test_dropped_constants_contribute_exactly_zero(self):
        results = explain.explain_samples(self.context, self.samples, top_k=None)
        for row in results:
            for c in row["top_features"]:
                if c["feature"] in config.CONSTANT_COLUMNS:
                    self.assertEqual(c["shap_contribution"], 0.0)

    def test_values_finite(self):
        results = explain.explain_samples(self.context, self.samples, top_k=None)
        for row in results:
            self.assertTrue(np.isfinite(row["anomaly_score"]))
            self.assertTrue(np.isfinite(row["base_value"]))
            for c in row["top_features"]:
                self.assertTrue(np.isfinite(c["shap_contribution"]))
                self.assertTrue(np.isfinite(c["abs_contribution"]))

    def test_deterministic_with_fixed_background(self):
        again, _, _, _ = _fitted_synthetic_context()
        first = explain.explain_samples(self.context, self.samples)
        second = explain.explain_samples(again, self.samples)
        self.assertEqual(first, second)

    def test_empty_input_returns_empty(self):
        self.assertEqual(explain.explain_samples(self.context, self.X.head(0)), [])


class TestSampleRowIdentity(unittest.TestCase):
    """sample_id (frame position) vs source_row_id (source row) semantics."""

    @classmethod
    def setUpClass(cls):
        csv_path = Path(_synthetic_training_csv())
        X, y = load_training_data(csv_path)
        fitted = build_pipeline().fit(X, y)
        cls.context = explain.build_context(
            model=fitted, training_data=(X, y), background_size=20
        )
        cls.X, cls.y = X, y
        # A report-style sample: positions spread through the source frame.
        cls.sample_idx = explain.background_sample_indices(len(X), 40, 42)
        cls.sample_X = X.iloc[cls.sample_idx]
        cls.results = explain.explain_samples(cls.context, cls.sample_X,
                                              top_k=None)

    def test_sample_id_is_position_within_sample(self):
        self.assertEqual([r["index"] for r in self.results],
                         list(range(len(self.results))))

    def test_source_row_id_equals_source_index_labels(self):
        self.assertEqual([r["source_row_id"] for r in self.results],
                         [int(v) for v in self.sample_idx])

    def test_source_row_id_not_reindexed_positions(self):
        # Data-driven: the sample's source rows are a spread draw, so they
        # cannot all coincide with the frame positions 0..N-1.
        self.assertEqual(self.results[0]["index"], 0)
        self.assertEqual(self.results[0]["source_row_id"],
                         int(self.sample_idx[0]))
        self.assertNotEqual(self.results[0]["index"],
                            self.results[0]["source_row_id"])
        self.assertEqual(self.results[-1]["source_row_id"],
                         int(self.sample_idx[-1]))
        self.assertTrue(
            any(r["index"] != r["source_row_id"] for r in self.results),
            "sample appears re-indexed: source_row_id equals frame position",
        )

    def test_source_row_id_values_match_source_rows_exactly(self):
        for row in self.results:
            src = self.X.iloc[row["source_row_id"]]
            for contribution in row["top_features"]:
                feature, value = contribution["feature"], contribution["value"]
                if pd.isna(value):
                    self.assertTrue(pd.isna(src[feature]))
                else:
                    self.assertEqual(
                        str(src[feature]), str(value),
                        f"row {row['source_row_id']} feature {feature}",
                    )

    def test_non_default_integer_index_is_source_row(self):
        rows = self.X.iloc[[4, 5, 6, 7]]
        results = explain.explain_samples(self.context, rows, top_k=1)
        self.assertEqual([r["index"] for r in results], [0, 1, 2, 3])
        self.assertEqual([r["source_row_id"] for r in results], [4, 5, 6, 7])

    def test_source_row_id_is_json_safe_int(self):
        for row in self.results:
            self.assertIsInstance(row["source_row_id"], int)

    def test_mapping_is_deterministic(self):
        again = explain.explain_samples(self.context, self.sample_X, top_k=None)
        self.assertEqual(
            [(r["index"], r["source_row_id"]) for r in self.results],
            [(r["index"], r["source_row_id"]) for r in again],
        )

    def test_report_frame_lineage_survives_report_assembly(self):
        report, _ = explain.build_explanation_report(
            self.context, self.X, self.y, sample_size=40, sample_seed=42
        )
        embedded = report["sample_explanations"]
        expected_rows = explain.background_sample_indices(
            len(self.X), min(40, len(self.X)), 42
        )
        self.assertEqual([r["index"] for r in embedded], list(range(5)))
        self.assertEqual([r["source_row_id"] for r in embedded],
                         [int(v) for v in expected_rows[:5]])
        for row in embedded:
            src = self.X.iloc[row["source_row_id"]]
            for contribution in row["top_features"]:
                self.assertEqual(
                    str(src[contribution["feature"]]),
                    str(contribution["value"]),
                )


class TestNoRetrainingAndNoTestCSV(unittest.TestCase):
    """The explain layer must be read-only w.r.t. model and datasets."""

    def test_explanation_never_calls_fit(self):
        context, X, _, _ = _fitted_synthetic_context()
        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")), \
             mock.patch.object(RandomForestClassifier, "fit",
                               side_effect=AssertionError("RF.fit called")):
            results = explain.explain_samples(context, X.head(5))
        self.assertEqual(len(results), 5)

    def test_build_context_with_provided_model_does_not_fit(self):
        csv_path = Path(_synthetic_training_csv())
        X, y = load_training_data(csv_path)
        fitted = build_pipeline().fit(X, y)  # fitting happens OUTSIDE explain
        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")):
            explain.build_context(model=fitted, training_data=(X, y),
                                  background_size=20)

    def test_saved_artifact_not_modified_by_explanation(self):
        sha_before = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        context = explain.build_context(background_size=20)  # saved artifact
        self.assertTrue(context.model_hash)
        sha_after = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)

    def test_explanation_flow_never_reads_test_data_csv(self):
        csv_path = Path(_synthetic_training_csv())
        real_read_csv = pd.read_csv

        def guarded_read_csv(path, *args, **kwargs):
            if "Test_data" in str(path):
                raise AssertionError(f"Test_data.csv was read: {path}")
            return real_read_csv(path, *args, **kwargs)

        X, y = load_training_data(csv_path)
        fitted = build_pipeline().fit(X, y)
        with mock.patch.object(config, "TRAIN_CSV", csv_path), \
                mock.patch("src.nids.data.pd.read_csv",
                           side_effect=guarded_read_csv), \
                mock.patch("src.nids.evaluate.pd.read_csv",
                           side_effect=guarded_read_csv):
            context = explain.build_context(
                model=fitted, training_data=(X, y), background_size=20
            )
            results = explain.explain_samples(context, X.head(5))
        self.assertEqual(len(results), 5)


class TestExplanationReport(unittest.TestCase):
    """Report assembly: schema, provenance, and global-importance math."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = Path(_synthetic_training_csv())
        X, y = load_training_data(cls.csv_path)
        fitted = build_pipeline().fit(X, y)
        with mock.patch.object(config, "TRAIN_CSV", cls.csv_path):
            cls.context = explain.build_context(
                model=fitted, training_data=(X, y),
                background_size=SMALL_BACKGROUND,
            )
            cls.report, cls.importance = explain.build_explanation_report(
                cls.context, X, y, sample_size=100
            )

    def test_report_schema_and_provenance(self):
        r = self.report
        self.assertEqual(r["report"], explain.REPORT_NAME)
        self.assertEqual(r["schema_version"], "1.0")
        self.assertIn("timestamp_utc", r)
        self.assertIn("training", r["evaluation_population"])
        model = r["model"]
        self.assertFalse(model["retrained"])
        self.assertEqual(model["classifier"], "RandomForestClassifier")
        self.assertEqual(model["n_estimators"], 200)
        self.assertEqual(model["random_state"], 42)
        shap_block = r["shap"]
        self.assertEqual(shap_block["explainer"], "shap.TreeExplainer")
        self.assertEqual(shap_block["feature_perturbation"], "interventional")
        self.assertIn("anomaly", shap_block["explained_output"])
        self.assertEqual(shap_block["background"]["size"], SMALL_BACKGROUND)
        self.assertEqual(shap_block["background"]["seed"], 42)
        self.assertTrue(shap_block["background"]["rows_are_training_only"])
        self.assertLess(shap_block["additivity_max_error"], 1e-6)
        self.assertIn("causal", shap_block["values_meaning"].lower())
        self.assertIn("shap", r["environment"])
        self.assertIn("git_revision", r["provenance"])

    def test_feature_mapping_block(self):
        block = self.report["feature_mapping"]
        self.assertEqual(block["original_feature_count"], 41)
        self.assertEqual(
            block["dropped_constant_features"],
            sorted(config.CONSTANT_COLUMNS),
        )
        total_members = sum(
            g["transformed_count"] for g in block["groups"].values()
        )
        self.assertEqual(total_members, block["transformed_feature_count"])

    def test_global_importance_calculated_correctly(self):
        # Recompute mean |SHAP| for key features straight from the explainer.
        X, _ = load_training_data(self.csv_path)
        sample_X, _ = explain._report_sample_frame(X, 100, 42)
        raw = np.asarray(self.context.explainer.shap_values(
            self.context.preprocessing.transform(sample_X)
        ))[:, :, 1]
        importance = {row["feature"]: row["mean_abs_shap"]
                      for row in self.report["global_importance"]}
        for feature in ("src_bytes", "dst_bytes", "flag"):
            expected = float(np.abs(raw[:, self.context.mapping[feature]]
                                    .sum(axis=1)).mean())
            self.assertAlmostEqual(importance[feature], expected, places=9,
                                   msg=feature)

    def test_global_importance_sorted_and_normalized(self):
        importance = self.report["global_importance"]
        self.assertEqual(len(importance), 41)
        values = [row["mean_abs_shap"] for row in importance]
        self.assertEqual(values, sorted(values, reverse=True))
        shares = [row["share_of_total"] for row in importance]
        self.assertAlmostEqual(sum(shares), 1.0, places=9)

    def test_per_class_importance_present(self):
        per_class = self.report["per_class_mean_abs_shap"]
        self.assertEqual(set(per_class), {"anomaly", "normal"})
        for label, block in per_class.items():
            self.assertEqual(len(block), 41, label)

    def test_methodology_and_limitations(self):
        self.assertIn("causal", self.report["explanation_methodology"].lower())
        self.assertIn("interventional", self.report["explanation_methodology"])
        self.assertIsInstance(self.report["limitations"], list)
        self.assertGreaterEqual(len(self.report["limitations"]), 3)

    def test_sample_explanations_embedded(self):
        samples = self.report["sample_explanations"]
        self.assertEqual(len(samples), 5)
        for row in samples:
            self.assertIn("prediction", row)
            self.assertLessEqual(len(row["top_features"]),
                                 explain.TOP_K_DEFAULT)


class TestGeneratedExplanationsArtifact(unittest.TestCase):
    """The real reports/explanations.json, when it has been generated."""

    @classmethod
    def setUpClass(cls):
        cls.path = explain.EXPLANATIONS_JSON_PATH

    def test_generated_report_schema(self):
        if not self.path.exists():
            self.skipTest("run 'python -m src.nids.explain' to generate reports/")
        r = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(r["report"], explain.REPORT_NAME)
        self.assertEqual(r["shap"]["feature_perturbation"], "interventional")
        self.assertRegex(r["model"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(r["model"]["retrained"])
        self.assertEqual(r["feature_mapping"]["transformed_feature_count"], 116)
        self.assertEqual(r["feature_mapping"]["explained_feature_count"], 39)
        self.assertEqual(r["feature_mapping"]["dropped_constant_features"],
                         sorted(config.CONSTANT_COLUMNS))
        self.assertEqual(len(r["global_importance"]), 41)
        top = r["global_importance"][0]
        self.assertTrue(
            all(row["mean_abs_shap"] <= top["mean_abs_shap"]
                for row in r["global_importance"])
        )
        self.assertIn("causal", r["explanation_methodology"].lower())
        self.assertIsNotNone(r["excluded_unlabelled_dataset"])
        self.assertFalse(r["excluded_unlabelled_dataset"]["has_class_column"])

    def test_generated_report_population_is_training_distribution(self):
        if not self.path.exists():
            self.skipTest("run 'python -m src.nids.explain' to generate reports/")
        r = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("Train_data.csv", r["evaluation_population"])
        self.assertIn("training", r["evaluation_population"])


class TestSlowRealDataShap(unittest.TestCase):
    """Opt-in: regenerate the real report end-to-end (a few minutes)."""

    def test_full_explanation_report_run(self):
        if os.environ.get("RUN_SLOW_EVAL_TESTS") != "1":
            self.skipTest("set RUN_SLOW_EVAL_TESTS=1 to run the full SHAP report")
        report = explain.main()
        self.assertEqual(report["feature_mapping"]["transformed_feature_count"], 116)
        self.assertEqual(len(report["global_importance"]), 41)
        self.assertLess(report["shap"]["additivity_max_error"], 1e-6)
        self.assertGreaterEqual(report["shap"]["additivity_tie_samples"], 0)
        self.assertIn("one_vote_split_threshold_tie",
                      report["shap"]["additivity_note_definition"])


if __name__ == "__main__":
    unittest.main()
