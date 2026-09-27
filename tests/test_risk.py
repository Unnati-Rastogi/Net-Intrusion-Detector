"""Tests for the transparent risk engine (src.nids.risk).

Pure-engine tests need no model or SHAP; integration tests run the risk
layer over real explain_samples() output on a tiny synthetic dataset. One
opt-in test regenerates the full real-data report behind
RUN_SLOW_EVAL_TESTS=1.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.nids import config, explain, risk
from src.nids.data import load_training_data
from src.nids.pipeline import build_pipeline

try:  # project-root invocation: python -m unittest tests.test_risk
    from tests.test_evaluate import _synthetic_training_csv
except ImportError:  # discover invocation: python -m unittest discover -s tests
    from test_evaluate import _synthetic_training_csv


class TestConsensusRatio(unittest.TestCase):
    def test_bounds_and_extremes(self):
        self.assertEqual(risk.consensus_ratio(1.0, 1.0), 1.0)
        self.assertEqual(risk.consensus_ratio(-1.0, 1.0), -1.0)
        self.assertEqual(risk.consensus_ratio(0.0, 5.0), 0.0)

    def test_zero_mass_yields_zero(self):
        self.assertEqual(risk.consensus_ratio(0.0, 0.0), 0.0)
        self.assertEqual(risk.consensus_ratio(0.3, 0.0), 0.0)

    def test_clips_floating_point_residue_only(self):
        self.assertEqual(risk.consensus_ratio(1.0000001, 1.0), 1.0)
        self.assertEqual(risk.consensus_ratio(-1.0000001, 1.0), -1.0)


class TestEvidenceScore(unittest.TestCase):
    def test_aggregation_of_known_contributions(self):
        block = risk.evidence_score(0.8, [0.10, -0.03, 0.05, 0.02])
        self.assertAlmostEqual(block["net_contribution"], 0.14, places=12)
        self.assertAlmostEqual(block["abs_mass"], 0.20, places=12)
        self.assertAlmostEqual(block["consensus_ratio"], 0.70, places=12)

    def test_additivity_consistency_passes_for_full_lists(self):
        base, s = 0.51715, 0.8
        contribs = [0.2, -0.05, 0.1, s - base - 0.25]
        block = risk.evidence_score(s, contribs, base_value=base)
        self.assertAlmostEqual(block["net_contribution"], s - base, places=12)

    def test_partial_contribution_list_is_rejected_with_base(self):
        with self.assertRaises(ValueError):
            risk.evidence_score(0.8, [0.1], base_value=0.5)

    def test_base_omitted_skips_consistency_check(self):
        block = risk.evidence_score(0.8, [0.1])
        self.assertAlmostEqual(block["net_contribution"], 0.1)

    def test_malformed_inputs_raise(self):
        with self.assertRaises(ValueError):
            risk.evidence_score(float("nan"), [0.1])
        with self.assertRaises(ValueError):
            risk.evidence_score(1.5, [0.1])
        with self.assertRaises(ValueError):
            risk.evidence_score(-0.1, [0.1])
        with self.assertRaises(ValueError):
            risk.evidence_score(0.5, [float("inf")])
        with self.assertRaises(ValueError):
            risk.evidence_score(0.5, [float("nan")])

    def test_explicit_tolerance_band(self):
        # net = 0.005 vs expected 0.0 -> outside the strict band, inside
        # the one-vote band for a 200-tree forest.
        with self.assertRaises(ValueError):
            risk.evidence_score(0.5, [0.01, -0.005], base_value=0.5)
        block = risk.evidence_score(
            0.5, [0.01, -0.005], base_value=0.5,
            tolerance=risk.additivity_tolerance(200, tie=True),
        )
        self.assertAlmostEqual(block["net_contribution"], 0.005)


class TestAdditivityTolerance(unittest.TestCase):
    def test_strict_by_default(self):
        self.assertEqual(risk.additivity_tolerance(200, tie=False),
                         risk.CONSISTENCY_TOLERANCE)

    def test_tie_adds_exactly_one_vote(self):
        self.assertAlmostEqual(risk.additivity_tolerance(200, tie=True),
                               risk.CONSISTENCY_TOLERANCE + 1 / 200)

    def test_tie_without_n_estimators_is_rejected(self):
        with self.assertRaises(ValueError):
            risk.additivity_tolerance(None, tie=True)
        with self.assertRaises(ValueError):
            risk.additivity_tolerance(0, tie=True)


class TestClassifyRisk(unittest.TestCase):
    def test_level_boundaries(self):
        self.assertEqual(risk.classify_risk(0.0), "LOW")
        self.assertEqual(risk.classify_risk(19.999), "LOW")
        self.assertEqual(risk.classify_risk(20.0), "MEDIUM")
        self.assertEqual(risk.classify_risk(49.999), "MEDIUM")
        self.assertEqual(risk.classify_risk(50.0), "HIGH")
        self.assertEqual(risk.classify_risk(79.999), "HIGH")
        self.assertEqual(risk.classify_risk(80.0), "CRITICAL")
        self.assertEqual(risk.classify_risk(100.0), "CRITICAL")

    def test_non_finite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                risk.classify_risk(bad)


class TestCalculateRisk(unittest.TestCase):
    def test_bounds_across_the_whole_plane(self):
        for s in np.linspace(0.0, 1.0, 21):
            for alpha in np.linspace(-1.0, 1.0, 21):
                contribs = [0.05, 0.05 * alpha, -0.05]  # arbitrary signs
                result = risk.calculate_risk(float(s), contribs)
                self.assertGreaterEqual(result["risk_score"], 0.0)
                self.assertLessEqual(result["risk_score"], 100.0)

    def test_extreme_cases(self):
        # s=0 -> risk 0 regardless of evidence.
        self.assertEqual(risk.calculate_risk(0.0, [0.3, 0.2])["risk_score"], 0.0)
        # s=1, unanimous positive evidence -> 100.
        self.assertEqual(
            risk.calculate_risk(1.0, [0.2, 0.1], base_value=0.7)["risk_score"],
            100.0,
        )
        # A flagged score whose attributed evidence pushes the OTHER way is
        # only reachable through the tie-slack path (sklearn counts exactly
        # one tree differently on split-threshold ties): s=0.5, base=1.0,
        # net=-0.505 differs from s-base=-0.5 by exactly one 200-tree vote.
        # The evidence is unanimously negative -> risk 0.
        result = risk.calculate_risk(
            0.5, [-0.3, -0.2, -0.005], base_value=1.0,
            tolerance=risk.additivity_tolerance(200, tie=True),
        )
        self.assertEqual(result["evidence_score"], -1.0)
        self.assertEqual(result["risk_score"], 0.0)

    def test_empty_contributions(self):
        # No evidence at all: consensus 0, risk = 50 * s.
        result = risk.calculate_risk(0.6, [])
        self.assertEqual(result["evidence"]["abs_mass"], 0.0)
        self.assertEqual(result["evidence_score"], 0.0)
        self.assertAlmostEqual(result["risk_score"], 30.0)

    def test_deterministic_output(self):
        # contributions sum exactly to s - base (0.72 - 0.51715)
        args = (0.72, [0.11, -0.04, 0.06, 0.07285])
        first = risk.calculate_risk(*args, base_value=0.51715)
        second = risk.calculate_risk(*args, base_value=0.51715)
        self.assertEqual(first, second)

    def test_no_nan_or_inf_in_output(self):
        result = risk.calculate_risk(0.37, [0.02, -0.01, 0.0])
        for value in (
            result["anomaly_score"], result["risk_score"],
            result["evidence_score"], *result["evidence"].values(),
        ):
            self.assertTrue(np.isfinite(value))

    def test_output_schema_and_methodology(self):
        result = risk.calculate_risk(0.6, [0.1, 0.05])
        self.assertEqual(
            set(result),
            {"anomaly_score", "evidence", "evidence_score", "risk_score",
             "risk_level", "methodology", "prediction", "additivity_note"}
            - {"prediction", "additivity_note"},
        )
        methodology = result["methodology"]
        self.assertEqual(methodology["name"], risk.METHOD_NAME)
        self.assertEqual(methodology["version"], risk.METHOD_VERSION)
        self.assertEqual(methodology["formula"], risk.RISK_FORMULA)
        self.assertEqual(methodology["thresholds"], risk.THRESHOLDS)
        self.assertIn("NOT a probability", methodology["interpretation"])
        self.assertIn("prioritization", methodology["interpretation"])

    def test_levels_match_score(self):
        for s, alpha in ((0.1, 0.9), (0.45, 0.95), (0.66, 0.3), (0.9, 0.8)):
            result = risk.calculate_risk(s, [0.2 * (1 + alpha), 0.2 * (1 - alpha)])
            self.assertEqual(result["risk_level"],
                             risk.classify_risk(result["risk_score"]))


class TestRiskForExplanationRow(unittest.TestCase):
    """Malformed / edge behaviour of the explanation-row entry point."""

    @staticmethod
    def _row(s, base, contribs, note=None):
        ordered = sorted(contribs, key=abs, reverse=True)
        return {
            "index": 7,
            "prediction": "anomaly" if s >= 0.5 else "normal",
            "anomaly_score": s,
            "base_value": base,
            "top_features": [
                {"feature": f"f{i}", "value": 1,
                 "shap_contribution": c, "direction": "x",
                 "abs_contribution": abs(c)}
                for i, c in enumerate(ordered)
            ],
            "additivity_error": 0.0,
            "additivity_note": note,
        }

    def test_happy_path(self):
        row = self._row(0.8, 0.5, [0.2, 0.05, 0.05])
        result = risk.calculate_risk_for_explanation(row)
        self.assertEqual(result["prediction"], "anomaly")
        # s=0.8 with unanimous positive evidence -> exactly 80.0, the
        # CRITICAL boundary (>= 80).
        self.assertEqual(result["risk_score"], 80.0)
        self.assertEqual(result["risk_level"], "CRITICAL")

    def test_tie_row_requires_n_estimators(self):
        # sum = 0.005 vs expected 0.0 -> the one-vote tie signature.
        row = self._row(0.5, 0.5, [0.01, -0.005],
                        note=risk.TIE_NOTE)
        with self.assertRaises(ValueError):
            risk.calculate_risk_for_explanation(row)
        result = risk.calculate_risk_for_explanation(row, n_estimators=200)
        self.assertEqual(result["additivity_note"], risk.TIE_NOTE)

    def test_missing_keys_raise(self):
        with self.assertRaises(KeyError):
            risk.calculate_risk_for_explanation({"anomaly_score": 0.5})
        row = self._row(float("nan"), 0.5, [0.1])
        with self.assertRaises(ValueError):
            risk.calculate_risk_for_explanation(row)


class TestRiskWithRealExplanations(unittest.TestCase):
    """Integration: risk over genuine explain_samples() output (synthetic)."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = Path(_synthetic_training_csv())
        X, y = load_training_data(cls.csv_path)
        cls.fitted = build_pipeline().fit(X, y)
        cls.context = explain.build_context(
            model=cls.fitted, training_data=(X, y), background_size=20
        )
        cls.X, cls.y = X, y
        cls.rows = X.iloc[[0, 1, 2, -2, -1]]
        cls.explanations = explain.explain_samples(cls.context, cls.rows,
                                                   top_k=None)

    def test_risk_over_real_explanation_rows(self):
        results = [risk.calculate_risk_for_explanation(
            e, n_estimators=self.fitted.named_steps["classifier"].n_estimators
        ) for e in self.explanations]
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertGreaterEqual(result["risk_score"], 0.0)
            self.assertLessEqual(result["risk_score"], 100.0)
            self.assertIn(result["risk_level"], risk.LEVEL_ORDER)
            self.assertTrue(np.isfinite(result["risk_score"]))

    def test_prediction_consistency_with_score(self):
        for explanation in self.explanations:
            result = risk.calculate_risk_for_explanation(explanation)
            self.assertEqual(result["prediction"], explanation["prediction"])
            self.assertEqual(
                result["prediction"],
                "anomaly" if explanation["anomaly_score"] >= 0.5 else "normal",
            )

    def test_no_model_fit_calls_during_risk_calculation(self):
        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")), \
             mock.patch.object(RandomForestClassifier, "fit",
                               side_effect=AssertionError("RF.fit called")):
            for explanation in self.explanations:
                risk.calculate_risk_for_explanation(explanation)

    def test_risk_layer_never_reads_test_data_csv(self):
        real_read_csv = __import__("pandas").read_csv

        def guarded_read_csv(path, *args, **kwargs):
            if "Test_data" in str(path):
                raise AssertionError(f"Test_data.csv was read: {path}")
            return real_read_csv(path, *args, **kwargs)

        import pandas as pd
        with mock.patch.object(config, "TRAIN_CSV", self.csv_path), \
                mock.patch("src.nids.data.pd.read_csv",
                           side_effect=guarded_read_csv), \
                mock.patch("src.nids.evaluate.pd.read_csv",
                           side_effect=guarded_read_csv):
            context = explain.build_context(
                model=self.fitted, training_data=(self.X, self.y),
                background_size=20,
            )
            explanations = explain.explain_samples(context, self.X.head(3),
                                                   top_k=None)
            for explanation in explanations:
                risk.calculate_risk_for_explanation(explanation)

    def test_saved_artifact_untouched_by_risk_flow(self):
        sha_before = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        context = explain.build_context(background_size=20)
        X, _ = load_training_data(self.csv_path)
        explanations = explain.explain_samples(context, X.head(2), top_k=None)
        for explanation in explanations:
            risk.calculate_risk_for_explanation(explanation)
        sha_after = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)


class TestRiskReport(unittest.TestCase):
    """build_risk_report schema and aggregate consistency (synthetic)."""

    @classmethod
    def setUpClass(cls):
        csv_path = Path(_synthetic_training_csv())
        X, y = load_training_data(csv_path)
        fitted = build_pipeline().fit(X, y)
        with mock.patch.object(config, "TRAIN_CSV", csv_path):
            context = explain.build_context(
                model=fitted, training_data=(X, y), background_size=20
            )
            sample_X, _ = explain._report_sample_frame(X, 40, 42)
            explanations = explain.explain_samples(context, sample_X,
                                                   top_k=None)
            cls.report = risk.build_risk_report(
                context, X, y, sample_size=40, sample_seed=42,
                explanations=explanations,
            )

    def test_report_schema(self):
        r = self.report
        self.assertEqual(r["report"], risk.REPORT_NAME)
        self.assertEqual(r["schema_version"], "1.0")
        self.assertIn("training distribution", r["evaluation_population"])
        self.assertIn(risk.CAVEAT.split(".")[0], r["risk"]["methodology"]["caveat"])
        methodology = r["risk"]["methodology"]
        for key in ("name", "version", "formula", "components", "thresholds",
                    "threshold_methodology", "double_counting_analysis",
                    "caveat"):
            self.assertIn(key, methodology)
        self.assertIn("rank-identical", methodology["double_counting_analysis"])

    def test_provenance_blocks(self):
        r = self.report
        self.assertFalse(r["model"]["retrained"])
        self.assertEqual(r["model"]["classifier"], "RandomForestClassifier")
        self.assertRegex(r["dataset"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNotNone(r["excluded_unlabelled_dataset"])
        self.assertFalse(r["excluded_unlabelled_dataset"]["has_class_column"])
        self.assertIn("git_revision", r["provenance"])

    def test_per_sample_entries(self):
        samples = self.report["samples"]
        self.assertEqual(len(samples), 40)
        for sample in samples:
            self.assertEqual(
                set(sample),
                {"sample_id", "true_label", "prediction", "anomaly_score",
                 "evidence_score", "risk_score", "risk_level", "top_evidence"},
            )
            self.assertIn(sample["true_label"], {"normal", "anomaly"})
            self.assertGreaterEqual(sample["risk_score"], 0.0)
            self.assertLessEqual(sample["risk_score"], 100.0)
            self.assertEqual(sample["risk_level"],
                             risk.classify_risk(sample["risk_score"]))
            self.assertLessEqual(len(sample["top_evidence"]),
                                 risk.TOP_EVIDENCE_IN_REPORT)

    def test_level_counts_and_histogram_consistent(self):
        samples = self.report["samples"]
        counts = self.report["risk"]["levels"]["counts"]
        self.assertEqual(sum(counts.values()), len(samples))
        self.assertEqual(list(counts), list(risk.LEVEL_ORDER))
        recomputed = {level: 0 for level in risk.LEVEL_ORDER}
        for sample in samples:
            recomputed[sample["risk_level"]] += 1
        self.assertEqual(counts, recomputed)
        histogram = self.report["risk"]["score_distribution"][
            "histogram_binwidth_10"
        ]
        self.assertEqual(sum(histogram.values()), len(samples))

    def test_score_distribution_finite(self):
        dist = self.report["risk"]["score_distribution"]
        for key in ("min", "max", "mean", "median"):
            self.assertTrue(np.isfinite(dist[key]))
        self.assertLessEqual(dist["min"], dist["median"] <= dist["max"])


class TestGeneratedRiskArtifact(unittest.TestCase):
    def test_generated_risk_report(self):
        path = explain.REPORTS_DIR / "risk_assessment.json"
        if not path.exists():
            self.skipTest("run 'python -m src.nids.risk' to generate reports/")
        r = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(r["report"], risk.REPORT_NAME)
        self.assertIn("1000", r["evaluation_population"])
        self.assertFalse(r["model"]["retrained"])
        self.assertTrue(r["model"]["sha256"])
        self.assertEqual(r["risk"]["methodology"]["name"], risk.METHOD_NAME)
        self.assertIn("training distribution", r["risk"]["methodology"]["caveat"])
        counts = r["risk"]["levels"]["counts"]
        self.assertEqual(sum(counts.values()), len(r["samples"]))


class TestSlowRealDataRisk(unittest.TestCase):
    def test_full_risk_report_run(self):
        if os.environ.get("RUN_SLOW_EVAL_TESTS") != "1":
            self.skipTest("set RUN_SLOW_EVAL_TESTS=1 to run the full risk report")
        report = risk.main()
        self.assertEqual(len(report["samples"]), risk.REPORT_SAMPLE_SIZE)
        self.assertEqual(sum(report["risk"]["levels"]["counts"].values()),
                         risk.REPORT_SAMPLE_SIZE)


if __name__ == "__main__":
    unittest.main()
