"""Tests for deterministic alert prioritization (src.nids.prioritize).

The prioritizer consumes existing risk records; tests use small synthetic
records plus the real reports/risk_assessment.json content. One opt-in test
regenerates the artifact behind RUN_SLOW_EVAL_TESTS=1.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import copy
import json
import os
import unittest
from unittest import mock

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.nids import prioritize, risk
from src.nids.pipeline import build_pipeline


def make_record(
    sample_id: int,
    risk_score: float,
    anomaly_score: float | None = None,
    evidence_score: float = 0.0,
    risk_level: str | None = None,
    prediction: str | None = None,
    true_label: str | None = None,
    source_row_id: int | None = None,
) -> dict:
    """A valid risk record; level defaults to the score's true classification."""
    if anomaly_score is None:
        anomaly_score = risk_score / 100.0
    if risk_level is None:
        risk_level = risk.classify_risk(risk_score)
    if prediction is None:
        prediction = "anomaly" if anomaly_score >= 0.5 else "normal"
    record = {
        "sample_id": sample_id,
        "prediction": prediction,
        "anomaly_score": float(anomaly_score),
        "evidence_score": float(evidence_score),
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "top_evidence": [
            {"feature": "src_bytes", "value": 100,
             "shap_contribution": 0.1, "direction": "increases_anomaly_score",
             "abs_contribution": 0.1},
        ],
    }
    if true_label is not None:
        record["true_label"] = true_label
    if source_row_id is not None:
        record["source_row_id"] = source_row_id
    return record


class TestOrdering(unittest.TestCase):
    def test_risk_score_descending(self):
        records = [make_record(i, s) for i, s in enumerate([30.0, 97.0, 5.0, 80.0, 42.0])]
        result = prioritize.prioritize_alerts(records, top_n=None)
        scores = [a["risk_score"] for a in result["ordered_alerts"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result["ordered_alerts"][0]["risk_score"], 97.0)

    def test_tie_on_risk_broken_by_anomaly_score(self):
        records = [
            make_record(1, 80.0, anomaly_score=0.8),
            make_record(2, 80.0, anomaly_score=0.95),
            make_record(3, 80.0, anomaly_score=0.7),
        ]
        result = prioritize.prioritize_alerts(records, top_n=None)
        self.assertEqual([a["sample_id"] for a in result["ordered_alerts"]],
                         [2, 1, 3])

    def test_tie_on_both_broken_by_sample_id(self):
        records = [
            make_record(9, 80.0, anomaly_score=0.8),
            make_record(4, 80.0, anomaly_score=0.8),
            make_record(6, 80.0, anomaly_score=0.8),
        ]
        result = prioritize.prioritize_alerts(records, top_n=None)
        self.assertEqual([a["sample_id"] for a in result["ordered_alerts"]],
                         [4, 6, 9])

    def test_level_rank_tiebreaker_map_is_explicit(self):
        # Rule 3 is explicit and documented; because risk_level must equal
        # risk.classify_risk(risk_score) (validated), it never reorders
        # consistent records — verify the map itself:
        self.assertEqual(prioritize.LEVEL_RANK,
                         {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1})
        # ...and that validation rejects records where it would be needed.
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts(
                [make_record(1, 80.0, risk_level="LOW")], top_n=None
            )

    def test_deterministic_output(self):
        records = [make_record(i, (i * 37) % 100) for i in range(50)]
        first = prioritize.prioritize_alerts(records, top_n=None)
        second = prioritize.prioritize_alerts(records, top_n=None)
        self.assertEqual(first, second)


class TestPriorityMapping(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(prioritize.PRIORITY_MAP,
                         {"CRITICAL": "P1", "HIGH": "P2",
                          "MEDIUM": "P3", "LOW": "P4"})

    def test_each_level_lands_in_its_bucket(self):
        records = [
            make_record(1, 92.0),   # CRITICAL -> P1
            make_record(2, 60.0),   # HIGH      -> P2
            make_record(3, 30.0),   # MEDIUM    -> P3
            make_record(4, 5.0),    # LOW       -> P4
        ]
        result = prioritize.prioritize_alerts(records, top_n=None)
        self.assertEqual(
            [a["priority"] for a in result["ordered_alerts"]],
            ["P1", "P2", "P3", "P4"],
        )
        self.assertEqual(result["counts_by_priority"],
                         {"P1": 1, "P2": 1, "P3": 1, "P4": 1})


class TestRankIntegrity(unittest.TestCase):
    def setUp(self):
        self.records = [make_record(i, (i * 13) % 101) for i in range(120)]
        self.alerts = prioritize.prioritize_alerts(
            self.records, top_n=None
        )["ordered_alerts"]

    def test_ranks_unique(self):
        ranks = [a["rank"] for a in self.alerts]
        self.assertEqual(len(set(ranks)), len(ranks))

    def test_ranks_complete_and_contiguous(self):
        self.assertEqual([a["rank"] for a in self.alerts],
                         list(range(1, len(self.alerts) + 1)))

    def test_complete_coverage_no_duplicates(self):
        self.assertEqual(len(self.alerts), len(self.records))
        self.assertEqual({a["sample_id"] for a in self.alerts},
                         {r["sample_id"] for r in self.records})

    def test_empty_input(self):
        result = prioritize.prioritize_alerts([], top_n=10)
        self.assertEqual(result["total_alerts"], 0)
        self.assertEqual(result["ordered_alerts"], [])
        self.assertEqual(result["top_n"], [])
        self.assertEqual(result["counts_by_priority"],
                         {"P1": 0, "P2": 0, "P3": 0, "P4": 0})


class TestTopNBehaviour(unittest.TestCase):
    def test_default_top_n_is_10(self):
        records = [make_record(i, 50.0 + i) for i in range(20)]
        result = prioritize.prioritize_alerts(records)
        self.assertEqual(len(result["top_n"]), 10)
        self.assertEqual(result["top_n"][0]["rank"], 1)

    def test_top_n_larger_than_total(self):
        records = [make_record(i, i) for i in range(5)]
        result = prioritize.prioritize_alerts(records, top_n=100)
        self.assertEqual(len(result["top_n"]), 5)

    def test_top_n_none_and_zero(self):
        records = [make_record(i, i) for i in range(5)]
        self.assertEqual(prioritize.prioritize_alerts(records, top_n=0)["top_n"], [])
        self.assertEqual(
            prioritize.prioritize_alerts(records, top_n=None)["top_n"], []
        )


class TestMalformedInput(unittest.TestCase):
    def test_missing_field(self):
        bad = make_record(1, 80.0)
        del bad["risk_score"]
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts([bad])

    def test_non_finite_rejected(self):
        for field, value in (("risk_score", float("nan")),
                             ("anomaly_score", float("inf")),
                             ("evidence_score", float("-inf"))):
            bad = make_record(1, 80.0)
            bad[field] = value
            with self.assertRaises(ValueError):
                prioritize.prioritize_alerts([bad])

    def test_anomaly_score_out_of_range(self):
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts([make_record(1, 80.0, anomaly_score=1.5)])

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts(
                [make_record(1, 80.0, risk_level="SEVERE")]
            )

    def test_level_score_inconsistency_rejected(self):
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts([make_record(1, 30.0, risk_level="CRITICAL")])

    def test_duplicate_sample_ids_rejected(self):
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts(
                [make_record(7, 80.0), make_record(7, 60.0)]
            )

    def test_top_evidence_must_be_list(self):
        bad = make_record(1, 80.0)
        bad["top_evidence"] = "src_bytes"
        with self.assertRaises(ValueError):
            prioritize.prioritize_alerts([bad])


class TestNoRecompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pathlib

        path = prioritize.RISK_REPORT_PATH
        cls.risk_report = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )
        cls.tmp_dir = None

    def test_true_labels_dropped_and_inert(self):
        records = [
            make_record(1, 97.0, true_label="normal"),
            make_record(2, 42.0, true_label="anomaly"),
        ]
        result = prioritize.prioritize_alerts(records, top_n=None)
        for alert in result["ordered_alerts"]:
            self.assertNotIn("true_label", alert)
        # Even the artifact builder must not carry labels into alerts.
        if self.risk_report is not None:
            report = prioritize.build_prioritization_report(self.risk_report)
            self.assertTrue(
                all("true_label" not in a for a in report["alerts"])
            )

    def test_ordering_invariant_under_true_label_permutation(self):
        if self.risk_report is None:
            self.skipTest("run 'python -m src.nids.risk' to generate reports/")
        import copy

        base = prioritize.prioritize_alerts(self.risk_report["samples"], top_n=None)
        flipped = copy.deepcopy(self.risk_report["samples"])
        for i, record in enumerate(flipped):
            record["true_label"] = "anomaly" if i % 2 == 0 else "normal"
        permuted = prioritize.prioritize_alerts(flipped, top_n=None)
        self.assertEqual(
            [a["sample_id"] for a in base["ordered_alerts"]],
            [a["sample_id"] for a in permuted["ordered_alerts"]],
        )

    def test_no_model_fitting_and_no_shap_recomputation(self):
        if self.risk_report is None:
            self.skipTest("run 'python -m src.nids.risk' to generate reports/")
        import shap

        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")), \
             mock.patch.object(RandomForestClassifier, "fit",
                               side_effect=AssertionError("RF.fit called")), \
             mock.patch.object(shap.TreeExplainer, "shap_values",
                               side_effect=AssertionError("SHAP recomputed")), \
             mock.patch.object(shap.TreeExplainer, "__init__",
                               side_effect=AssertionError("explainer rebuilt"),
                               return_value=None):
            report = prioritize.build_prioritization_report(self.risk_report)
        self.assertEqual(report["total_alerts"], len(self.risk_report["samples"]))

    def test_no_test_data_csv_access(self):
        if self.risk_report is None:
            self.skipTest("run 'python -m src.nids.risk' to generate reports/")
        import pandas as pd

        real_read_csv = pd.read_csv

        def guarded_read_csv(path, *args, **kwargs):
            if "Test_data" in str(path):
                raise AssertionError(f"Test_data.csv was read: {path}")
            return real_read_csv(path, *args, **kwargs)

        with mock.patch("src.nids.data.pd.read_csv",
                        side_effect=guarded_read_csv), \
             mock.patch("src.nids.evaluate.pd.read_csv",
                        side_effect=guarded_read_csv):
            report = prioritize.build_prioritization_report(self.risk_report)
        self.assertIsNotNone(report["excluded_unlabelled_dataset"])
        self.assertFalse(report["excluded_unlabelled_dataset"]["has_class_column"])


class TestPrioritizationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pathlib

        path = prioritize.RISK_REPORT_PATH
        if not path.exists():
            raise unittest.SkipTest(
                "run 'python -m src.nids.risk' to generate reports/"
            )
        cls.risk_report = json.loads(path.read_text(encoding="utf-8"))
        cls.report = prioritize.build_prioritization_report(cls.risk_report)

    def test_report_schema(self):
        r = self.report
        self.assertEqual(r["report"], prioritize.REPORT_NAME)
        self.assertEqual(r["schema_version"], "1.0")
        self.assertEqual(
            r["note"], "An offline deterministic prioritization of model-generated alerts."
        )
        self.assertIn("REVIEW priorities", r["caveat"])
        self.assertIn("not attack severity", r["caveat"])
        methodology = r["methodology"]
        for key in ("ordering_version", "ordering_rules", "priority_mapping",
                    "mapping_rationale", "unflagged_records_policy",
                    "true_labels", "no_second_score", "upstream"):
            self.assertIn(key, methodology)
        self.assertIn("highest analyst review priority",
                      methodology["priority_mapping"]["CRITICAL"])
        self.assertIn("no new score", methodology["no_second_score"])
        self.assertEqual(r["evaluation_population"],
                         self.risk_report["evaluation_population"])

    def test_counts_consistent_with_risk_levels(self):
        self.assertEqual(r := self.report["counts_by_priority"],
                         {p: c for p, c in (("P1", self.report["counts_by_risk_level"]["CRITICAL"]),
                                            ("P2", self.report["counts_by_risk_level"]["HIGH"]),
                                            ("P3", self.report["counts_by_risk_level"]["MEDIUM"]),
                                            ("P4", self.report["counts_by_risk_level"]["LOW"]))})
        self.assertEqual(sum(r.values()), self.report["total_alerts"])
        self.assertEqual(self.report["counts_by_risk_level"],
                         self.risk_report["risk"]["levels"]["counts"])
        self.assertEqual(self.report["total_alerts"],
                         len(self.risk_report["samples"]))

    def test_alert_record_shape(self):
        for alert in self.report["alerts"]:
            self.assertEqual(
                set(alert),
                {"rank", "sample_id", "source_row_id", "prediction",
                 "anomaly_score", "evidence_score", "risk_score", "risk_level",
                 "priority", "top_evidence"},
            )
            self.assertEqual(alert["priority"],
                             prioritize.PRIORITY_MAP[alert["risk_level"]])

    def test_alerts_preserve_source_row_ids(self):
        risk_by_id = {
            int(r["sample_id"]): r["source_row_id"]
            for r in self.risk_report["samples"]
        }
        for alert in self.report["alerts"]:
            self.assertEqual(alert["source_row_id"],
                             risk_by_id[alert["sample_id"]])

    def test_provenance_blocks(self):
        r = self.report
        self.assertEqual(r["model"]["sha256"],
                         self.risk_report["model"]["sha256"])
        self.assertFalse(r["model"]["retrained"])
        self.assertIsNotNone(r["excluded_unlabelled_dataset"])
        self.assertIn("git_revision", r["provenance"])
        self.assertEqual(r["methodology"]["upstream"]["risk_report"],
                         self.risk_report["report"])


class TestGeneratedArtifact(unittest.TestCase):
    def test_generated_prioritization_json(self):
        path = prioritize.ALERT_REPORT_PATH
        if not path.exists():
            self.skipTest(
                "run 'python -m src.nids.prioritize' to generate reports/"
            )
        r = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(r["report"], prioritize.REPORT_NAME)
        self.assertEqual(len(r["alerts"]), r["total_alerts"])
        self.assertEqual([a["rank"] for a in r["alerts"]],
                         list(range(1, r["total_alerts"] + 1)))
        self.assertTrue(
            all(r["alerts"][i]["risk_score"] >= r["alerts"][i + 1]["risk_score"]
                for i in range(len(r["alerts"]) - 1))
        )
        self.assertEqual(len(r["top_n"]), 10)
        self.assertEqual(r["top_n"][0]["risk_score"],
                         max(a["risk_score"] for a in r["alerts"]))
        self.assertTrue(all("true_label" not in a for a in r["alerts"]))


class TestSourceRowIdentity(unittest.TestCase):
    """source_row_id is pass-through lineage metadata, never ordering input."""

    @staticmethod
    def _mini_risk_report(records):
        """The blocks build_prioritization_report reads from a risk report."""
        return {
            "samples": records,
            "report": "nids_risk_assessment",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "evaluation_population": "synthetic mini risk report",
            "excluded_unlabelled_dataset": {"has_class_column": False},
            "dataset": {},
            "model": {"sha256": ""},
            "risk": {
                "methodology": {"version": "1.0.0", "formula": "test"},
                "levels": {"counts": {}},
            },
        }

    def test_sample_id_semantics_unchanged(self):
        # sample_id remains the record's own identity field and the final
        # sort key: the records below carry distinct risk/anomaly scores, so
        # ties resolve by sample_id ascending.
        records = [
            make_record(9, 80.0, anomaly_score=0.8),
            make_record(2, 80.0, anomaly_score=0.8),
            make_record(7, 80.0, anomaly_score=0.8),
        ]
        result = prioritize.prioritize_alerts(records, top_n=None)
        self.assertEqual([a["sample_id"] for a in result["ordered_alerts"]],
                         [2, 7, 9])

    def test_both_ids_propagate_from_risk_records(self):
        records = [
            make_record(0, 97.0, source_row_id=18473),
            make_record(1, 42.0, source_row_id=17),
        ]
        report = prioritize.build_prioritization_report(
            self._mini_risk_report(records), top_n=None
        )
        by_id = {a["sample_id"]: a for a in report["alerts"]}
        self.assertEqual(by_id[0]["source_row_id"], 18473)
        self.assertEqual(by_id[1]["source_row_id"], 17)

    def test_source_row_id_absent_stays_absent(self):
        # Backward compatibility: legacy risk records without the field
        # produce legacy alert records without it.
        report = prioritize.build_prioritization_report(
            self._mini_risk_report([make_record(0, 97.0)]), top_n=None
        )
        self.assertNotIn("source_row_id", report["alerts"][0])

    def test_changing_source_row_id_cannot_change_ordering(self):
        records = [
            make_record(i, 50.0 + (i % 3) * 10)
            for i in range(12)
        ]
        base = prioritize.prioritize_alerts(
            copy.deepcopy(records), top_n=None
        )
        scrambled = copy.deepcopy(records)
        for i, record in enumerate(scrambled):
            record["source_row_id"] = (i * 7919) % 1000 - 500
        permuted = prioritize.prioritize_alerts(scrambled, top_n=None)
        self.assertEqual(
            [(a["sample_id"], a["risk_score"]) for a in base["ordered_alerts"]],
            [(a["sample_id"], a["risk_score"]) for a in permuted["ordered_alerts"]],
        )

    def test_source_row_id_not_in_ordering_rules(self):
        records = [make_record(0, 97.0, source_row_id=12345)]
        report = prioritize.build_prioritization_report(
            self._mini_risk_report(records), top_n=None
        )
        joined = " ".join(report["methodology"]["ordering_rules"])
        self.assertNotIn("source_row_id", joined)
        self.assertIn("source_row_id", report["methodology"]["data_lineage"])

    def test_source_row_id_not_validated(self):
        # Metadata only: out-of-range values are passed through untouched
        # (the field is never interpreted, so there is nothing to validate).
        records = [make_record(0, 97.0, source_row_id=-1)]
        report = prioritize.build_prioritization_report(
            self._mini_risk_report(records), top_n=None
        )
        self.assertEqual(report["alerts"][0]["source_row_id"], -1)


class TestSlowRealDataPrioritize(unittest.TestCase):
    def test_full_prioritization_run(self):
        if os.environ.get("RUN_SLOW_EVAL_TESTS") != "1":
            self.skipTest("set RUN_SLOW_EVAL_TESTS=1 to run the full report")
        report = prioritize.main()
        self.assertEqual(report["total_alerts"], 1000)
        self.assertEqual(sum(report["counts_by_priority"].values()), 1000)


if __name__ == "__main__":
    unittest.main()
