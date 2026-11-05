"""Tests for the FastAPI layer.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import hashlib
import json
import unittest
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.nids import analyze as analyze_module
from src.nids import config, explain, prioritize, risk
from src.nids.app import app
from src.nids.data import load_training_data
from src.nids.sample_data import SAMPLE_CONNECTION
from src.nids.schemas import FlowFeatures

# data/*.csv are gitignored: tests that read the real datasets are skipped
# when they are absent (clean checkout / CI). Add the datasets locally for
# the full suite.
_HAS_TRAIN = config.TRAIN_CSV.exists()
_HAS_TEST = config.TEST_CSV.exists()


class TestSchemaPinning(unittest.TestCase):
    """The API schema must match the real dataset schema exactly."""

    @unittest.skipUnless(
        _HAS_TRAIN, "requires data/Train_data.csv (gitignored)"
    )
    def test_flow_features_match_dataset_columns(self):
        X, _ = load_training_data()
        self.assertEqual(list(FlowFeatures.model_fields.keys()), list(X.columns))
        self.assertEqual(list(FlowFeatures.model_fields.keys()), config.FEATURE_COLUMNS)

    def test_unknown_fields_are_forbidden(self):
        self.assertEqual(FlowFeatures.model_config.get("extra"), "forbid")


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["feature_count"], 41)
        self.assertEqual(body["classes"], [0, 1])

    def test_predict_valid_sample(self):
        response = self.client.post("/predict", json={"features": SAMPLE_CONNECTION})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body),
            {"prediction", "label", "probability_normal",
             "probability_anomaly", "confidence"},
        )
        self.assertIn(body["prediction"], ("normal", "anomaly"))
        self.assertIn(body["label"], (0, 1))
        self.assertAlmostEqual(
            body["probability_normal"] + body["probability_anomaly"], 1.0, places=6
        )
        self.assertEqual(body["confidence"], max(body["probability_normal"],
                                                 body["probability_anomaly"]))

    def test_predict_missing_field_returns_feature_errors(self):
        broken = {k: v for k, v in SAMPLE_CONNECTION.items() if k != "src_bytes"}
        response = self.client.post("/predict", json={"features": broken})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("feature_errors", body)
        self.assertIn("detail", body)
        self.assertTrue(any("src_bytes" in e for e in body["feature_errors"]))

    def test_predict_extra_field_rejected(self):
        bad = dict(SAMPLE_CONNECTION)
        bad["made_up_feature"] = 1
        response = self.client.post("/predict", json={"features": bad})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any("made_up_feature" in e for e in response.json()["feature_errors"])
        )

    def test_predict_bad_type_rejected(self):
        bad = dict(SAMPLE_CONNECTION)
        bad["src_bytes"] = "not-a-number"
        response = self.client.post("/predict", json={"features": bad})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any("src_bytes" in e for e in response.json()["feature_errors"])
        )

    def test_batch_predict(self):
        payload = {"features": [SAMPLE_CONNECTION, SAMPLE_CONNECTION]}
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["predictions"]), 2)
        for prediction in body["predictions"]:
            self.assertIn(prediction["prediction"], ("normal", "anomaly"))

    def test_batch_empty_list_rejected(self):
        response = self.client.post("/predict/batch", json={"features": []})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any("features" in e for e in response.json()["feature_errors"])
        )

    def test_openapi_docs_available(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertTrue({
            "/health",
            "/detections/cic/status",
            "/detections/cic/scan",
            "/detections/cic/explain",
            "/evaluation/cic-ids2017",
        }.issubset(paths))


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dashboard_served_at_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html = response.text
        self.assertIn("Scan dataset", html)
        self.assertIn("Model validation", html)
        self.assertIn("Why this flow was flagged", html)
        self.assertIn("/static/app.js", html)

    def test_static_js_served(self):
        response = self.client.get("/static/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/detections/cic/scan", response.text)
        self.assertIn("/detections/cic/explain", response.text)

    def test_dashboard_renders_api_data_safely(self):
        # API-derived values must never pass through innerHTML.
        js = self.client.get("/static/app.js").text
        self.assertNotIn("innerHTML", js)
        self.assertIn("createElement", js)
        self.assertIn("textContent", js)

    @unittest.skipUnless(
        _HAS_TRAIN, "requires data/Train_data.csv (category values)"
    )
    def test_features_endpoint(self):
        response = self.client.get("/features")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["feature_columns"], config.FEATURE_COLUMNS)
        self.assertEqual(body["categorical_columns"], config.CATEGORICAL_COLUMNS)
        for column in config.CATEGORICAL_COLUMNS:
            self.assertIn(column, body["category_values"])
            self.assertTrue(len(body["category_values"][column]) > 0)

    def test_sample_endpoint_matches_canonical_sample(self):
        response = self.client.get("/sample")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["features"], SAMPLE_CONNECTION)

    @unittest.skipUnless(
        _HAS_TEST, "requires data/Test_data.csv (gitignored)"
    )
    def test_demo_test_rows_default_and_limit(self):
        response = self.client.get("/demo/test-rows")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 10)

        response = self.client.get("/demo/test-rows?limit=3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 3)

        response = self.client.get("/demo/test-rows?limit=5000")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 100)  # capped

    @unittest.skipUnless(
        _HAS_TEST, "requires data/Test_data.csv (gitignored)"
    )
    def test_demo_test_rows_are_valid_predict_payloads(self):
        demo = self.client.get("/demo/test-rows?limit=5").json()
        batch = self.client.post("/predict/batch", json={"features": demo["rows"]})
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(batch.json()["count"], 5)


@unittest.skipUnless(
    _HAS_TRAIN,
    "requires data/Train_data.csv (SHAP background + report lineage)",
)
class TestAnalyzeWorkflow(unittest.TestCase):
    """POST /analyze: detection -> SHAP explanation -> risk -> priority."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def _analyze(self, payload):
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_analyze_valid_record_full_schema(self):
        body = self._analyze({"features": SAMPLE_CONNECTION})
        self.assertEqual(
            set(body),
            {"prediction", "label", "anomaly_score", "probability_normal",
             "probability_anomaly", "confidence", "risk_score", "risk_level",
             "priority", "sample_id", "source_row_id", "explanation",
             "methodology"},
        )
        self.assertEqual(set(body["methodology"]),
                         set(analyze_module.METHODOLOGY_NOTE))

    def test_analyze_matches_fast_predict_path(self):
        fast = self.client.post("/predict", json={"features": SAMPLE_CONNECTION}).json()
        body = self._analyze({"features": SAMPLE_CONNECTION})
        self.assertEqual(body["prediction"], fast["prediction"])
        self.assertEqual(body["anomaly_score"], fast["probability_anomaly"])
        self.assertEqual(body["label"], fast["label"])
        self.assertEqual(body["probability_anomaly"], fast["probability_anomaly"])

    def test_analyze_returns_shap_evidence(self):
        body = self._analyze({"features": SAMPLE_CONNECTION})
        evidence = body["explanation"]
        self.assertEqual(len(evidence), 41)  # full per-feature attribution list
        self.assertEqual(
            set(evidence[0]),
            {"feature", "value", "shap_value", "direction",
             "abs_contribution"},
        )
        absolutes = [item["abs_contribution"] for item in evidence]
        self.assertEqual(absolutes, sorted(absolutes, reverse=True))
        for item in evidence:
            self.assertIn(item["direction"],
                          ("increases_anomaly_score", "decreases_anomaly_score"))
            self.assertAlmostEqual(
                item["abs_contribution"], abs(item["shap_value"]), places=12
            )

    def test_analyze_risk_matches_domain_implementation(self):
        body = self._analyze({"features": SAMPLE_CONNECTION})
        context = analyze_module.get_explanation_context()
        frame = pd.DataFrame(
            [SAMPLE_CONNECTION], columns=list(app.state.model.feature_names_in_)
        )
        explanation_row = explain.explain_samples(context, frame, top_k=None)[0]
        expected = risk.calculate_risk_for_explanation(
            explanation_row, n_estimators=context.classifier.n_estimators
        )
        self.assertEqual(body["risk_score"], expected["risk_score"])
        self.assertEqual(body["risk_level"], expected["risk_level"])
        self.assertEqual(body["anomaly_score"], expected["anomaly_score"])

    def test_analyze_priority_matches_prioritization_implementation(self):
        body = self._analyze({"features": SAMPLE_CONNECTION})
        self.assertEqual(
            body["priority"], prioritize.PRIORITY_MAP[body["risk_level"]]
        )
        # Same result the batch prioritizer would produce for this record.
        record = {
            "sample_id": body["sample_id"] if body["sample_id"] is not None else 0,
            "prediction": body["prediction"],
            "anomaly_score": body["anomaly_score"],
            "evidence_score": body["risk_score"] / body["anomaly_score"] * 2 / 100 - 1
            if body["anomaly_score"] else 0.0,
            "risk_score": body["risk_score"],
            "risk_level": body["risk_level"],
            "top_evidence": [
                {"feature": e["feature"], "value": e["value"],
                 "shap_contribution": e["shap_value"],
                 "direction": e["direction"],
                 "abs_contribution": e["abs_contribution"]}
                for e in body["explanation"]
            ],
        }
        record["evidence_score"] = (
            2.0 * body["risk_score"] / (100.0 * body["anomaly_score"]) - 1.0
            if body["anomaly_score"] else 0.0
        )
        ordered = prioritize.prioritize_alerts([record], top_n=None)
        self.assertEqual(ordered["ordered_alerts"][0]["priority"], body["priority"])

    def test_analyze_shap_evidence_matches_explanation_implementation(self):
        body = self._analyze({"features": SAMPLE_CONNECTION})
        context = analyze_module.get_explanation_context()
        frame = pd.DataFrame(
            [SAMPLE_CONNECTION], columns=list(app.state.model.feature_names_in_)
        )
        expected = explain.explain_samples(context, frame, top_k=None)[0]
        for got, want in zip(body["explanation"], expected["top_features"]):
            self.assertEqual(got["feature"], want["feature"])
            self.assertEqual(got["value"], want["value"])
            self.assertEqual(got["shap_value"], want["shap_contribution"])
            self.assertEqual(got["direction"], want["direction"])

    def test_analyze_does_not_retrain_or_touch_artifact(self):
        sha_before = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")), \
             mock.patch.object(RandomForestClassifier, "fit",
                               side_effect=AssertionError("RF.fit called")):
            self._analyze({"features": SAMPLE_CONNECTION})
        sha_after = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)

    def test_analyze_uses_existing_saved_artifact(self):
        # /health already points at the saved artifact; /analyze must run on
        # the same loaded model (no refit, no second artifact).
        health = self.client.get("/health").json()
        self.assertTrue(health["model_loaded"])
        self.assertIn("rf_ids_pipeline.joblib", health["model_path"])
        body = self._analyze({"features": SAMPLE_CONNECTION})
        self.assertIn(body["prediction"], ("normal", "anomaly"))

    def test_analyze_validation_errors(self):
        # Neither field.
        response = self.client.post("/analyze", json={})
        self.assertEqual(response.status_code, 422)
        # Both fields.
        response = self.client.post(
            "/analyze", json={"features": SAMPLE_CONNECTION, "sample_id": 0}
        )
        self.assertEqual(response.status_code, 422)
        # Missing feature.
        broken = {k: v for k, v in SAMPLE_CONNECTION.items() if k != "src_bytes"}
        response = self.client.post("/analyze", json={"features": broken})
        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any("src_bytes" in e for e in response.json()["feature_errors"])
        )
        # Bad type.
        bad = dict(SAMPLE_CONNECTION)
        bad["src_bytes"] = "not-a-number"
        response = self.client.post("/analyze", json={"features": bad})
        self.assertEqual(response.status_code, 422)

    def test_analyze_unknown_categorical_value_tolerated(self):
        # A service value absent from the training data must still work
        # (OneHotEncoder handle_unknown='ignore') - no error, full workflow.
        values = self.client.get("/features").json()["category_values"]["service"]
        unknown = next(
            (v for v in ("tftp_u", "har", "red_i", "uucp", "zzz_unseen_svc")
             if v not in values),
            "zzz_unseen_svc",
        )
        record = dict(SAMPLE_CONNECTION)
        record["service"] = unknown
        body = self._analyze({"features": record})
        self.assertIn(body["prediction"], ("normal", "anomaly"))
        self.assertEqual(len(body["explanation"]), 41)
        self.assertIsNone(body["sample_id"])


@unittest.skipUnless(
    _HAS_TRAIN,
    "requires data/Train_data.csv (report lineage join)",
)
class TestAnalyzeLineage(unittest.TestCase):
    """source_row_id is preserved for known samples, never fabricated."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.lineage = analyze_module.get_report_lineage()
        cls.report = json.loads(
            analyze_module.RISK_REPORT_PATH.read_text(encoding="utf-8")
        )

    def test_known_sample_preserves_both_ids(self):
        report_record = self.report["samples"][3]
        sample_id = report_record["sample_id"]
        body = self.client.post(
            "/analyze", json={"sample_id": sample_id}
        ).json()
        self.assertEqual(body["sample_id"], sample_id)
        self.assertEqual(body["source_row_id"],
                         report_record["source_row_id"])

    def test_known_sample_features_match_report_top_evidence(self):
        # The analyzed record really is the report's source row: the raw
        # feature values in the explanation must equal the stored sample's
        # top-evidence values.
        report_record = self.report["samples"][10]
        body = self.client.post(
            "/analyze", json={"sample_id": report_record["sample_id"]}
        ).json()
        stored_values = {
            item["feature"]: item["value"]
            for item in report_record["top_evidence"]
        }
        analyzed_values = {
            item["feature"]: item["value"] for item in body["explanation"]
        }
        for feature, value in stored_values.items():
            self.assertEqual(analyzed_values[feature], value, feature)

    def test_features_mode_recognizes_exact_report_sample(self):
        sample_id, entry = next(iter(self.lineage.items()))
        body = self.client.post(
            "/analyze", json={"features": entry["features"]}
        ).json()
        self.assertEqual(body["sample_id"], sample_id)
        self.assertEqual(body["source_row_id"], entry["source_row_id"])

    def test_arbitrary_vector_gets_no_fabricated_ids(self):
        record = dict(SAMPLE_CONNECTION)
        record["src_bytes"] = 123457  # not any stored report sample
        body = self.client.post("/analyze", json={"features": record}).json()
        self.assertIsNone(body["sample_id"])
        self.assertIsNone(body["source_row_id"])

    def test_unknown_sample_id_is_404(self):
        response = self.client.post("/analyze", json={"sample_id": 999999})
        self.assertEqual(response.status_code, 404)
        self.assertIn("unknown sample_id", response.json()["detail"])

    def test_negative_sample_id_rejected_by_schema(self):
        response = self.client.post("/analyze", json={"sample_id": -1})
        self.assertEqual(response.status_code, 422)


class TestMatchReportSample(unittest.TestCase):
    """Unit tests for the exact-match lineage resolver."""

    def test_single_exact_match(self):
        lineage = {
            5: {"source_row_id": 500, "features": {"src_bytes": 10, "service": "http"}},
            6: {"source_row_id": 600, "features": {"src_bytes": 20, "service": "http"}},
        }
        features = {"src_bytes": 20, "service": "http"}
        self.assertEqual(
            analyze_module._match_report_sample(features, lineage), 6
        )

    def test_no_match_returns_none(self):
        lineage = {5: {"source_row_id": 500, "features": {"src_bytes": 10}}}
        self.assertIsNone(
            analyze_module._match_report_sample({"src_bytes": 11}, lineage)
        )

    def test_ambiguous_match_returns_none(self):
        lineage = {
            5: {"source_row_id": 500, "features": {"src_bytes": 10}},
            6: {"source_row_id": 600, "features": {"src_bytes": 10}},
        }
        self.assertIsNone(
            analyze_module._match_report_sample({"src_bytes": 10}, lineage)
        )


class TestAlertsEndpoint(unittest.TestCase):
    """GET /alerts: the committed prioritization report, verbatim."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.committed = json.loads(
            prioritize.ALERT_REPORT_PATH.read_text(encoding="utf-8")
        )

    def test_alerts_report_metadata(self):
        response = self.client.get("/alerts")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["report"], self.committed["report"])
        self.assertEqual(body["total_alerts"], self.committed["total_alerts"])
        self.assertEqual(body["counts_by_priority"],
                         self.committed["counts_by_priority"])
        self.assertEqual(body["timestamp_utc"], self.committed["timestamp_utc"])

    def test_alerts_served_verbatim_no_reranking(self):
        body = self.client.get("/alerts").json()
        for committed_alert, served_alert in zip(
            self.committed["alerts"], body["alerts"]
        ):
            self.assertEqual(served_alert["rank"], committed_alert["rank"])
            self.assertEqual(served_alert["sample_id"], committed_alert["sample_id"])
            self.assertEqual(served_alert["risk_score"], committed_alert["risk_score"])
            self.assertEqual(served_alert["priority"], committed_alert["priority"])
            self.assertEqual(
                served_alert["source_row_id"], committed_alert["source_row_id"]
            )
        # Report order (risk descending) is preserved exactly.
        scores = [alert["risk_score"] for alert in body["alerts"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        ranks = [alert["rank"] for alert in body["alerts"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_alerts_record_shape(self):
        body = self.client.get("/alerts").json()
        for alert in body["alerts"][:50]:
            self.assertEqual(
                set(alert),
                {"rank", "sample_id", "source_row_id", "prediction",
                 "anomaly_score", "risk_score", "risk_level", "priority",
                 "top_evidence"},
            )


if __name__ == "__main__":
    unittest.main()
