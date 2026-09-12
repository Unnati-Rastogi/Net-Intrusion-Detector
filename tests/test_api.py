"""Tests for the FastAPI layer.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import unittest

from fastapi.testclient import TestClient

from src.nids import config
from src.nids.app import app
from src.nids.data import load_training_data
from src.nids.sample_data import SAMPLE_CONNECTION
from src.nids.schemas import FlowFeatures


class TestSchemaPinning(unittest.TestCase):
    """The API schema must match the real dataset schema exactly."""

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
        self.assertEqual(
            sorted(paths),
            ["/demo/test-rows", "/features", "/health",
             "/predict", "/predict/batch", "/sample"],
        )


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dashboard_served_at_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html = response.text
        self.assertIn("ML Network Intrusion Detection", html)
        self.assertIn("Load Sample", html)
        self.assertIn("Predict", html)
        self.assertIn("/static/app.js", html)

    def test_static_js_served(self):
        response = self.client.get("/static/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/predict", response.text)

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

    def test_demo_test_rows_are_valid_predict_payloads(self):
        demo = self.client.get("/demo/test-rows?limit=5").json()
        batch = self.client.post("/predict/batch", json={"features": demo["rows"]})
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(batch.json()["count"], 5)


if __name__ == "__main__":
    unittest.main()
