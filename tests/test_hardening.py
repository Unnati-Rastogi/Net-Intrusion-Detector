"""Engineering-hardening tests: API robustness, error handling, state safety.

Complements tests/test_api.py (functional coverage). Focus here:

- input edge cases for every endpoint (missing/extra/wrong-type/non-finite
  values, unknown categoricals, empty and malformed batches)
- error contracts: 422 validation errors are structured and readable,
  404s are appropriate, non-finite model inputs never reach the forest
  (a NaN rate would be read as 0, +/-inf raises inside sklearn)
- /alerts serves the committed report verbatim; a deliberately corrupted
  report produces a clear 500, never a silent success
- concurrent /analyze requests share the cached explanation context and
  lineage snapshot without corrupting state (threaded smoke test)
- the committed prioritization report is well-formed (consistency guard
  for the API contract)

Run from the project root:
    python -m unittest discover -s tests -v
"""

import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.nids import analyze as analyze_module
from src.nids import config, prioritize
from src.nids.app import app
from src.nids.sample_data import SAMPLE_CONNECTION

# data/Train_data.csv is gitignored: /analyze needs it (SHAP background and
# report lineage), so those tests are skipped in a clean checkout / CI.
_HAS_TRAIN = config.TRAIN_CSV.exists()
_TRAIN_SKIP_MSG = "requires data/Train_data.csv (gitignored)"


def sample(**overrides) -> dict:
    record = dict(SAMPLE_CONNECTION)
    record.update(overrides)
    return record


class ApiTestCase(unittest.TestCase):
    """Shared TestClient that does not raise on 5xx responses."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)


class TestPredictRobustness(ApiTestCase):
    """POST /predict edge cases."""

    def test_valid_request(self):
        r = self.client.post("/predict", json={"features": sample()})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["prediction"], ("normal", "anomaly"))
        self.assertAlmostEqual(
            body["probability_normal"] + body["probability_anomaly"], 1.0, places=6
        )

    def test_missing_field(self):
        r = self.client.post("/predict", json={"features": {
            k: v for k, v in sample().items() if k != "src_bytes"
        }})
        self.assertEqual(r.status_code, 422)
        self.assertTrue(any("src_bytes" in e for e in r.json()["feature_errors"]))

    def test_extra_field(self):
        r = self.client.post(
            "/predict", json={"features": sample(made_up_field=1)}
        )
        self.assertEqual(r.status_code, 422)
        self.assertTrue(
            any("made_up_field" in e for e in r.json()["feature_errors"])
        )

    def test_wrong_type(self):
        r = self.client.post("/predict", json={"features": sample(src_bytes="abc")})
        self.assertEqual(r.status_code, 422)

    def test_non_finite_rate_is_rejected(self):
        # 'NaN' would silently become 0 inside the forest; '+inf' raises
        # inside sklearn. Both must be 422 validation errors, never 500s.
        for value in ("NaN", "nan", "inf", "+inf", "-inf", "Infinity"):
            r = self.client.post(
                "/predict", json={"features": sample(serror_rate=value)}
            )
            self.assertEqual(r.status_code, 422, value)
            self.assertIn("feature_errors", r.json(), value)

    def test_non_finite_numeric_rate_is_rejected(self):
        # Same contract for native JSON numbers (sent via raw content so the
        # test client does not pre-validate float values).
        raw = json.dumps({"features": sample(serror_rate=float("nan"))}).replace(
            '"serror_rate":nan', '"serror_rate":NaN'
        )
        r = self.client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
        self.assertEqual(r.status_code, 422)

    def test_unknown_categorical_value_is_tolerated(self):
        # tftp_u is absent from the training data; the OneHotEncoder's
        # handle_unknown='ignore' makes this a valid request by design.
        r = self.client.post(
            "/predict", json={"features": sample(service="tftp_u")}
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["prediction"], ("normal", "anomaly"))

    def test_malformed_json_body(self):
        r = self.client.post(
            "/predict", content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 422)

    def test_wrong_body_shape(self):
        r = self.client.post("/predict", json={"records": sample()})
        self.assertEqual(r.status_code, 422)


class TestBatchRobustness(ApiTestCase):
    """POST /predict/batch edge cases."""

    def test_valid_batch(self):
        r = self.client.post(
            "/predict/batch", json={"features": [sample(), sample(src_bytes=100)]}
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 2)

    def test_empty_batch_rejected(self):
        r = self.client.post("/predict/batch", json={"features": []})
        self.assertEqual(r.status_code, 422)

    def test_batch_over_limit_rejected(self):
        r = self.client.post(
            "/predict/batch",
            json={"features": [sample() for _ in range(10_001)]},
        )
        self.assertEqual(r.status_code, 422)

    def test_malformed_row_rejected(self):
        r = self.client.post(
            "/predict/batch",
            json={"features": [sample(), {"src_bytes": 1}]},
        )
        self.assertEqual(r.status_code, 422)
        self.assertIn("feature_errors", r.json())

    def test_mixed_valid_invalid_rejected_atomically(self):
        # One bad row rejects the whole request: the batch is a single
        # validated payload, not a partial-success stream.
        r = self.client.post(
            "/predict/batch",
            json={"features": [sample(), sample(serror_rate="inf")]},
        )
        self.assertEqual(r.status_code, 422)

    def test_validation_error_structure(self):
        r = self.client.post(
            "/predict/batch",
            json={"features": [sample(serror_rate="NaN"), sample()]},
        )
        self.assertEqual(r.status_code, 422)
        body = r.json()
        self.assertIn("detail", body)
        self.assertIn("feature_errors", body)
        self.assertTrue(any("serror_rate" in e for e in body["feature_errors"]))


class TestAnalyzeRobustness(ApiTestCase):
    """POST /analyze edge cases (workflow coverage lives in test_api.py)."""

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_features_mode(self):
        r = self.client.post("/analyze", json={"features": sample()})
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["risk_level"],
                      ("LOW", "MEDIUM", "HIGH", "CRITICAL"))

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_sample_id_mode(self):
        r = self.client.post("/analyze", json={"sample_id": 0})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["sample_id"], 0)
        self.assertIsNotNone(body["source_row_id"])

    def test_missing_both_rejected(self):
        r = self.client.post("/analyze", json={})
        self.assertEqual(r.status_code, 422)
        self.assertIn("exactly one", r.json()["detail"])

    def test_both_supplied_rejected(self):
        r = self.client.post(
            "/analyze", json={"features": sample(), "sample_id": 0}
        )
        self.assertEqual(r.status_code, 422)
        self.assertIn("exactly one", r.json()["detail"])

    def test_negative_sample_id_rejected(self):
        r = self.client.post("/analyze", json={"sample_id": -1})
        self.assertEqual(r.status_code, 422)

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_nonexistent_sample_id_is_404(self):
        r = self.client.post("/analyze", json={"sample_id": 999_999})
        self.assertEqual(r.status_code, 404)
        self.assertIn("unknown sample_id", r.json()["detail"])

    def test_malformed_feature_vector(self):
        r = self.client.post("/analyze", json={"features": {"service": "http"}})
        self.assertEqual(r.status_code, 422)

    def test_non_finite_rate_rejected(self):
        r = self.client.post("/analyze", json={"features": sample(serror_rate="inf")})
        self.assertEqual(r.status_code, 422)

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_unknown_categorical_value(self):
        r = self.client.post(
            "/analyze", json={"features": sample(service="tftp_u")}
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["prediction"], ("normal", "anomaly"))

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_arbitrary_vector_does_not_fabricate_lineage(self):
        r = self.client.post(
            "/analyze", json={"features": sample(src_bytes=987_654)}
        )
        body = r.json()
        self.assertIsNone(body["sample_id"])
        self.assertIsNone(body["source_row_id"])

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_known_sample_preserves_lineage(self):
        report = json.loads(
            analyze_module.RISK_REPORT_PATH.read_text(encoding="utf-8")
        )
        record = report["samples"][5]
        r = self.client.post("/analyze", json={"sample_id": record["sample_id"]})
        body = r.json()
        self.assertEqual(body["source_row_id"], record["source_row_id"])


class TestAlertsRobustness(ApiTestCase):
    """GET /alerts behaviour, including a corrupted-report guard."""

    def test_report_served_with_schema_and_ordering(self):
        r = self.client.get("/alerts")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["total_alerts"], len(body["alerts"]))
        scores = [a["risk_score"] for a in body["alerts"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([a["rank"] for a in body["alerts"]],
                         list(range(1, len(scores) + 1)))
        for alert in body["alerts"][:25]:
            self.assertIn(alert["priority"], ("P1", "P2", "P3", "P4"))
            self.assertIn(alert["risk_level"],
                          ("LOW", "MEDIUM", "HIGH", "CRITICAL"))

    def test_missing_report_is_404(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            with mock.patch.object(prioritize, "ALERT_REPORT_PATH", missing):
                r = self.client.get("/alerts")
        self.assertEqual(r.status_code, 404)
        self.assertIn("python -m src.nids.prioritize", r.json()["detail"])

    def test_corrupted_report_is_clear_500(self):
        with TemporaryDirectory() as tmp:
            broken = Path(tmp) / "alert_prioritization.json"
            broken.write_text("{definitely not json", encoding="utf-8")
            with mock.patch.object(prioritize, "ALERT_REPORT_PATH", broken):
                r = self.client.get("/alerts")
        self.assertEqual(r.status_code, 500)
        # The message must name the artifact and the fix, not leak a traceback.
        detail = r.json()["detail"]
        self.assertIn("alert_prioritization.json", detail)
        self.assertIn("python -m src.nids.prioritize", detail)


class TestHealthEndpoint(ApiTestCase):

    def test_healthy_response_structure(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(
            set(body),
            {"status", "model_loaded", "model_path", "feature_count", "classes"},
        )
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["feature_count"], 41)
        self.assertEqual(body["classes"], [0, 1])


class TestErrorHandling(ApiTestCase):
    """Process-level error contract: bad input never crashes the server."""

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_process_survives_a_burst_of_bad_requests(self):
        bad_requests = [
            ("/predict", {"features": sample(src_bytes="x")}),
            ("/predict", {"features": sample(serror_rate="inf")}),
            ("/predict", {}),
            ("/predict/batch", {"features": []}),
            ("/analyze", {}),
            ("/analyze", {"sample_id": -5}),
            ("/analyze", {"sample_id": 10**9}),
        ]
        for path, payload in bad_requests:
            r = self.client.post(path, json=payload)
            self.assertIn(r.status_code, (404, 422), (path, payload))
        # The app is still healthy afterwards.
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_404_on_unknown_route(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)


class TestConcurrentAnalyze(ApiTestCase):
    """Concurrent /analyze calls share cached state without corruption."""

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_parallel_analyze_is_consistent(self):
        # Warm the caches once (model/context/lineage built at first use).
        first = self.client.post("/analyze", json={"features": sample()}).json()
        n = 8
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(
                lambda sid: self.client.post(
                    "/analyze", json={"sample_id": sid}
                ).json(),
                range(n),
            ))
        # Every concurrent call preserved its own lineage and matched the
        # serialized baseline for the same record.
        for sid, result in zip(range(n), results):
            self.assertEqual(result["sample_id"], sid)
        baseline = self.client.post("/analyze", json={"sample_id": 0}).json()
        self.assertEqual(results[0], baseline)
        # Ad-hoc mode stays lineage-free under concurrency.
        adhoc = self.client.post(
            "/analyze", json={"features": sample(src_bytes=555_001)}
        ).json()
        self.assertIsNone(adhoc["sample_id"])
        self.assertIsNone(adhoc["source_row_id"])
        self.assertEqual(first["prediction"], "normal")


class TestNoRetrainDuringHardenedFlow(unittest.TestCase):
    """Hardening must not have introduced any fitting path."""

    @unittest.skipUnless(_HAS_TRAIN, _TRAIN_SKIP_MSG)
    def test_no_fit_calls_across_endpoints(self):
        client = TestClient(app)
        sha_before = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        with mock.patch.object(Pipeline, "fit",
                               side_effect=AssertionError("pipeline.fit called")), \
             mock.patch.object(RandomForestClassifier, "fit",
                               side_effect=AssertionError("RF.fit called")):
            self.assertEqual(client.post(
                "/predict", json={"features": sample()}).status_code, 200)
            self.assertEqual(client.post(
                "/analyze", json={"sample_id": 1}).status_code, 200)
            self.assertEqual(client.get("/alerts").status_code, 200)
        sha_after = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)


class TestCommittedReportsWellFormed(unittest.TestCase):
    """Contract guard for the artifacts the API serves."""

    def test_prioritization_report_matches_risk_report(self):
        risk = json.loads(
            analyze_module.RISK_REPORT_PATH.read_text(encoding="utf-8")
        )
        alerts = json.loads(
            prioritize.ALERT_REPORT_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(alerts["total_alerts"], len(risk["samples"]))
        self.assertEqual(
            alerts["counts_by_risk_level"], risk["risk"]["levels"]["counts"]
        )
        self.assertEqual(
            {a["sample_id"]: a["source_row_id"] for a in alerts["alerts"]},
            {s["sample_id"]: s["source_row_id"] for s in risk["samples"]},
        )


if __name__ == "__main__":
    unittest.main()
