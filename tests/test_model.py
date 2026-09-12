"""Tests for the NIDS ML core.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import unittest

from src.nids import config
from src.nids.data import load_inference_data, load_training_data
from src.nids.pipeline import build_pipeline
from src.nids.predict import ModelNotTrainedError, load_model, predict_connection, validate_sample
from src.nids.sample_data import SAMPLE_CONNECTION

class TestDataLoading(unittest.TestCase):
    def test_training_data_shape_and_schema(self):
        X, y = load_training_data()
        self.assertEqual(X.shape, (25192, 41))
        self.assertEqual(len(y), 25192)
        self.assertNotIn("class", X.columns)
        self.assertEqual(set(y.unique()), {0, 1})

    def test_inference_data_has_no_target(self):
        df = load_inference_data()
        self.assertEqual(df.shape, (22544, 41))
        self.assertNotIn("class", df.columns)

    def test_train_and_inference_columns_match(self):
        X, _ = load_training_data()
        df = load_inference_data()
        self.assertEqual(list(X.columns), list(df.columns))

    def test_unexpected_label_raises(self):
        import pandas as pd
        import tempfile
        import os

        df = pd.read_csv(config.TRAIN_CSV).head(50)
        df.loc[df.index[0], "class"] = "bogus"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_training_data(path)


class TestPipeline(unittest.TestCase):
    def test_build_pipeline_hyperparameters(self):
        model = build_pipeline()
        clf = model.named_steps["classifier"]
        self.assertEqual(clf.n_estimators, 200)
        self.assertEqual(clf.class_weight, "balanced")
        self.assertEqual(clf.random_state, 42)

    def test_pipeline_steps(self):
        model = build_pipeline()
        self.assertEqual(model.named_steps["preprocessing"].transformers[0][0], "const_drop")
        self.assertEqual(model.named_steps["preprocessing"].transformers[1][0], "categorical")


class TestModelLoading(unittest.TestCase):
    def test_load_model_returns_fitted_pipeline(self):
        model = load_model()
        self.assertTrue(hasattr(model, "feature_names_in_"))
        self.assertEqual(len(model.feature_names_in_), 41)
        self.assertEqual(sorted(model.classes_), [0, 1])

    def test_load_model_missing_artifact_raises(self):
        from pathlib import Path

        with self.assertRaises(ModelNotTrainedError):
            load_model(Path("does/not/exist.joblib"))


class TestSchemaValidation(unittest.TestCase):
    def test_valid_sample_passes(self):
        expected = list(SAMPLE_CONNECTION.keys())
        validate_sample(dict(SAMPLE_CONNECTION), expected)  # must not raise

    def test_missing_feature_raises(self):
        broken = dict(SAMPLE_CONNECTION)
        del broken["src_bytes"]
        with self.assertRaises(ValueError) as ctx:
            validate_sample(broken, list(SAMPLE_CONNECTION.keys()))
        self.assertIn("src_bytes", str(ctx.exception))

    def test_extra_feature_raises(self):
        broken = dict(SAMPLE_CONNECTION)
        broken["made_up_feature"] = 1
        with self.assertRaises(ValueError) as ctx:
            validate_sample(broken, list(SAMPLE_CONNECTION.keys()))
        self.assertIn("made_up_feature", str(ctx.exception))


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_sample_prediction_shape(self):
        result = predict_connection(dict(SAMPLE_CONNECTION), model=self.model)
        self.assertIn(result["prediction"], ("normal", "anomaly"))
        self.assertIn(result["label"], (0, 1))
        self.assertEqual(result["prediction"], "anomaly" if result["label"] else "normal")

    def test_sample_prediction_probabilities(self):
        result = predict_connection(dict(SAMPLE_CONNECTION), model=self.model)
        p_n, p_a = result["probability_normal"], result["probability_anomaly"]
        self.assertAlmostEqual(p_n + p_a, 1.0, places=6)
        for p in (p_n, p_a, result["confidence"]):
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        expected_conf = max(p_n, p_a)
        self.assertAlmostEqual(result["confidence"], expected_conf, places=9)

    def test_missing_feature_in_prediction_raises(self):
        broken = dict(SAMPLE_CONNECTION)
        del broken["service"]
        with self.assertRaises(ValueError):
            predict_connection(broken, model=self.model)

    def test_unknown_category_is_handled(self):
        # handle_unknown='ignore' must keep unseen categories from crashing.
        sample = dict(SAMPLE_CONNECTION)
        sample["service"] = "totally_new_service"
        result = predict_connection(sample, model=self.model)
        self.assertIn(result["prediction"], ("normal", "anomaly"))


if __name__ == "__main__":
    unittest.main()
