"""
test_app.py
------------
Automated tests for the Flask app (main.py).

Uses a small dummy model (same architecture, untrained) instead of the real
trained model, so these tests run fast and don't depend on the CIFAR-10
dataset being downloaded or the real model file existing.

Run:
    python -m unittest test_app.py -v
"""

import io
import unittest
import numpy as np
from PIL import Image

import main as app_module


def make_dummy_model():
    """A tiny untrained model with the correct input/output shape, just
    to exercise the Flask request/response pipeline end-to-end."""
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return m


def make_test_image_bytes(size=(64, 64), color=(120, 60, 200)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class CifarAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # inject a dummy model so tests don't require training first
        app_module.model = make_dummy_model()
        app_module.model_load_error = None
        cls.client = app_module.app.test_client()

    def test_index_page_loads(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"CIFAR-10 Classifier", resp.data)

    def test_health_endpoint_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_predict_with_valid_image(self):
        img_bytes = make_test_image_bytes()
        data = {"image": (img_bytes, "test.png")}
        resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)

        body = resp.get_json()
        self.assertIn("prediction", body)
        self.assertIn("confidence", body)
        self.assertIn("all_probabilities", body)
        self.assertEqual(len(body["all_probabilities"]), 10)
        self.assertIn(body["prediction"], app_module.CLASS_NAMES)

    def test_predict_with_no_file(self):
        resp = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_predict_with_non_image_file(self):
        bad_file = (io.BytesIO(b"not an image"), "notes.txt")
        data = {"image": bad_file}
        resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_predict_probabilities_sum_to_roughly_one(self):
        img_bytes = make_test_image_bytes(color=(10, 200, 40))
        data = {"image": (img_bytes, "test2.png")}
        resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
        probs = resp.get_json()["all_probabilities"]
        total = sum(probs.values())
        self.assertAlmostEqual(total, 100.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
