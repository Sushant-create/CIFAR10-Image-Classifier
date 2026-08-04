"""
main.py
--------
Flask web app for the CIFAR-10 CNN Image Classifier.

Serves a simple upload page (templates/index.html). The user uploads an
image, it gets resized to 32x32 and normalized the same way as during
training, then the saved model (model/cifar10_cnn_model.h5) predicts one
of the 10 CIFAR-10 classes.

Run:
    python main.py
or use the launch script:
    ./launch.sh      (Linux/Mac)
    launch.bat       (Windows)
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image

MODEL_PATH = os.path.join("model", "cifar10_cnn_model.h5")
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
IMG_SIZE = (32, 32)

app = Flask(__name__)

model = None
model_load_error = None


def load_model():
    """Lazily loads the Keras model, so the app can still start (and show
    a friendly error) even if training hasn't been run yet."""
    global model, model_load_error
    if model is not None or model_load_error is not None:
        return
    try:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"'{MODEL_PATH}' not found. Run 'python train_model.py' "
                f"or the CIFAR-10.ipynb notebook first to generate it."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        model_load_error = str(e)


def preprocess_image(file_bytes):
    """Resizes an uploaded image to 32x32 RGB and normalizes to [0, 1],
    matching the preprocessing used during training."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)
    return arr, img


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    if model_load_error:
        return jsonify({"error": model_load_error}), 503

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    try:
        file_bytes = file.read()
        arr, img = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    result = {
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(float(preds[top_idx]) * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(preds)
        }
    }

    # send back a small preview of the resized image for the UI
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result["preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    load_model()
    if model_load_error:
        return jsonify({"status": "model_not_loaded", "error": model_load_error}), 503
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_model()
    if model_load_error:
        print(f"[WARNING] {model_load_error}")
    app.run(debug=True, host="0.0.0.0", port=5000)
