from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
ZON_MODEL_NAME = os.getenv("ZON_MODEL_NAME", "energy-zon-production")
WIND_MODEL_NAME = os.getenv("WIND_MODEL_NAME", "energy-wind-production")

FEATURE_COLS = [
    "wind_speed_kmh",
    "solar_radiation_wm2",
    "day_of_year",
    "month",
    "weekday",
]

# ---------------------------------------------------------------------------
# Lazy model loading (loaded on first request, not at import time)
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_models: dict = {}


def get_models() -> dict:
    """Load models from MLflow registry on first call, then cache them."""
    if not _models:
        for _name, _key in [(ZON_MODEL_NAME, "zon"), (WIND_MODEL_NAME, "wind")]:
            _models[_key] = mlflow.pyfunc.load_model(f"models:/{_name}/latest")
    return _models


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": list(_models.keys())})


@app.post("/predict")
def predict():
    try:
        models = get_models()
    except Exception as exc:
        return jsonify({"error": f"Models not available: {exc}"}), 503

    data = request.get_json(force=True)
    if not data or "forecasts" not in data:
        return jsonify({"error": "Missing 'forecasts' key"}), 400

    forecasts = data["forecasts"]
    if not isinstance(forecasts, list) or len(forecasts) == 0:
        return jsonify({"error": "'forecasts' must be a non-empty list"}), 400
    if len(forecasts) > 30:
        return jsonify({"error": "'forecasts' may contain at most 30 items"}), 400

    rows = []
    for f in forecasts:
        try:
            d = pd.Timestamp(f["date"])
            rows.append(
                {
                    "wind_speed_kmh": float(f["wind_speed_kmh"]),
                    "solar_radiation_wm2": float(f["solar_radiation_wm2"]),
                    "day_of_year": d.day_of_year,
                    "month": d.month,
                    "weekday": d.weekday(),
                }
            )
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"Invalid forecast entry: {exc}"}), 400

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    X = X.astype({"day_of_year": "int32", "month": "int32", "weekday": "int32"})
    zon_preds = models["zon"].predict(X)
    wind_preds = models["wind"].predict(X)

    predictions = [
        {
            "date": forecasts[i]["date"],
            "zon_mwh": round(float(zon_preds[i]), 3),
            "wind_mwh": round(float(wind_preds[i]), 3),
        }
        for i in range(len(forecasts))
    ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
