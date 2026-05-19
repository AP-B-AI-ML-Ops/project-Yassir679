"""
Web Service – Energy Production Forecast API

Endpoints
─────────
  GET  /health    Liveness check; reports which models are loaded.
  POST /predict   Accepts 1–30 days of weather forecasts and returns
                  predicted solar and wind energy production (MWh/day).

Expected input per day
──────────────────────
  date               : YYYY-MM-DD
  wind_speed_kmh     : daily-average wind speed (km/h)
  solar_radiation_wm2: daily-average solar radiation (W/m²)

The API loads the latest versions of the registered MLflow models
  energy-zon-production   → solar predictions
  energy-wind-production  → wind predictions
on startup.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
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
# Global model store (populated at startup)
# ---------------------------------------------------------------------------
_models: dict[str, mlflow.pyfunc.PyFuncModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both MLflow models once when the container starts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    for registry_name, key in [
        (ZON_MODEL_NAME, "zon"),
        (WIND_MODEL_NAME, "wind"),
    ]:
        uri = f"models:/{registry_name}/latest"
        try:
            _models[key] = mlflow.pyfunc.load_model(uri)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{registry_name}' from {MLFLOW_TRACKING_URI}: {exc}"
            ) from exc

    yield

    _models.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Energy Production Forecast API",
    description=(
        "Predicts solar and wind energy production (MWh/day) "
        "for the Antwerp region based on weather forecast data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class WeatherForecast(BaseModel):
    date: date = Field(..., description="Forecast date (YYYY-MM-DD)")
    wind_speed_kmh: float = Field(
        ..., ge=0, description="Daily-average wind speed (km/h)"
    )
    solar_radiation_wm2: float = Field(
        ..., ge=0, description="Daily-average solar radiation (W/m²)"
    )


class PredictionRequest(BaseModel):
    forecasts: list[WeatherForecast] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Between 1 and 30 days of weather forecasts",
    )


class DayPrediction(BaseModel):
    date: date
    zon_mwh: float = Field(..., description="Predicted solar production (MWh/day)")
    wind_mwh: float = Field(..., description="Predicted wind production (MWh/day)")


class PredictionResponse(BaseModel):
    predictions: list[DayPrediction]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if len(_models) < 2:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    rows = [
        {
            "wind_speed_kmh": f.wind_speed_kmh,
            "solar_radiation_wm2": f.solar_radiation_wm2,
            "day_of_year": pd.Timestamp(f.date).day_of_year,
            "month": pd.Timestamp(f.date).month,
            "weekday": pd.Timestamp(f.date).weekday(),
        }
        for f in request.forecasts
    ]

    X = pd.DataFrame(rows, columns=FEATURE_COLS)

    zon_preds = _models["zon"].predict(X)
    wind_preds = _models["wind"].predict(X)

    predictions = [
        DayPrediction(
            date=request.forecasts[i].date,
            zon_mwh=round(float(zon_preds[i]), 3),
            wind_mwh=round(float(wind_preds[i]), 3),
        )
        for i in range(len(request.forecasts))
    ]

    return PredictionResponse(predictions=predictions)
