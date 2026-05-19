"""
Batch Inference Pipeline

Prefect flow that runs on a daily schedule (06:00 UTC):

  1. fetch_weather_forecast   – Pull 7-day ECMWF forecast from Open-Meteo for Antwerp
  2. run_inference             – Load registered MLflow models; predict zon_mwh & wind_mwh
  3. save_predictions          – Append predictions to /batch-data/predictions.csv
  4. load_actuals              – Read historical Elia/Vlaanderen production actuals
  5. compute_metrics           – Join past predictions with actuals; compute RMSE per target
  6. save_metrics              – Append RMSE row to /batch-data/metrics.csv
  7. check_retraining_threshold– If RMSE > threshold, trigger the training-pipeline deployment
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.deployments import run_deployment
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Configuration  (all overridable via environment variables)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
ZON_MODEL_NAME = os.getenv("ZON_MODEL_NAME", "energy-zon-production")
WIND_MODEL_NAME = os.getenv("WIND_MODEL_NAME", "energy-wind-production")

PRODUCTIE_CSV = os.getenv("PRODUCTIE_CSV", "/data/productie_comnbined.csv")
BATCH_DATA_DIR = os.getenv("BATCH_DATA_DIR", "/batch-data")

# RMSE thresholds above which retraining is triggered (MWh/day)
RMSE_THRESHOLD_ZON = float(os.getenv("RMSE_THRESHOLD_ZON", "500"))
RMSE_THRESHOLD_WIND = float(os.getenv("RMSE_THRESHOLD_WIND", "300"))

# Name of the Prefect deployment to trigger for retraining
RETRAIN_DEPLOYMENT = os.getenv("RETRAIN_DEPLOYMENT", "training-pipeline/energy-training")

# Open-Meteo / Antwerp settings
LATITUDE = 51.2194
LONGITUDE = 4.4025
FORECAST_DAYS = 7

FEATURE_COLS = [
    "wind_speed_kmh",
    "solar_radiation_wm2",
    "day_of_year",
    "month",
    "weekday",
]


# ---------------------------------------------------------------------------
# Task 1 – Fetch weather forecast from Open-Meteo (ECMWF)
# ---------------------------------------------------------------------------

@task(name="fetch-weather-forecast", retries=3, retry_delay_seconds=15, cache_policy=NO_CACHE)
def fetch_weather_forecast() -> pd.DataFrame:
    """Call the Open-Meteo ECMWF API and return daily aggregated weather features."""
    logger = get_run_logger()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "shortwave_radiation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "forecast_days": FORECAST_DAYS,
        "timezone": "Europe/Brussels",
        "models": "ecmwf_ifs04",
    }

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    hourly = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "shortwave_radiation": data["hourly"]["shortwave_radiation"],
        "wind_speed_10m": data["hourly"]["wind_speed_10m"],
    })

    hourly["datum"] = hourly["time"].dt.normalize()
    daily = (
        hourly.groupby("datum")
        .agg(
            solar_radiation_wm2=("shortwave_radiation", "mean"),
            wind_speed_kmh=("wind_speed_10m", "mean"),
        )
        .reset_index()
    )

    daily["day_of_year"] = daily["datum"].dt.day_of_year
    daily["month"] = daily["datum"].dt.month
    daily["weekday"] = daily["datum"].dt.weekday

    logger.info("Fetched %d days of ECMWF forecast data from Open-Meteo.", len(daily))
    return daily


# ---------------------------------------------------------------------------
# Task 2 – Run inference with the registered MLflow models
# ---------------------------------------------------------------------------

@task(name="run-inference", cache_policy=NO_CACHE)
def run_inference(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Load the latest registered MLflow models and predict solar & wind production."""
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    zon_model = mlflow.pyfunc.load_model(f"models:/{ZON_MODEL_NAME}/latest")
    wind_model = mlflow.pyfunc.load_model(f"models:/{WIND_MODEL_NAME}/latest")

    X = forecast_df[FEATURE_COLS]

    predictions = forecast_df[["datum"]].copy()
    predictions["zon_mwh_predicted"] = zon_model.predict(X)
    predictions["wind_mwh_predicted"] = wind_model.predict(X)
    predictions["predicted_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("Inference done for %d days.", len(predictions))
    return predictions


# ---------------------------------------------------------------------------
# Task 3 – Persist predictions
# ---------------------------------------------------------------------------

@task(name="save-predictions", cache_policy=NO_CACHE)
def save_predictions(predictions: pd.DataFrame) -> None:
    """Upsert predictions into /batch-data/predictions.csv (replace rows by date)."""
    logger = get_run_logger()

    out_path = Path(BATCH_DATA_DIR) / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["datum"])
        # Remove stale rows for dates that are being re-predicted
        existing = existing[~existing["datum"].isin(predictions["datum"])]
        combined = pd.concat([existing, predictions], ignore_index=True)
    else:
        combined = predictions

    combined.sort_values("datum").to_csv(out_path, index=False)
    logger.info("Predictions saved to %s (%d total rows).", out_path, len(combined))


# ---------------------------------------------------------------------------
# Task 4 – Load Elia / Vlaanderen actuals
# ---------------------------------------------------------------------------

@task(name="load-actuals", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_actuals() -> pd.DataFrame:
    """Aggregate hourly production CSV to daily MWh totals."""
    logger = get_run_logger()

    df = pd.read_csv(PRODUCTIE_CSV, na_values=["NULL", "null", ""])
    df["tijd"] = pd.to_datetime(df["tijd"], utc=True)
    df["datum"] = df["tijd"].dt.tz_convert(None).dt.normalize()

    agg = (
        df.groupby("datum")[
            ["vlaanderen zon kwh", "vlaanderen wind kwh",
             "elia zon kwh", "elia wind kwh"]
        ]
        .sum()
        .reset_index()
    )
    agg["zon_mwh_actual"] = (agg["vlaanderen zon kwh"] + agg["elia zon kwh"]) / 1_000
    agg["wind_mwh_actual"] = (agg["vlaanderen wind kwh"] + agg["elia wind kwh"]) / 1_000

    result = agg[["datum", "zon_mwh_actual", "wind_mwh_actual"]].dropna()
    logger.info("Loaded %d days of actuals from %s.", len(result), PRODUCTIE_CSV)
    return result


# ---------------------------------------------------------------------------
# Task 5 – Compute RMSE metrics
# ---------------------------------------------------------------------------

@task(name="compute-metrics", cache_policy=NO_CACHE)
def compute_metrics(actuals: pd.DataFrame) -> dict | None:
    """
    Join saved predictions with actuals on 'datum'.
    Returns a metrics dict, or None if there is not enough overlap.
    """
    logger = get_run_logger()

    pred_path = Path(BATCH_DATA_DIR) / "predictions.csv"
    if not pred_path.exists():
        logger.warning("No predictions file found at %s – skipping metrics.", pred_path)
        return None

    predictions = pd.read_csv(pred_path, parse_dates=["datum"])
    joined = predictions.merge(actuals, on="datum", how="inner")

    if len(joined) < 2:
        logger.warning(
            "Only %d overlapping dates between predictions and actuals "
            "– need at least 2 for RMSE.", len(joined),
        )
        return None

    zon_rmse = float(
        mean_squared_error(joined["zon_mwh_actual"], joined["zon_mwh_predicted"]) ** 0.5
    )
    wind_rmse = float(
        mean_squared_error(joined["wind_mwh_actual"], joined["wind_mwh_predicted"]) ** 0.5
    )

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_days": len(joined),
        "zon_rmse": round(zon_rmse, 3),
        "wind_rmse": round(wind_rmse, 3),
    }
    logger.info(
        "Metrics: zon_rmse=%.3f  wind_rmse=%.3f  (n=%d days)",
        zon_rmse, wind_rmse, len(joined),
    )
    return metrics


# ---------------------------------------------------------------------------
# Task 6 – Save metrics
# ---------------------------------------------------------------------------

@task(name="save-metrics", cache_policy=NO_CACHE)
def save_metrics(metrics: dict) -> None:
    """Append one row to /batch-data/metrics.csv."""
    logger = get_run_logger()

    out_path = Path(BATCH_DATA_DIR) / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([metrics])
    write_header = not out_path.exists()
    row.to_csv(out_path, mode="a", header=write_header, index=False)
    logger.info("Metrics row appended to %s.", out_path)


# ---------------------------------------------------------------------------
# Task 7 – Check retraining threshold
# ---------------------------------------------------------------------------

@task(name="check-retraining-threshold", cache_policy=NO_CACHE)
def check_retraining_threshold(metrics: dict) -> None:
    """Trigger the training-pipeline deployment when RMSE exceeds a threshold."""
    logger = get_run_logger()

    needs_retrain = False

    if metrics["zon_rmse"] > RMSE_THRESHOLD_ZON:
        logger.warning(
            "Solar RMSE %.3f > threshold %.3f – retraining needed.",
            metrics["zon_rmse"], RMSE_THRESHOLD_ZON,
        )
        needs_retrain = True

    if metrics["wind_rmse"] > RMSE_THRESHOLD_WIND:
        logger.warning(
            "Wind RMSE %.3f > threshold %.3f – retraining needed.",
            metrics["wind_rmse"], RMSE_THRESHOLD_WIND,
        )
        needs_retrain = True

    if needs_retrain:
        try:
            run_deployment(name=RETRAIN_DEPLOYMENT, timeout=0)
            logger.info("Retraining deployment triggered: %s", RETRAIN_DEPLOYMENT)
        except Exception as exc:
            logger.error("Could not trigger retraining deployment: %s", exc)
    else:
        logger.info(
            "RMSE within thresholds – no retraining needed "
            "(zon=%.3f/%.3f, wind=%.3f/%.3f).",
            metrics["zon_rmse"], RMSE_THRESHOLD_ZON,
            metrics["wind_rmse"], RMSE_THRESHOLD_WIND,
        )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(
    name="batch-inference-pipeline",
    description=(
        "Daily batch: fetch ECMWF forecast, run inference, "
        "compute RMSE vs actuals, trigger retraining if needed."
    ),
)
def batch_inference_pipeline() -> None:
    logger = get_run_logger()
    logger.info("=== Batch inference pipeline started ===")

    forecast_df = fetch_weather_forecast()
    predictions = run_inference(forecast_df)
    save_predictions(predictions)

    actuals = load_actuals()
    metrics = compute_metrics(actuals)

    if metrics is not None:
        save_metrics(metrics)
        check_retraining_threshold(metrics)
    else:
        logger.info("Skipping metrics / retraining check (insufficient data).")

    logger.info("=== Batch inference pipeline finished ===")


# ---------------------------------------------------------------------------
# Entry point – serve the flow so Prefect can schedule it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch_inference_pipeline.serve(
        name="energy-batch",
        cron="0 6 * * *",   # every day at 06:00 UTC
    )
