"""One-off backfill: predict historical dates so monitoring has overlap with actuals.

Fetches archive weather from Open-Meteo for the date range that exists in the actuals CSV,
runs the latest registered MLflow models, and appends rows to /batch-data/predictions.csv.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import requests

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEATURE_COLS = [
    "wind_speed_kmh",
    "solar_radiation_wm2",
    "day_of_year",
    "month",
    "weekday",
]

# Actuals window
df_act = pd.read_csv("/data/productie_comnbined.csv", na_values=["NULL", "null", ""])
df_act["tijd"] = pd.to_datetime(df_act["tijd"], utc=True)
df_act["datum"] = df_act["tijd"].dt.tz_convert(None).dt.normalize()
start = df_act["datum"].min().strftime("%Y-%m-%d")
end = df_act["datum"].max().strftime("%Y-%m-%d")
print(f"Backfill window: {start} → {end}")

# Archive weather (Antwerp)
r = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": 51.2194,
        "longitude": 4.4025,
        "start_date": start,
        "end_date": end,
        "hourly": "shortwave_radiation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "Europe/Brussels",
    },
    timeout=60,
)
r.raise_for_status()
data = r.json()
hourly = pd.DataFrame(
    {
        "time": pd.to_datetime(data["hourly"]["time"]),
        "shortwave_radiation": data["hourly"]["shortwave_radiation"],
        "wind_speed_10m": data["hourly"]["wind_speed_10m"],
    }
)
hourly["datum"] = hourly["time"].dt.normalize()
daily = (
    hourly.groupby("datum")
    .agg(
        solar_radiation_wm2=("shortwave_radiation", "mean"),
        wind_speed_kmh=("wind_speed_10m", "mean"),
    )
    .reset_index()
).dropna()
daily["day_of_year"] = daily["datum"].dt.day_of_year
daily["month"] = daily["datum"].dt.month
daily["weekday"] = daily["datum"].dt.weekday

X = daily[FEATURE_COLS].astype(
    {"day_of_year": "int32", "month": "int32", "weekday": "int32"}
)

zon = mlflow.pyfunc.load_model("models:/energy-zon-production/latest")
wind = mlflow.pyfunc.load_model("models:/energy-wind-production/latest")

out = daily[["datum"]].copy()
out["zon_mwh_predicted"] = zon.predict(X)
out["wind_mwh_predicted"] = wind.predict(X)
out["predicted_at"] = datetime.now(timezone.utc).isoformat()

out_path = Path("/batch-data/predictions.csv")
if out_path.exists():
    existing = pd.read_csv(out_path, parse_dates=["datum"])
    existing = existing[~existing["datum"].isin(out["datum"])]
    combined = pd.concat([existing, out], ignore_index=True)
else:
    combined = out
combined.sort_values("datum").to_csv(out_path, index=False)
print(f"Saved {len(combined)} rows to {out_path}")
