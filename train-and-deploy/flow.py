"""
Prefect training pipeline for energy production forecasting.

Tasks
─────
  load_wind_data        Load & normalise Wind_final.csv
  load_sun_data         Load & normalise sun_combined.csv
  load_productie_data   Load, aggregate and normalise productie_comnbined.csv
  join_datasets         Inner-join the three DataFrames on 'datum'
  prepare_features      Add calendar features; split into X / y
  train_and_log         Train one (model × hyperparams) combination and log to MLflow
  register_best_model   Register the best-RMSE run in the MLflow Model Registry

Flow
────
  training_pipeline     Orchestrates all tasks; deployable and re-runnable
"""

from __future__ import annotations

import os
import warnings
from itertools import product as cartesian_product

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "/data")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "energy-production-forecasting")

WIND_CSV = os.path.join(DATA_DIR, "Wind_final.csv")
SUN_CSV = os.path.join(DATA_DIR, "sun_combined.csv")
PRODUCTIE_CSV = os.path.join(DATA_DIR, "productie_comnbined.csv")

TARGETS = ["zon", "wind"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FEATURE_COLS = ["wind_speed_kmh", "solar_radiation_wm2"]
CALENDAR_FEATURES = ["day_of_year", "month", "weekday"]

MODELS: dict[str, dict] = {
    "Ridge": {
        "cls": Ridge,
        "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    },
    "RandomForest": {
        "cls": RandomForestRegressor,
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
        },
    },
    "GradientBoosting": {
        "cls": GradientBoostingRegressor,
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
}


# ---------------------------------------------------------------------------
# Tasks — data loading
# ---------------------------------------------------------------------------

@task(name="load-wind-data", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_wind_data(path: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df = df.rename(columns={"date": "datum"})
    df["datum"] = pd.to_datetime(df["datum"]).dt.normalize()
    speed_cols = [c for c in df.columns if "windspeed" in c.lower()]
    df["wind_speed_kmh"] = df[speed_cols].mean(axis=1)
    result = df[["datum", "wind_speed_kmh"]].dropna(subset=["wind_speed_kmh"])
    logger.info("Wind data loaded: %d rows", len(result))
    return result


@task(name="load-sun-data", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_sun_data(path: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df["datum"] = pd.to_datetime(df["datum"]).dt.normalize()
    rad_cols = [c for c in df.columns if "radiation" in c.lower()]
    df["solar_radiation_wm2"] = df[rad_cols].mean(axis=1)
    result = df[["datum", "solar_radiation_wm2"]].dropna(subset=["solar_radiation_wm2"])
    logger.info("Sun data loaded: %d rows", len(result))
    return result


@task(name="load-productie-data", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_productie_data(path: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
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
    agg["zon_mwh"] = (agg["vlaanderen zon kwh"] + agg["elia zon kwh"]) / 1_000
    agg["wind_mwh"] = (agg["vlaanderen wind kwh"] + agg["elia wind kwh"]) / 1_000
    result = agg[["datum", "zon_mwh", "wind_mwh"]]
    logger.info("Productie data loaded: %d daily rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Task — join
# ---------------------------------------------------------------------------

@task(name="join-datasets", cache_policy=NO_CACHE)
def join_datasets(
    prod: pd.DataFrame,
    wind: pd.DataFrame,
    sun: pd.DataFrame,
) -> pd.DataFrame:
    logger = get_run_logger()
    df = prod.merge(wind, on="datum", how="inner")
    df = df.merge(sun, on="datum", how="inner")
    df = df.dropna().sort_values("datum").reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Joined dataset is empty – check that date ranges in the CSVs overlap."
        )
    logger.info(
        "Joined dataset: %d rows | %s – %s",
        len(df),
        df["datum"].min().date(),
        df["datum"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Task — feature preparation
# ---------------------------------------------------------------------------

@task(name="prepare-features", cache_policy=NO_CACHE)
def prepare_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test."""
    logger = get_run_logger()
    data = df.copy()
    data["day_of_year"] = data["datum"].dt.dayofyear
    data["month"] = data["datum"].dt.month
    data["weekday"] = data["datum"].dt.weekday

    X = data[FEATURE_COLS + CALENDAR_FEATURES]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        "Features for '%s': train=%d  test=%d  features=%d",
        target_col, len(X_train), len(X_test), X.shape[1],
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task — train one model variant and log to MLflow
# ---------------------------------------------------------------------------

@task(name="train-and-log", cache_policy=NO_CACHE)
def train_and_log(
    model_name: str,
    hyperparams: dict,
    target_col: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_id: str,
) -> tuple[str, float]:
    """Train, evaluate, log to MLflow. Returns (run_id, test_rmse)."""
    logger = get_run_logger()
    model_cls = MODELS[model_name]["cls"]
    extra = (
        {"random_state": RANDOM_STATE}
        if "random_state" in model_cls.__init__.__code__.co_varnames
        else {}
    )
    estimator = model_cls(**hyperparams, **extra)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    cv_rmse = float(
        -cross_val_score(
            pipe, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=CV_FOLDS,
        ).mean()
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    r2 = float(r2_score(y_test, y_pred))

    hp_str = "__".join(f"{k}={v}" for k, v in hyperparams.items())
    run_name = f"{target_col}__{model_name}__{hp_str}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_param("target", target_col)
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        signature = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        run_id = run.info.run_id

    logger.info(
        "[%s] %s  cv_rmse=%.3f  test_rmse=%.3f  r2=%.3f",
        target_col, model_name, cv_rmse, rmse, r2,
    )
    return run_id, rmse


# ---------------------------------------------------------------------------
# Task — register best model
# ---------------------------------------------------------------------------

@task(name="register-best-model", cache_policy=NO_CACHE)
def register_best_model(
    target_col: str,
    run_results: list[tuple[str, float]],
) -> None:
    logger = get_run_logger()
    best_run_id, best_rmse = min(run_results, key=lambda t: t[1])
    registry_name = f"energy-{target_col.replace('_mwh', '')}-production"
    model_uri = f"runs:/{best_run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)

    client = mlflow.tracking.MlflowClient()
    client.update_registered_model(
        name=registry_name,
        description=(
            f"Best model for predicting {target_col} (MWh/day). "
            f"Test RMSE: {best_rmse:.3f}."
        ),
    )
    logger.info(
        "Registered '%s' version %s (run=%s…, rmse=%.3f)",
        registry_name, mv.version, best_run_id[:8], best_rmse,
    )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(
    name="training-pipeline",
    description="Load energy data, train regression models, log to MLflow.",
)
def training_pipeline(
    targets: list[str] | None = None,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    data_dir: str = DATA_DIR,
) -> None:
    logger = get_run_logger()
    targets = targets or TARGETS

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)

    # Resolve experiment id
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)

    # ── Data loading (can run concurrently) ──────────────────────────────────
    wind_future = load_wind_data.submit(os.path.join(data_dir, "Wind_final.csv"))
    sun_future = load_sun_data.submit(os.path.join(data_dir, "sun_combined.csv"))
    prod_future = load_productie_data.submit(os.path.join(data_dir, "productie_comnbined.csv"))

    df = join_datasets(prod_future.result(), wind_future.result(), sun_future.result())

    # ── Train a model family for each target ─────────────────────────────────
    for target in targets:
        target_col = f"{target}_mwh"
        if target_col not in df.columns:
            logger.warning("Target column '%s' not found – skipping.", target_col)
            continue

        logger.info("=" * 50)
        logger.info("Training models for target: %s", target_col)

        X_train, X_test, y_train, y_test = prepare_features(df, target_col)

        run_results: list[tuple[str, float]] = []

        for model_name, config in MODELS.items():
            param_names = list(config["params"].keys())
            param_values = list(config["params"].values())

            for combo in cartesian_product(*param_values):
                hp = dict(zip(param_names, combo))
                run_id, rmse = train_and_log(
                    model_name=model_name,
                    hyperparams=hp,
                    target_col=target_col,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    experiment_id=experiment_id,
                )
                run_results.append((run_id, rmse))

        register_best_model(target_col, run_results)

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _wait_for_prefect(api_url: str, retries: int = 12, delay: int = 10) -> None:
    """Block until the Prefect API responds or raise after `retries` attempts."""
    import time
    import urllib.request
    import urllib.error

    health_url = api_url.rstrip("/").replace("/api", "") + "/api/health"
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlopen(health_url, timeout=5)
            print(f"[prefect] API ready at {api_url}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[prefect] Waiting for API ({attempt}/{retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Prefect API not reachable after {retries * delay}s: {api_url}")


if __name__ == "__main__":
    _api = os.getenv("PREFECT_API_URL", "")
    if _api:
        _wait_for_prefect(_api)
    training_pipeline()
