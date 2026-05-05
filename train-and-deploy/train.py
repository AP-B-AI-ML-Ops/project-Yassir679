"""
Train regression models to predict solar and wind production (MWh/day).

Data sources (all in /data):
  - Wind_final.csv       : daily wind speed (km/h) – 'date' column
  - sun_combined.csv     : daily solar radiation (W/m²) – 'datum' column
  - productie_comnbined.csv : hourly production (kWh) – 'tijd' column

Pipeline:
  1. Load & join the three datasets on date.
  2. Prepare feature matrix X and two target vectors (zon, wind).
  3. Grid-search over three regressors.
  4. Log every run to MLflow; register the best model per target.
"""

import os
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "/data")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "energy-production-forecasting")

WIND_CSV = os.path.join(DATA_DIR, "Wind_final.csv")
SUN_CSV = os.path.join(DATA_DIR, "sun_combined.csv")
PRODUCTIE_CSV = os.path.join(DATA_DIR, "productie_comnbined.csv")

# Which production target to predict: "zon" or "wind" (or both).
TARGETS = ["zon", "wind"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 1. Load & join
# ---------------------------------------------------------------------------

def load_wind(path: str) -> pd.DataFrame:
    """Load daily wind speed data and return a normalised DataFrame."""
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df = df.rename(columns={"date": "datum"})
    df["datum"] = pd.to_datetime(df["datum"]).dt.normalize()

    speed_cols = [c for c in df.columns if "windspeed" in c.lower()]
    df["wind_speed_kmh"] = df[speed_cols].mean(axis=1)
    return df[["datum", "wind_speed_kmh"]].dropna(subset=["wind_speed_kmh"])


def load_sun(path: str) -> pd.DataFrame:
    """Load daily solar radiation data and return a normalised DataFrame."""
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df["datum"] = pd.to_datetime(df["datum"]).dt.normalize()

    rad_cols = [c for c in df.columns if "radiation" in c.lower()]
    df["solar_radiation_wm2"] = df[rad_cols].mean(axis=1)
    return df[["datum", "solar_radiation_wm2"]].dropna(subset=["solar_radiation_wm2"])


def load_productie(path: str) -> pd.DataFrame:
    """
    Load hourly production data (kWh), aggregate to daily MWh totals,
    and return a normalised DataFrame.
    """
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df["tijd"] = pd.to_datetime(df["tijd"], utc=True)
    df["datum"] = df["tijd"].dt.tz_convert(None).dt.normalize()

    # Sum hourly kWh → daily kWh → convert to MWh
    agg = (
        df.groupby("datum")[
            ["vlaanderen zon kwh", "vlaanderen wind kwh",
             "elia zon kwh", "elia wind kwh"]
        ]
        .sum()
        .reset_index()
    )

    # Combine Vlaanderen + Elia totals
    agg["zon_mwh"] = (agg["vlaanderen zon kwh"] + agg["elia zon kwh"]) / 1_000
    agg["wind_mwh"] = (agg["vlaanderen wind kwh"] + agg["elia wind kwh"]) / 1_000

    return agg[["datum", "zon_mwh", "wind_mwh"]]


def build_dataset() -> pd.DataFrame:
    """Join wind, sun, and productie on 'datum'; drop rows with missing values."""
    wind = load_wind(WIND_CSV)
    sun = load_sun(SUN_CSV)
    prod = load_productie(PRODUCTIE_CSV)

    df = prod.merge(wind, on="datum", how="inner")
    df = df.merge(sun, on="datum", how="inner")
    df = df.dropna()
    df = df.sort_values("datum").reset_index(drop=True)

    print(f"[data] Joined dataset: {len(df)} rows, date range "
          f"{df['datum'].min().date()} – {df['datum'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = ["wind_speed_kmh", "solar_radiation_wm2"]

# Simple calendar features that help tree-based models capture seasonality.
CALENDAR_FEATURES = ["day_of_year", "month", "weekday"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_year"] = df["datum"].dt.dayofyear
    df["month"] = df["datum"].dt.month
    df["weekday"] = df["datum"].dt.weekday
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    return df[FEATURE_COLS + CALENDAR_FEATURES]


# ---------------------------------------------------------------------------
# 3. Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    "Ridge": {
        "model": Ridge(),
        "params": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    },
}


# ---------------------------------------------------------------------------
# 4. Training & MLflow tracking
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }


def train_target(target_col: str, df: pd.DataFrame, experiment_id: str):
    """
    Run a grid search across all models for a single target.
    Each (model, hyperparameter set) combination is logged as its own MLflow run.
    The best run is registered in the model registry.
    """
    X = prepare_features(df)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    best_run_id = None
    best_rmse = float("inf")
    best_model_name = None

    for model_name, config in MODELS.items():
        # Build a flat list of all hyperparameter combinations.
        from itertools import product as cartesian_product

        param_names = list(config["params"].keys())
        param_values = list(config["params"].values())

        for combo in cartesian_product(*param_values):
            hp = dict(zip(param_names, combo))

            # Build sklearn pipeline: scaler + model with these hyperparameters.
            estimator = config["model"].__class__(
                **{k.replace("model__", ""): v for k, v in hp.items()},
                **({"random_state": RANDOM_STATE}
                   if hasattr(config["model"], "random_state") else {}),
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

            cv_rmse = -cross_val_score(
                pipe, X_train, y_train,
                scoring="neg_root_mean_squared_error",
                cv=CV_FOLDS,
            ).mean()

            pipe.fit(X_train, y_train)
            test_metrics = evaluate(pipe, X_test, y_test)

            run_name = (
                f"{target_col}__{model_name}__"
                + "__".join(f"{k.replace('model__', '')}={v}" for k, v in hp.items())
            )

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                # Log parameters
                mlflow.log_param("target", target_col)
                mlflow.log_param("model", model_name)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                for k, v in hp.items():
                    mlflow.log_param(k.replace("model__", ""), v)

                # Log metrics
                mlflow.log_metric("cv_rmse", cv_rmse)
                mlflow.log_metric("test_mae", test_metrics["mae"])
                mlflow.log_metric("test_rmse", test_metrics["rmse"])
                mlflow.log_metric("test_r2", test_metrics["r2"])

                # Log model artifact with input/output signature
                signature = infer_signature(X_train, pipe.predict(X_train))
                mlflow.sklearn.log_model(pipe, artifact_path="model",
                                         signature=signature,
                                         input_example=X_train.head(3))

                print(
                    f"  [{target_col}] {model_name:20s} "
                    f"cv_rmse={cv_rmse:.3f}  "
                    f"test_rmse={test_metrics['rmse']:.3f}  "
                    f"r2={test_metrics['r2']:.3f}"
                )

                if test_metrics["rmse"] < best_rmse:
                    best_rmse = test_metrics["rmse"]
                    best_run_id = run.info.run_id
                    best_model_name = model_name

    # Register the best model
    if best_run_id:
        registry_name = f"energy-{target_col}-production"
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(
            name=registry_name,
            description=(
                f"Best model for predicting {target_col} production (MWh/day). "
                f"Algorithm: {best_model_name}. Test RMSE: {best_rmse:.3f}."
            ),
        )
        print(
            f"\n[registry] Registered '{registry_name}' "
            f"version {mv.version} (run={best_run_id[:8]}..., "
            f"best_model={best_model_name}, test_rmse={best_rmse:.3f})"
        )


# ---------------------------------------------------------------------------
# 5. Entrypoint
# ---------------------------------------------------------------------------

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"[mlflow] Tracking URI: {MLFLOW_TRACKING_URI}")

    # Create (or reuse) the experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"[mlflow] Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[mlflow] Using experiment '{EXPERIMENT_NAME}' (id={experiment_id})")

    # Build dataset
    df = build_dataset()

    if df.empty:
        raise ValueError(
            "Joined dataset is empty. Check that the date ranges in the CSVs overlap."
        )

    # Train one model family per production target
    for target in TARGETS:
        target_col = f"{target}_mwh"
        if target_col not in df.columns:
            print(f"[warn] Target column '{target_col}' not found – skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"  Training models for target: {target_col}")
        print(f"{'='*60}")
        train_target(target_col, df, experiment_id)

    print("\n[done] All experiments completed.")


if __name__ == "__main__":
    main()
