import os
import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from evidently import Dataset, DataDefinition, Regression, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount
from evidently.presets import DataSummaryPreset
from prefect import flow, task

load_dotenv()

BATCH_DATA_DIR = os.getenv("BATCH_DATA_DIR", "/batch-data")
PRODUCTIE_CSV = os.getenv("PRODUCTIE_CSV", "/data/productie_comnbined.csv")
DB_URI = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'mlflow')}:{os.getenv('POSTGRES_PASSWORD', 'password')}"
    "@database/metrics"
)


@task
def load_joined_data() -> pd.DataFrame:
    pred_path = Path(BATCH_DATA_DIR) / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions file found at {pred_path}")

    predictions = pd.read_csv(pred_path, parse_dates=["datum"])

    df = pd.read_csv(PRODUCTIE_CSV, na_values=["NULL", "null", ""])
    df["tijd"] = pd.to_datetime(df["tijd"], utc=True)
    df["datum"] = df["tijd"].dt.tz_convert(None).dt.normalize()
    agg = (
        df.groupby("datum")[
            ["vlaanderen zon kwh", "vlaanderen wind kwh", "elia zon kwh", "elia wind kwh"]
        ]
        .sum()
        .reset_index()
    )
    agg["zon_mwh_actual"] = (agg["vlaanderen zon kwh"] + agg["elia zon kwh"]) / 1_000
    agg["wind_mwh_actual"] = (agg["vlaanderen wind kwh"] + agg["elia wind kwh"]) / 1_000
    actuals = agg[["datum", "zon_mwh_actual", "wind_mwh_actual"]].dropna()

    joined = predictions.merge(actuals, on="datum", how="inner").sort_values("datum")
    print(f"Loaded {len(joined)} rows of predictions joined with actuals")
    return joined


@task
def run_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame):
    definition = DataDefinition(
        numerical_columns=[
            "zon_mwh_predicted", "zon_mwh_actual",
            "wind_mwh_predicted", "wind_mwh_actual",
        ],
        regression=[Regression(target="zon_mwh_actual", prediction="zon_mwh_predicted")],
    )
    report = Report([
        ValueDrift(column="zon_mwh_predicted"),
        ValueDrift(column="wind_mwh_predicted"),
        DriftedColumnsCount(),
        DataSummaryPreset(),
    ])
    return report.run(
        Dataset.from_pandas(df_cur, data_definition=definition),
        Dataset.from_pandas(df_ref, data_definition=definition),
    )


@task
def store_metrics(run, df_cur: pd.DataFrame, run_time: datetime.datetime) -> pd.DataFrame:
    json_data = run.dict()
    rows = [
        {
            "run_time": run_time,
            "metric_name": metric["metric_id"],
            "value": str(metric["value"]),
        }
        for metric in json_data["metrics"]
    ]

    # Add explicit RMSE metrics with predictable names for Grafana dashboards
    zon_rmse = float(
        mean_squared_error(df_cur["zon_mwh_actual"], df_cur["zon_mwh_predicted"]) ** 0.5
    )
    wind_rmse = float(
        mean_squared_error(df_cur["wind_mwh_actual"], df_cur["wind_mwh_predicted"]) ** 0.5
    )
    rows += [
        {"run_time": run_time, "metric_name": "zon_rmse", "value": str(zon_rmse)},
        {"run_time": run_time, "metric_name": "wind_rmse", "value": str(wind_rmse)},
    ]

    df = pd.DataFrame(rows)

    if not database_exists(DB_URI):
        create_database(DB_URI)

    engine = create_engine(DB_URI)
    df.to_sql("evidently_metrics", engine, if_exists="append", index=False)
    print(f"Stored {len(df)} metrics rows")
    return df


@flow
def monitoring_flow():
    joined = load_joined_data()

    if len(joined) < 4:
        print(f"Not enough data ({len(joined)} rows) – need at least 4 for ref/current split.")
        return

    mid = len(joined) // 2
    df_ref = joined.iloc[:mid].copy()
    df_cur = joined.iloc[mid:].copy()

    run = run_report(df_ref, df_cur)
    run.save_html(str(Path(BATCH_DATA_DIR) / "evidently_report.html"))

    metrics = store_metrics(run, df_cur, datetime.datetime.utcnow())
    return metrics


if __name__ == "__main__":
    monitoring_flow.serve(
        name="energy-monitoring",
        cron="30 6 * * *",  # daily at 06:30 UTC, after the batch pipeline at 06:00
    )
