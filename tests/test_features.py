import pandas as pd
import pytest


# Functions copied from train-and-deploy/train.py (no mlflow needed here)

def add_calendar_features(df):
    df = df.copy()
    df["day_of_year"] = df["datum"].dt.dayofyear
    df["month"]       = df["datum"].dt.month
    df["weekday"]     = df["datum"].dt.weekday
    return df


def load_wind(path):
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df = df.rename(columns={"date": "datum"})
    df["datum"] = pd.to_datetime(df["datum"]).dt.normalize()
    speed_cols = [c for c in df.columns if "windspeed" in c.lower()]
    df["wind_speed_kmh"] = df[speed_cols].mean(axis=1)
    return df[["datum", "wind_speed_kmh"]].dropna(subset=["wind_speed_kmh"])


def load_productie(path):
    df = pd.read_csv(path, na_values=["NULL", "null", ""])
    df["tijd"] = pd.to_datetime(df["tijd"], utc=True)
    df["datum"] = df["tijd"].dt.tz_convert(None).dt.normalize()
    cols = ["vlaanderen zon kwh", "vlaanderen wind kwh", "elia zon kwh", "elia wind kwh"]
    agg = df.groupby("datum")[cols].sum().reset_index()
    agg["zon_mwh"]  = (agg["vlaanderen zon kwh"] + agg["elia zon kwh"])  / 1000
    agg["wind_mwh"] = (agg["vlaanderen wind kwh"] + agg["elia wind kwh"]) / 1000
    return agg[["datum", "zon_mwh", "wind_mwh"]]


# --- tests ---

def test_calendar_columns_added():
    df = pd.DataFrame({"datum": pd.to_datetime(["2023-03-15"])})
    result = add_calendar_features(df)
    assert "day_of_year" in result.columns
    assert "month"       in result.columns
    assert "weekday"     in result.columns


def test_calendar_day_of_year():
    df = pd.DataFrame({"datum": pd.to_datetime(["2023-03-15", "2023-06-21"])})
    result = add_calendar_features(df)
    assert result.loc[0, "day_of_year"] == 74   # 15 maart
    assert result.loc[1, "day_of_year"] == 172  # 21 juni


def test_calendar_does_not_mutate_input():
    df = pd.DataFrame({"datum": pd.to_datetime(["2023-01-01"])})
    add_calendar_features(df)
    assert "day_of_year" not in df.columns


def test_load_wind_columns(tmp_path):
    p = tmp_path / "wind.csv"
    p.write_text("date,windspeed_uccle,windspeed_antwerp\n2023-01-01,15.0,12.5\n")
    assert list(load_wind(str(p)).columns) == ["datum", "wind_speed_kmh"]


def test_load_wind_averages_stations(tmp_path):
    p = tmp_path / "wind.csv"
    p.write_text("date,windspeed_uccle,windspeed_antwerp\n2023-01-01,15.0,12.5\n")
    assert load_wind(str(p)).loc[0, "wind_speed_kmh"] == pytest.approx(13.75)


def test_load_wind_drops_null_rows(tmp_path):
    p = tmp_path / "wind.csv"
    p.write_text("date,windspeed_uccle,windspeed_antwerp\nNULL,NULL,NULL\n2023-01-02,10.0,8.0\n")
    assert len(load_wind(str(p))) == 1


def test_load_productie_aggregates_to_daily(tmp_path):
    p = tmp_path / "prod.csv"
    p.write_text(
        "tijd,vlaanderen zon kwh,vlaanderen wind kwh,elia zon kwh,elia wind kwh\n"
        "2023-01-01T00:00:00+00:00,1000,2000,500,800\n"
        "2023-01-01T01:00:00+00:00,1000,2000,500,800\n"
        "2023-01-02T00:00:00+00:00,900,1800,400,700\n"
    )
    assert len(load_productie(str(p))) == 2


def test_load_productie_kwh_to_mwh(tmp_path):
    p = tmp_path / "prod.csv"
    p.write_text(
        "tijd,vlaanderen zon kwh,vlaanderen wind kwh,elia zon kwh,elia wind kwh\n"
        "2023-01-01T00:00:00+00:00,1000,2000,500,800\n"
        "2023-01-01T01:00:00+00:00,1100,2100,550,850\n"
    )
    df = load_productie(str(p))
    row = df[df["datum"] == pd.Timestamp("2023-01-01")].iloc[0]
    assert row["zon_mwh"]  == pytest.approx(3.15)   # (1000+1100+500+550)/1000
    assert row["wind_mwh"] == pytest.approx(5.75)   # (2000+2100+800+850)/1000
