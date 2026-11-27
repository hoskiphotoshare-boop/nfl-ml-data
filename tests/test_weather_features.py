import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import polars as pl
from weather_features import (
    WEATHER_FEATURES,
    attach_weather,
    forecast_weather,
    WEATHER_FEATURES as WF,
)


def test_forecast_dome_neutral():
    out = forecast_weather(temp=20, wind=25, gust=35, snow=2, precip_mm=3, roof="dome")
    assert out["wx_outdoor"] == 0
    assert out["wx_temp"] == 60.0
    assert out["wx_wind"] == 0.0
    assert out["wx_gust"] == 0.0
    assert out["wx_snow"] == 0.0
    assert out["wx_precip"] == 0.0
    assert out["wx_wind_15"] == 0
    assert out["wx_gust_30"] == 0
    assert out["wx_snow_any"] == 0
    assert out["wx_cold_32"] == 0


def test_forecast_outdoor_flags():
    out = forecast_weather(temp=28, wind=16, gust=31, snow=0.5, precip_mm=1.2, roof="outdoors")
    assert out["wx_outdoor"] == 1
    assert out["wx_temp"] == 28.0
    assert out["wx_wind_15"] == 1
    assert out["wx_gust_30"] == 1
    assert out["wx_snow_any"] == 1
    assert out["wx_cold_32"] == 1


def test_attach_weather_dome_neutral():
    df = pl.DataFrame({"game_id": ["g1", "g2"]})
    import tempfile, os
    # build a weather parquet
    w = pl.DataFrame({
        "game_id": ["g1", "g2"],
        "is_dome": [1, 0],
        "actual_temp": [None, 40.0],
        "actual_wind": [None, 12.0],
        "actual_gust": [None, 20.0],
        "actual_precip": [None, 1.0],
        "actual_snow": [None, 0.0],
        "actual_humidity": [None, 50.0],
    })
    path = os.path.join(tempfile.gettempdir(), "wx_test.parquet")
    w.write_parquet(path)
    out = attach_weather(df, source="actual", weather_path=path)
    assert "wx_outdoor" in out.columns
    row0 = out.filter(pl.col("game_id") == "g1").to_dicts()[0]
    assert row0["wx_outdoor"] == 0
    assert row0["wx_temp"] == 60.0
    row1 = out.filter(pl.col("game_id") == "g2").to_dicts()[0]
    assert row1["wx_outdoor"] == 1
    assert row1["wx_temp"] == 40.0
    os.remove(path)


def test_weather_features_list():
    assert WEATHER_FEATURES == WF
    assert len(WEATHER_FEATURES) == 10
