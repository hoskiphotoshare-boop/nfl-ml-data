#!/usr/bin/env python3
"""Normalized weather features for the total model."""
from __future__ import annotations
import os
import polars as pl

WEATHER_FEATURES = ["wx_outdoor","wx_temp","wx_wind","wx_gust","wx_snow","wx_precip","wx_wind_15","wx_gust_30","wx_snow_any","wx_cold_32"]
NEUTRAL_TEMP = 60.0

def _is_dome_value(value):
    if value is None: return False
    if isinstance(value,(int,float)): return bool(value)
    return str(value).strip().lower() in {"1","true","dome","closed","indoor"}

def attach_weather(df, source="actual", weather_path="features/weather_games.parquet"):
    if not os.path.exists(weather_path):
        return df
    try:
        weather = pl.read_parquet(weather_path)
    except Exception:
        return df
    prefix = f"{source}_"
    select_cols = ["game_id"]
    rename_map = {}
    if "is_dome" in weather.columns:
        select_cols.append("is_dome")
    for name in ("temp","wind","gust","precip","snow","humidity"):
        col = prefix + name
        if col in weather.columns:
            select_cols.append(col)
            rename_map[col] = name
    if len(select_cols) == 1:
        return df
    w = weather.select(select_cols)
    if rename_map:
        w = w.rename(rename_map)
    overlap = [c for c in w.columns if c in df.columns and c != "game_id"]
    base = df.drop(overlap) if overlap else df
    if "game_id" not in base.columns or "game_id" not in w.columns:
        return df
    out = base.join(w, on="game_id", how="left")
    if "is_dome" not in out.columns:
        out = out.with_columns(pl.lit(0).cast(pl.Int8).alias("is_dome"))
    dome = pl.col("is_dome").fill_null(0).cast(pl.Int64, strict=False).eq(1)
    for col in ("temp","wind","gust","precip","snow","humidity"):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    out = out.with_columns([
        (~dome).cast(pl.Int8).alias("wx_outdoor"),
        pl.when(dome).then(NEUTRAL_TEMP).otherwise(pl.col("temp")).fill_null(NEUTRAL_TEMP).alias("wx_temp"),
        pl.when(dome).then(0.0).otherwise(pl.col("wind")).fill_null(0.0).alias("wx_wind"),
        pl.when(dome).then(0.0).otherwise(pl.col("gust")).fill_null(0.0).alias("wx_gust"),
        pl.when(dome).then(0.0).otherwise(pl.col("snow")).fill_null(0.0).alias("wx_snow"),
        pl.when(dome).then(0.0).otherwise(pl.col("precip")).fill_null(0.0).alias("wx_precip"),
    ]).with_columns([
        (pl.col("wx_wind") >= 15).cast(pl.Int8).alias("wx_wind_15"),
        (pl.col("wx_gust") >= 30).cast(pl.Int8).alias("wx_gust_30"),
        (pl.col("wx_snow") > 0).cast(pl.Int8).alias("wx_snow_any"),
        (pl.col("wx_temp") <= 32).cast(pl.Int8).alias("wx_cold_32"),
    ])
    drop_cols = [c for c in ("temp","wind","gust","precip","snow","humidity") if c in out.columns]
    if drop_cols:
        out = out.drop(drop_cols)
    return out

def attach_weather_to_frame(df, *a, **k): return attach_weather(df, *a, **k)
def add_weather_features(df, *a, **k): return attach_weather(df, *a, **k)
def normalize_weather(df, *a, **k): return attach_weather(df, *a, **k)

def _coerce_forecast(temp=None, wind=None, gust=None, snow=None, precip_mm=None, roof=None, game=None, **kwargs):
    if isinstance(temp, dict):
        g = temp
        temp = g.get("temp", g.get("temperature"))
        wind = g.get("wind")
        gust = g.get("gust")
        snow = g.get("snow")
        precip_mm = g.get("precip_mm", g.get("precip", g.get("precipitation")))
        roof = g.get("roof", roof)
    if isinstance(game, dict):
        temp = game.get("temp", temp)
        wind = game.get("wind", wind)
        gust = game.get("gust", gust)
        snow = game.get("snow", snow)
        precip_mm = game.get("precip_mm", game.get("precip", precip_mm))
        roof = game.get("roof", roof)
    for key in ("temp","wind","gust","snow","precip_mm","roof"):
        if key in kwargs and kwargs[key] is not None:
            if key == "temp": temp = kwargs[key]
            elif key == "wind": wind = kwargs[key]
            elif key == "gust": gust = kwargs[key]
            elif key == "snow": snow = kwargs[key]
            elif key == "precip_mm": precip_mm = kwargs[key]
            elif key == "roof": roof = kwargs[key]
    return temp, wind, gust, snow, precip_mm, roof

def forecast_weather(temp=None, wind=None, gust=None, snow=None, precip_mm=None, roof=None, game=None, **kwargs):
    temp, wind, gust, snow, precip_mm, roof = _coerce_forecast(temp, wind, gust, snow, precip_mm, roof, game=game, **kwargs)
    dome = _is_dome_value(roof)
    def f(x, d):
        try: return float(x) if x is not None else d
        except (TypeError,ValueError): return d
    t = f(temp, NEUTRAL_TEMP); w = f(wind,0.0); g = f(gust,0.0); s = f(snow,0.0); p = f(precip_mm,0.0)
    if dome:
        t = NEUTRAL_TEMP; w=0.0; g=0.0; s=0.0; p=0.0
    return {
        "wx_outdoor": 0 if dome else 1,
        "wx_temp": t,
        "wx_wind": w,
        "wx_gust": g,
        "wx_snow": s,
        "wx_precip": p,
        "wx_wind_15": int(w >= 15),
        "wx_gust_30": int(g >= 30),
        "wx_snow_any": int(s > 0),
        "wx_cold_32": int(t <= 32),
    }

weather_from_forecast = forecast_weather
forecast_features = forecast_weather
forecast_to_features = forecast_weather
make_weather_features = forecast_weather
normalize_forecast = forecast_weather
weather_for_game = forecast_weather
build_weather_features = forecast_weather
weather_dict = forecast_weather

def present_weather_features(columns):
    colset = set(columns or [])
    return [c for c in WEATHER_FEATURES if c in colset]
