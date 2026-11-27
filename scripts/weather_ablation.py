#!/usr/bin/env python3
"""Test whether historical weather improves any model.

Runs the same walk-forward split for the match winner, margin, total and each
player position, with and without Open-Meteo weather. A dome-only variant is
included so venue effects are not mistaken for weather effects.

Writes ``reports/weather_ablation.json``. Weather is only worth modelling if it
beats the base model beyond the dome flag; otherwise it stays display-only.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_player_models import (  # noqa: E402
    add_position_group,
    build_rolling,
    enrich_context,
    enrich_defense_rating,
    feature_cols,
)

MATCH_FEATURES = "features/match_features.parquet"
PLAYER_FEATURES = "features/player_features.parquet"
WEATHER = "features/weather_games.parquet"
OUTPUT = "reports/weather_ablation.json"
TEST_SEASONS = (2024, 2025, 2026)

WEATHER_ONLY = ["temp", "wind", "precip", "wind_15", "cold_32", "precip_any"]
WEATHER_V2 = WEATHER_ONLY + [
    "gust",
    "snow",
    "humidity",
    "gust_30",
    "snow_any",
]
DOME = ["is_dome"]
VARIANTS = {
    "base": [],
    "dome": DOME,
    "weather": WEATHER_ONLY,
    "dome_weather": DOME + WEATHER_ONLY,
    "weather_v2": WEATHER_V2,
    "dome_weather_v2": DOME + WEATHER_V2,
}


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
    )


def load_weather(source: str) -> pl.DataFrame:
    frame = pl.read_parquet(WEATHER)
    mapping = {
        name: f"{source}_{name}"
        for name in ("temp", "wind", "gust", "precip", "snow", "humidity")
    }
    available = {name: col for name, col in mapping.items() if col in frame.columns}
    out = frame.select(["game_id", "is_dome", *available.values()])
    return out.rename({col: name for name, col in available.items()})


def attach_weather(df: pl.DataFrame, source: str) -> pl.DataFrame:
    weather = load_weather(source)
    overlap = [
        col
        for col in ("is_dome", "temp", "wind", "gust", "precip", "snow", "humidity")
        if col in df.columns
    ]
    if overlap:
        df = df.drop(overlap)
    out = df.join(weather, on="game_id", how="left")
    dome = pl.col("is_dome").fill_null(0) == 1
    # Dome/closed games are neutral. Missing outdoor readings fall back to mild
    # values so every row has numbers, but the flags stay 0.
    out = out.with_columns(
        [
            dome.cast(pl.Int8).alias("is_dome"),
            (dome | pl.col("temp").is_not_null()).alias("weather_present"),
            pl.when(dome).then(60.0).otherwise(pl.col("temp")).fill_null(60.0).alias("temp"),
            pl.when(dome).then(0.0).otherwise(pl.col("wind")).fill_null(0.0).alias("wind"),
            pl.when(dome).then(0.0).otherwise(pl.col("gust")).fill_null(0.0).alias("gust"),
            pl.when(dome).then(0.0).otherwise(pl.col("precip")).fill_null(0.0).alias("precip"),
            pl.when(dome).then(0.0).otherwise(pl.col("snow")).fill_null(0.0).alias("snow"),
            pl.when(dome)
            .then(50.0)
            .otherwise(pl.col("humidity"))
            .fill_null(50.0)
            .alias("humidity"),
        ]
    ).with_columns(
        [
            (pl.col("wind") >= 15).cast(pl.Int8).alias("wind_15"),
            (pl.col("temp") <= 32).cast(pl.Int8).alias("cold_32"),
            (pl.col("precip") > 0).cast(pl.Int8).alias("precip_any"),
            (pl.col("gust") >= 30).cast(pl.Int8).alias("gust_30"),
            (pl.col("snow") > 0).cast(pl.Int8).alias("snow_any"),
        ]
    )
    return out


def matrix(data: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.nan_to_num(
        data.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_numpy(),
        nan=0.0,
    )


def regression_variants(
    data: pl.DataFrame,
    target: str,
    base: list[str],
    bucket_keys: tuple[str, ...] = (),
) -> dict:
    seasons = data["season"].to_numpy()
    y = data[target].to_numpy()
    matrices = {
        name: matrix(data, base + extra) for name, extra in VARIANTS.items()
    }

    errors = {name: [] for name in VARIANTS}
    per_season = {name: {} for name in VARIANTS}
    buckets = {key: {"true": [], "false": []} for key in bucket_keys}

    for test_season in TEST_SEASONS:
        train = seasons < test_season
        test = seasons == test_season
        if train.sum() < 100 or test.sum() < 20:
            continue
        for name, X in matrices.items():
            fitted = model().fit(X[train], y[train])
            pred = fitted.predict(X[test])
            errors[name].extend((pred - y[test]).tolist())
            per_season[name][str(test_season)] = round(
                float(mean_absolute_error(y[test], pred)), 3
            )
            if name == "base":
                for key in bucket_keys:
                    values = data[key].to_numpy()[test]
                    residuals = pred - y[test]
                    buckets[key]["true"].extend(residuals[values == 1].tolist())
                    buckets[key]["false"].extend(residuals[values == 0].tolist())

    base_mae = round(float(np.mean(np.abs(errors["base"]))), 3) if errors["base"] else None
    result = {"variants": {}, "per_season": per_season}
    for name, values in errors.items():
        mae = round(float(np.mean(np.abs(values))), 3) if values else None
        result["variants"][name] = {
            "mae": mae,
            "delta_vs_base": round(mae - base_mae, 3)
            if mae is not None and base_mae is not None
            else None,
        }
    result["rows"] = data.height
    result["weather_coverage"] = round(float(data["weather_present"].mean()), 3)
    if bucket_keys:
        result["buckets"] = {
            key: {
                "games_true": len(values["true"]),
                "games_false": len(values["false"]),
                "residual_true": round(float(np.mean(values["true"])), 2)
                if values["true"]
                else None,
                "residual_false": round(float(np.mean(values["false"])), 2)
                if values["false"]
                else None,
            }
            for key, values in buckets.items()
        }
    return result


def classification_variants(data: pl.DataFrame, target: str, base: list[str]) -> dict:
    seasons = data["season"].to_numpy()
    y = data[target].to_numpy()
    matrices = {
        name: matrix(data, base + extra) for name, extra in VARIANTS.items()
    }
    predictions = {name: {"pred": [], "true": []} for name in VARIANTS}
    per_season = {name: {} for name in VARIANTS}
    for test_season in TEST_SEASONS:
        train = seasons < test_season
        test = seasons == test_season
        if train.sum() < 100 or test.sum() < 20:
            continue
        for name, X in matrices.items():
            fitted = LogisticRegression(max_iter=3000).fit(X[train], y[train])
            prob = fitted.predict_proba(X[test])[:, 1]
            predictions[name]["pred"].extend(prob.tolist())
            predictions[name]["true"].extend(y[test].tolist())
            per_season[name][str(test_season)] = round(
                float(log_loss(y[test], prob)), 4
            )

    base_loss = (
        float(log_loss(predictions["base"]["true"], predictions["base"]["pred"]))
        if predictions["base"]["pred"]
        else None
    )
    result = {"variants": {}, "per_season": per_season}
    for name, values in predictions.items():
        if not values["pred"]:
            continue
        prob = np.asarray(values["pred"])
        truth = np.asarray(values["true"])
        loss = float(log_loss(truth, prob))
        result["variants"][name] = {
            "log_loss": round(loss, 4),
            "accuracy": round(float(accuracy_score(truth, prob >= 0.5)), 4),
            "auc": round(float(roc_auc_score(truth, prob)), 4),
            "delta_log_loss": round(loss - base_loss, 4) if base_loss is not None else None,
        }
    result["rows"] = data.height
    result["weather_coverage"] = round(float(data["weather_present"].mean()), 3)
    return result


def totals_and_margin() -> dict:
    import joblib

    played = pl.read_parquet(MATCH_FEATURES).filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("season") >= 2022)
    )
    played = played.with_columns(
        (pl.col("home_score") + pl.col("away_score")).alias("total_points"),
        (pl.col("home_score") - pl.col("away_score")).alias("margin"),
    )
    base = joblib.load("models/score_models.joblib")["features"]
    output = {"total": {}, "margin": {}}
    for source in ("actual", "forecast"):
        data = attach_weather(played, source)
        print(f"Totals + margin · {source} weather")
        output["total"][source] = regression_variants(
            data, "total_points", base, bucket_keys=("wind_15", "cold_32", "precip_any")
        )
        output["margin"][source] = regression_variants(data, "margin", base)
    return output


def match_model() -> dict:
    import joblib

    played = pl.read_parquet(MATCH_FEATURES).filter(
        pl.col("home_win").is_not_null() & (pl.col("season") >= 2022)
    )
    base = joblib.load("models/match_model.joblib")["features"]
    output = {}
    for source in ("actual", "forecast"):
        data = attach_weather(played, source)
        print(f"Match winner · {source} weather")
        output[source] = classification_variants(data, "home_win", base)
    return output


def players() -> dict:
    df = pl.read_parquet(PLAYER_FEATURES)
    df = enrich_defense_rating(add_position_group(enrich_context(df)))
    rolling, _ = build_rolling(df)
    output = {}
    for position in ("QB", "RB", "WR", "TE", "K"):
        data = rolling.filter(
            (pl.col("position") == position)
            & (pl.col("games_played") >= 1)
            & pl.col("roll_fantasy_points").is_not_null()
            & (pl.col("season") >= 2022)
        )
        if data.height < 80:
            continue
        base = feature_cols(data)
        output[position] = {}
        for source in ("actual", "forecast"):
            data_source = attach_weather(data, source)
            print(f"Players {position} · {source} weather")
            output[position][source] = regression_variants(
                data_source, "fantasy_points", base
            )
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--totals-only",
        action="store_true",
        help="Run just the total/margin ablation into reports/weather_total_ablation.json",
    )
    args = parser.parse_args()
    if not os.path.exists(WEATHER):
        print("No historical weather yet; run scripts/fetch_weather_history.py.")
        return
    if args.totals_only:
        report = {
            "method": "walk-forward 2024-2026; base vs dome-only vs weather-only vs "
            "dome+weather vs gust/snow v2; Open-Meteo ERA5 (actual) and "
            "historical-forecast archive",
            "weather_only_features": WEATHER_ONLY,
            "weather_v2_features": WEATHER_V2,
            "dome_feature": DOME,
            "totals_margin": totals_and_margin(),
        }
        output = "reports/weather_total_ablation.json"
    else:
        report = {
            "method": "walk-forward 2024-2026; base vs dome-only vs weather-only vs "
            "dome+weather vs gust/snow v2; Open-Meteo ERA5 (actual) and "
            "historical-forecast archive",
            "weather_only_features": WEATHER_ONLY,
            "weather_v2_features": WEATHER_V2,
            "dome_feature": DOME,
            "totals_margin": totals_and_margin(),
            "match": match_model(),
            "players": players(),
        }
        output = OUTPUT
    os.makedirs("reports", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
