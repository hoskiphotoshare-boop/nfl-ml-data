#!/usr/bin/env python3
"""Train per-position player fantasy models and write player projections.

Builds pre-game rolling player features from ``features/player_features.parquet``,
trains a HistGradientBoosting regressor per position group, evaluates on the
latest season, and writes ``predictions/players_week.json``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import joblib
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

WINDOW = 4
POSITION_GROUPS = {
    "QB": ["QB"],
    "RB": ["RB", "FB", "HB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "K": ["K"],
}
ROLL_COLS = [
    "fantasy_points",
    "targets",
    "receptions",
    "yards",
    "touchdowns",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "attempts",
    "completions",
]


def load_player_features() -> pl.DataFrame:
    path = "features/player_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def build_rolling(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    id_col = "player_id" if "player_id" in df.columns else ("gsis_id" if "gsis_id" in df.columns else "player_name")
    df = df.sort([id_col, "season", "week"])

    exprs = []
    for col in ROLL_COLS:
        if col in df.columns:
            exprs.append(
                pl.col(col)
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .shift(1)
                .rolling_mean(window_size=WINDOW, min_samples=1)
                .over(id_col)
                .alias(f"roll_{col}")
            )
    exprs.append(
        pl.int_range(1, pl.len() + 1).over(id_col).alias("game_num")
    )
    out = df.with_columns(exprs)
    out = out.with_columns((pl.col("game_num") - 1).alias("games_played"))
    return out, id_col


def feature_cols(df: pl.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("roll_")]
    return cols


def train_group(df: pl.DataFrame, group: str, positions: list[str]):
    data = df.filter(
        pl.col("position").is_in(positions)
        & (pl.col("games_played") >= 1)
        & pl.col("roll_fantasy_points").is_not_null()
    )
    if data.height < 50:
        print(f"{group}: only {data.height} rows; using rolling average fallback.")
        return None, {}

    seasons = sorted(int(s) for s in data["season"].unique().to_list())
    if len(seasons) < 2:
        print(f"{group}: need at least two seasons; using fallback.")
        return None, {}

    test_season = seasons[-1]
    train = data.filter(pl.col("season") < test_season)
    test = data.filter(pl.col("season") == test_season)
    if train.height < 50 or test.height < 10:
        print(f"{group}: split too small; using fallback.")
        return None, {}

    cols = feature_cols(data)
    X_train = train.select(cols).to_numpy()
    y_train = train["fantasy_points"].to_numpy()
    X_test = test.select(cols).to_numpy()
    y_test = test["fantasy_points"].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "rows": int(len(y_test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
    }
    print(f"{group}: n={metrics['rows']} MAE={metrics['mae']} RMSE={metrics['rmse']}")
    return model, metrics


def predict_latest(df: pl.DataFrame, models: dict, id_col: str) -> list[dict]:
    # Latest row per player.
    latest = (
        df.sort(["season", "week"])
        .group_by(id_col)
        .agg(pl.all().last())
    )

    records = []
    for row in latest.iter_rows(named=True):
        pos = row.get("position") or ""
        group = next((g for g, ps in POSITION_GROUPS.items() if pos in ps), None)
        if group is None:
            continue

        model = models.get(group)
        stats = row
        if model is not None:
            cols = feature_cols(df)
            x = [[float(stats.get(c) or 0.0) for c in cols]]
            fantasy = float(model.predict(x)[0])
        else:
            fantasy = float(stats.get("roll_fantasy_points") or 0.0)

        records.append(
            {
                "player_id": stats.get(id_col),
                "name": stats.get("player_name") or stats.get("player_display_name") or stats.get(id_col),
                "team": stats.get("team") or "",
                "position": pos,
                "projected_targets": round(float(stats.get("roll_targets") or 0.0), 1),
                "projected_receptions": round(float(stats.get("roll_receptions") or 0.0), 1),
                "projected_yards": round(float(stats.get("roll_yards") or 0.0), 1),
                "projected_tds": round(float(stats.get("roll_touchdowns") or 0.0), 2),
                "projected_fantasy_points": round(fantasy, 1),
                "projected_pass_yards": round(float(stats.get("roll_passing_yards") or 0.0), 1)
                if pos == "QB"
                else None,
                "projected_pass_tds": round(float(stats.get("roll_passing_tds") or 0.0), 2)
                if pos == "QB"
                else None,
            }
        )

    records.sort(key=lambda r: r["projected_fantasy_points"], reverse=True)
    return records


def main() -> None:
    df = load_player_features()
    if df.height == 0:
        print("No player features found; skipping player models.")
        return

    rolling, id_col = build_rolling(df)
    models = {}
    metrics = {}

    for group, positions in POSITION_GROUPS.items():
        model, group_metrics = train_group(rolling, group, positions)
        models[group] = model
        metrics[group] = group_metrics

    os.makedirs("models", exist_ok=True)
    joblib.dump(models, "models/player_models.joblib")

    os.makedirs("reports", exist_ok=True)
    with open("reports/player_backtest.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    records = predict_latest(rolling, models, id_col)
    records = [r for r in records if r["projected_fantasy_points"] > 0][:250]

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/players_week.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote models/player_models.joblib and {len(records)} player projections.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Player model error: {exc}")
    sys.exit(0)