#!/usr/bin/env python3
"""Research ablation: usage, red-zone and trend features for player models.

The player models roll four games of volume and production, plus snaps. This
script tests three additions that are not in the shipped feature set:

- rolling target share, air-yards share and WOPR (role, not just volume)
- red-zone targets and carries per game (scoring opportunity)
- short-term usage trend (last game minus the four-game average)

Walk-forward by season, grouped by position, same estimator as production.
Writes ``reports/player_ablation.json``. Nothing is promoted unless it wins.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import train_player_models as T

PBP = "data/pbp_latest.parquet"
OUT = "reports/player_ablation.json"
TEST_SEASONS = (2023, 2024, 2025, 2026)
WINDOW = T.WINDOW


def red_zone_counts() -> pl.DataFrame:
    pbp = pl.read_parquet(PBP)
    pbp = pbp.filter(
        pl.col("yardline_100").is_not_null()
        & (pl.col("yardline_100") <= 20)
        & pl.col("play_type").is_in(["pass", "run"])
    )
    targets = (
        pbp.filter(pl.col("receiver_player_id").is_not_null())
        .group_by(["game_id", pl.col("receiver_player_id").alias("player_id")])
        .agg(pl.len().alias("rz_targets"))
    )
    carries = (
        pbp.filter(pl.col("rusher_player_id").is_not_null())
        .group_by(["game_id", pl.col("rusher_player_id").alias("player_id")])
        .agg(pl.len().alias("rz_carries"))
    )
    return targets.join(carries, on=["game_id", "player_id"], how="full", coalesce=True)


def add_rolling(df: pl.DataFrame, id_col: str, cols: list[str]) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            pl.col(col)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .shift(1)
            .rolling_mean(window_size=WINDOW, min_samples=1)
            .over(id_col)
            .alias(f"roll_{col}")
        )
    return df


def add_trends(df: pl.DataFrame, id_col: str, cols: list[str]) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            (
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).shift(1).over(id_col)
                - pl.col(f"roll_{col}")
            ).alias(f"trend_{col}")
        )
    return df


def evaluate(df: pl.DataFrame, features: list[str], groups: dict) -> dict:
    out = {}
    for group, positions in groups.items():
        maes = []
        rows = 0
        for season in TEST_SEASONS:
            train = df.filter(
                (pl.col("season") < season)
                & pl.col("position").is_in(positions)
                & (pl.col("games_played") >= 1)
                & pl.col("roll_fantasy_points").is_not_null()
            )
            test = df.filter(
                (pl.col("season") == season)
                & pl.col("position").is_in(positions)
                & (pl.col("games_played") >= 1)
                & pl.col("roll_fantasy_points").is_not_null()
            )
            if train.height < 50 or test.height < 10:
                continue
            model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            model.fit(
                train.select(features).to_numpy(),
                train["fantasy_points"].to_numpy(),
            )
            pred = model.predict(test.select(features).to_numpy())
            y = test["fantasy_points"].to_numpy()
            maes.append(mean_absolute_error(y, pred) * len(y))
            rows += len(y)
        out[group] = round(sum(maes) / rows, 4) if rows else None
    if out:
        out["_overall"] = round(
            float(np.mean([v for k, v in out.items() if k != "_overall" and v is not None])),
            4,
        )
    return out


def main() -> None:
    df = T.load_player_features()
    df = T.enrich_context(df)
    df = T.add_position_group(df)
    df = T.enrich_defense_rating(df)

    rz = red_zone_counts()
    df = df.join(rz, on=["game_id", "player_id"], how="left").with_columns(
        pl.col("rz_targets").fill_null(0), pl.col("rz_carries").fill_null(0)
    )

    rolling, id_col = T.build_rolling(df)
    base_features = T.feature_cols(rolling)

    rolling = add_rolling(
        rolling, id_col, ["target_share", "air_yards_share", "wopr", "rz_targets", "rz_carries"]
    )
    rolling = add_trends(rolling, id_col, ["targets", "carries", "offense_pct", "fantasy_points"])

    share = ["roll_target_share", "roll_air_yards_share", "roll_wopr"]
    rz_cols = ["roll_rz_targets", "roll_rz_carries"]
    trend = ["trend_targets", "trend_carries", "trend_offense_pct", "trend_fantasy_points"]

    variants = {
        "base": base_features,
        "base+share": base_features + share,
        "base+rz": base_features + rz_cols,
        "base+trend": base_features + trend,
        "base+share+rz": base_features + share + rz_cols,
        "base+all": base_features + share + rz_cols + trend,
    }

    results = {}
    for name, feats in variants.items():
        present = [c for c in feats if c in rolling.columns]
        results[name] = evaluate(rolling, present, T.POSITION_GROUPS)
        print(name, results[name])

    base = results["base"]["_overall"]
    best = min(results, key=lambda k: results[k]["_overall"])
    print(f"baseline={base:.4f} best={best} mae={results[best]['_overall']:.4f}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
