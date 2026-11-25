#!/usr/bin/env python3
"""Add weather and weekly-roster context to player features.

- weather: roof/dome, temperature, wind (from schedules)
- roster: weekly status (ACT, RES, INA, DEV, ...) by player-week

Run after ``build_features.py`` and before ``train_player_models.py``.
"""

from __future__ import annotations

import os

import polars as pl

PLAYERS = "features/player_features.parquet"
SCHEDULES = "data/schedules.parquet"
ROSTERS = ["data/rosters_weekly_history.parquet", "data/rosters_weekly_latest.parquet"]


def weather_table() -> pl.DataFrame | None:
    if not os.path.exists(SCHEDULES):
        return None
    schedules = pl.read_parquet(SCHEDULES)
    cols = ["season", "week", "home_team", "away_team", "roof", "surface", "temp", "wind"]
    if not set(cols).issubset(set(schedules.columns)):
        return None
    base = schedules.select(cols)
    home = base.select(
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.lit(1).cast(pl.Int8).alias("is_home"),
        "roof",
        "surface",
        "temp",
        "wind",
    )
    away = base.select(
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.lit(0).cast(pl.Int8).alias("is_home"),
        "roof",
        "surface",
        "temp",
        "wind",
    )
    table = pl.concat([home, away]).with_columns(
        pl.col("roof").is_in(["dome", "closed"]).alias("is_dome"),
        pl.col("roof").is_in(["outdoors", "open"]).alias("is_outdoor"),
    )
    return table


def roster_table() -> pl.DataFrame | None:
    path = next((p for p in ROSTERS if os.path.exists(p)), None)
    if path is None:
        return None
    rosters = pl.read_parquet(path)
    required = {"season", "week", "team", "gsis_id", "status"}
    if not required.issubset(set(rosters.columns)):
        return None
    return (
        rosters.select(
            "season",
            "week",
            "team",
            pl.col("gsis_id").alias("player_id"),
            pl.col("status").alias("roster_status"),
        )
        .unique(subset=["season", "week", "team", "player_id"], keep="last")
    )


def main() -> None:
    if not os.path.exists(PLAYERS):
        print("No player features; skipping context.")
        return
    players = pl.read_parquet(PLAYERS)

    weather = weather_table()
    if weather is not None:
        players = players.join(
            weather.drop("roof", "surface"),
            on=["season", "week", "team"],
            how="left",
        )
        players = players.with_columns(
            pl.col("is_dome").fill_null(0).cast(pl.Int8),
            pl.col("is_outdoor").fill_null(0).cast(pl.Int8),
            pl.col("temp").fill_null(60).cast(pl.Float64),
            pl.col("wind").fill_null(8).cast(pl.Float64),
        )
    else:
        players = players.with_columns(
            pl.lit(0).cast(pl.Int8).alias("is_dome"),
            pl.lit(0).cast(pl.Int8).alias("is_outdoor"),
            pl.lit(60.0).alias("temp"),
            pl.lit(8.0).alias("wind"),
        )

    rosters = roster_table()
    if rosters is not None:
        players = players.join(
            rosters,
            on=["season", "week", "team", "player_id"],
            how="left",
        )
    else:
        players = players.with_columns(pl.lit(None).cast(pl.Utf8).alias("roster_status"))

    players.write_parquet(PLAYERS)
    print(
        f"Added weather/roster context to {players.height} player rows "
        f"(statuses: {players['roster_status'].n_unique() if 'roster_status' in players.columns else 0})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Player context error: {exc}")
