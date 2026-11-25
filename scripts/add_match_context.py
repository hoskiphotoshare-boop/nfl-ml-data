#!/usr/bin/env python3
"""Add team-record and quarterback-context features to match features.

Adds pre-game, leakage-safe columns:

- prior wins / games / win percentage / point differential per game
  (opponent-adjusted only in the sense that they are actual results)
- the team's primary quarterback for the week, whether the starter changed,
  and that quarterback's trailing EPA per attempt and starts

These give the match model the two signals it was missing: results and
quarterback continuity. Run after ``build_features.py`` and before
``train_models.py``.
"""

from __future__ import annotations

import os
import sys

import polars as pl

MATCH_FEATURES = "features/match_features.parquet"
PLAYER_FEATURES = "features/player_features.parquet"

ADDED_COLUMNS = [
    "home_prior_games",
    "away_prior_games",
    "home_win_pct",
    "away_win_pct",
    "home_diff_pg",
    "away_diff_pg",
    "win_pct_diff",
    "diff_pg_diff",
    "home_qb_id",
    "away_qb_id",
    "home_qb_prior_epa",
    "away_qb_prior_epa",
    "home_qb_starts",
    "away_qb_starts",
    "home_qb_changed",
    "away_qb_changed",
    "home_qb_missing",
    "away_qb_missing",
    "home_inj_impact",
    "away_inj_impact",
    "inj_impact_diff",
    "home_inj_missing",
    "away_inj_missing",
]

INJURY_PATHS = ["data/injuries_history.parquet", "data/injuries_latest.parquet"]
STATUS_WEIGHTS = {"out": 4.0, "doubtful": 2.5, "questionable": 1.2, "probable": 0.5}
POSITION_WEIGHTS = {
    "QB": 4.0,
    "T": 2.5,
    "G": 2.5,
    "C": 2.5,
    "WR": 1.5,
    "TE": 1.5,
    "RB": 1.5,
    "FB": 0.8,
    "CB": 1.2,
    "S": 1.2,
    "LB": 1.2,
    "DE": 1.2,
    "DT": 1.2,
    "NT": 1.2,
    "K": 0.5,
    "P": 0.5,
    "LS": 0.2,
}


def add_record_features(df: pl.DataFrame) -> pl.DataFrame:
    played = df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    ).select("season", "week", "home_team", "away_team", "home_score", "away_score")

    home = played.select(
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
    )
    away = played.select(
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
    )
    long = (
        pl.concat([home, away])
        .with_columns(
            (pl.col("pf") - pl.col("pa")).alias("margin"),
            (pl.col("pf") > pl.col("pa")).cast(pl.Int8).alias("win"),
        )
        .sort(["season", "team", "week"])
    )

    long = long.with_columns(
        (pl.col("win").cum_sum().over(["season", "team"]) - pl.col("win")).alias(
            "prior_wins"
        ),
        (pl.col("margin").cum_sum().over(["season", "team"]) - pl.col("margin")).alias(
            "prior_margin"
        ),
        (pl.int_range(1, pl.len() + 1).over(["season", "team"]) - 1).alias(
            "prior_games"
        ),
    ).with_columns(
        (pl.col("prior_wins") / pl.col("prior_games").clip(1)).alias("win_pct"),
        (pl.col("prior_margin") / pl.col("prior_games").clip(1)).alias("diff_pg"),
    )
    long = long.with_columns(
        pl.when(pl.col("prior_games") == 0)
        .then(0.5)
        .otherwise(pl.col("win_pct"))
        .alias("win_pct")
    )

    team_week = long.select(
        "season", "week", "team", "prior_games", "win_pct", "diff_pg"
    ).sort(["season", "week"])

    out = df.sort(["season", "week"])
    for side in ("home", "away"):
        table = team_week.rename(
            {
                "team": f"{side}_team",
                "prior_games": f"{side}_prior_games",
                "win_pct": f"{side}_win_pct",
                "diff_pg": f"{side}_diff_pg",
            }
        ).sort(["season", "week"])
        out = out.join_asof(
            table,
            on="week",
            by=["season", f"{side}_team"],
            strategy="backward",
            allow_exact_matches=False,
        )

    out = out.with_columns(
        [
            pl.col(f"{side}_{name}").fill_null(0).alias(f"{side}_{name}")
            for side in ("home", "away")
            for name in ("prior_games", "win_pct", "diff_pg")
        ]
    )
    out = out.with_columns(
        (pl.col("home_win_pct") - pl.col("away_win_pct")).alias("win_pct_diff"),
        (pl.col("home_diff_pg") - pl.col("away_diff_pg")).alias("diff_pg_diff"),
    )
    return out


def qb_week_table() -> pl.DataFrame:
    players = pl.read_parquet(PLAYER_FEATURES)
    qbs = players.filter(
        (pl.col("position") == "QB")
        & (pl.col("attempts").is_not_null())
        & (pl.col("attempts") > 0)
    )
    primary = (
        qbs.sort(
            ["season", "week", "team", "attempts"],
            descending=[False, False, False, True],
        )
        .group_by(["season", "week", "team"], maintain_order=True)
        .agg(
            pl.col("player_id").first().alias("qb_id"),
            pl.col("player_name").first().alias("qb_name"),
            pl.col("attempts").first().alias("qb_attempts"),
            pl.col("passing_epa").first().alias("qb_epa"),
        )
    )
    primary = primary.with_columns(
        (pl.col("qb_epa") / pl.col("qb_attempts").clip(1)).alias("qb_epa_per_att")
    ).sort(["season", "team", "week"])

    primary = primary.with_columns(
        pl.col("qb_id").shift(1).over(["season", "team"]).alias("qb_prev_id"),
    ).with_columns(
        pl.when(pl.col("qb_prev_id").is_null())
        .then(0)
        .otherwise((pl.col("qb_id") != pl.col("qb_prev_id")).cast(pl.Int8))
        .alias("qb_changed")
    )

    # Count prior starts for the same quarterback in the season.
    starts = (
        primary.sort(["season", "team", "qb_id", "week"])
        .with_columns(
            (pl.int_range(1, pl.len() + 1).over(["season", "team", "qb_id"]) - 1).alias(
                "qb_starts"
            ),
            pl.col("qb_epa_per_att")
            .shift(1)
            .rolling_mean(3)
            .over(["season", "team", "qb_id"])
            .alias("qb_prior_epa"),
        )
        .select("season", "week", "team", "qb_id", "qb_starts", "qb_prior_epa")
    )
    primary = primary.join(starts, on=["season", "week", "team", "qb_id"], how="left")
    return primary


def add_qb_features(df: pl.DataFrame) -> pl.DataFrame:
    qb = qb_week_table()
    out = df
    for side in ("home", "away"):
        table = qb.select(
            "season",
            "week",
            pl.col("team").alias(f"{side}_team"),
            pl.col("qb_id").alias(f"{side}_qb_id"),
            pl.col("qb_prior_epa").alias(f"{side}_qb_prior_epa"),
            pl.col("qb_starts").alias(f"{side}_qb_starts"),
            pl.col("qb_changed").alias(f"{side}_qb_changed"),
        )
        out = out.join(table, on=["season", "week", f"{side}_team"], how="left")

    expressions = []
    for side in ("home", "away"):
        expressions.extend(
            [
                pl.col(f"{side}_qb_prior_epa").fill_null(0.0).alias(f"{side}_qb_prior_epa"),
                pl.col(f"{side}_qb_starts").fill_null(0).alias(f"{side}_qb_starts"),
                pl.col(f"{side}_qb_changed").fill_null(0).alias(f"{side}_qb_changed"),
                pl.col(f"{side}_qb_id").is_null().cast(pl.Int8).alias(f"{side}_qb_missing"),
            ]
        )
    out = out.with_columns(expressions)
    return out


def injury_table() -> pl.DataFrame | None:
    for path in INJURY_PATHS:
        if os.path.exists(path):
            break
    else:
        return None
    injuries = pl.read_parquet(path)
    if "report_status" not in injuries.columns:
        return None
    required = {"season", "week", "team", "position", "report_status"}
    if not required.issubset(set(injuries.columns)):
        return None

    injuries = injuries.filter(pl.col("report_status").is_not_null())
    status = pl.col("report_status").str.to_lowercase()
    impact = (
        pl.when(status == "out").then(STATUS_WEIGHTS["out"])
        .when(status == "doubtful").then(STATUS_WEIGHTS["doubtful"])
        .when(status == "questionable").then(STATUS_WEIGHTS["questionable"])
        .when(status == "probable").then(STATUS_WEIGHTS["probable"])
        .otherwise(0.0)
    )
    position = pl.col("position").fill_null("")
    pos_weight = (
        pl.when(position == "QB").then(POSITION_WEIGHTS["QB"])
        .when(position.is_in(["T", "G", "C"])).then(POSITION_WEIGHTS["T"])
        .when(position.is_in(["WR", "TE", "RB"])).then(POSITION_WEIGHTS["WR"])
        .when(position.is_in(["CB", "S", "LB", "DE", "DT", "NT"])).then(POSITION_WEIGHTS["CB"])
        .when(position.is_in(["K", "P"])).then(POSITION_WEIGHTS["K"])
        .otherwise(0.3)
    )
    injuries = injuries.with_columns((impact * pos_weight).alias("impact"))
    return (
        injuries.group_by(["season", "week", "team"])
        .agg(
            pl.col("impact").sum().alias("inj_impact"),
            pl.len().alias("inj_players"),
        )
        .sort(["season", "week"])
    )


def add_injury_features(df: pl.DataFrame) -> pl.DataFrame:
    table = injury_table()
    if table is None:
        return df
    out = df
    for side in ("home", "away"):
        side_table = table.rename(
            {
                "team": f"{side}_team",
                "inj_impact": f"{side}_inj_impact",
            }
        ).select(["season", "week", f"{side}_team", f"{side}_inj_impact"])
        out = out.join(side_table, on=["season", "week", f"{side}_team"], how="left")
    out = out.with_columns(
        pl.col("home_inj_impact").is_null().cast(pl.Int8).alias("home_inj_missing"),
        pl.col("away_inj_impact").is_null().cast(pl.Int8).alias("away_inj_missing"),
    ).with_columns(
        pl.col("home_inj_impact").fill_null(0.0),
        pl.col("away_inj_impact").fill_null(0.0),
    )
    out = out.with_columns(
        (pl.col("home_inj_impact") - pl.col("away_inj_impact")).alias("inj_impact_diff")
    )
    return out


def main() -> None:
    if not os.path.exists(MATCH_FEATURES):
        print("No match features found; skipping context features.")
        return

    df = pl.read_parquet(MATCH_FEATURES)
    df = df.drop([c for c in ADDED_COLUMNS if c in df.columns])
    before = df.height

    df = add_record_features(df)
    df = add_injury_features(df)
    if os.path.exists(PLAYER_FEATURES):
        df = add_qb_features(df)
    else:
        print("No player features; adding neutral quarterback columns.")
        for side in ("home", "away"):
            df = df.with_columns(
                [
                    pl.lit(None).cast(pl.Utf8).alias(f"{side}_qb_id"),
                    pl.lit(0.0).alias(f"{side}_qb_prior_epa"),
                    pl.lit(0).alias(f"{side}_qb_starts"),
                    pl.lit(0).alias(f"{side}_qb_changed"),
                    pl.lit(1).cast(pl.Int8).alias(f"{side}_qb_missing"),
                ]
            )

    df.write_parquet(MATCH_FEATURES)
    print(f"Added context features to {before} match rows")
    sample = df.select(
        "season", "week", "home_team", "away_team",
        "home_win_pct", "away_win_pct", "home_diff_pg", "away_diff_pg",
        "home_qb_prior_epa", "away_qb_prior_epa", "home_qb_changed", "away_qb_changed",
    ).tail(3)
    print(sample)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Context feature error: {exc}")
    sys.exit(0)
