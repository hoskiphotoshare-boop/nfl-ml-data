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
    "home_prev_diff_pg",
    "away_prev_diff_pg",
    "home_prev_win_pct",
    "away_prev_win_pct",
    "home_blend_diff",
    "away_blend_diff",
    "blend_diff_diff",
    "home_blend_win",
    "away_blend_win",
    "blend_win_diff",
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
    "home_inj_qb",
    "away_inj_qb",
    "home_inj_ol",
    "away_inj_ol",
    "home_inj_skill",
    "away_inj_skill",
    "home_inj_def",
    "away_inj_def",
    "inj_qb_diff",
    "inj_ol_diff",
    "inj_skill_diff",
    "inj_def_diff",
    "home_roster_ol",
    "away_roster_ol",
    "home_roster_qb",
    "away_roster_qb",
    "home_roster_db",
    "away_roster_db",
    "roster_ol_diff",
    "roster_qb_diff",
    "roster_db_diff",
    "home_roster_missing",
    "away_roster_missing",
]

ROSTERS = ["data/rosters_weekly_history.parquet", "data/rosters_weekly_latest.parquet"]

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


def add_prior_season_features(df: pl.DataFrame) -> pl.DataFrame:
    """Shrink current-season results toward last season's finish.

    Week 1 has no current-season sample, and a 3-1 start is a noisy estimate of
    strength. Blending the current-season margin/win rate with the previous
    season's final values acts as an empirical-Bayes prior: ``k`` games of
    weight on last season, rising current-season weight as games accumulate.
    """
    played = df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    home = played.select(
        pl.col("season"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Float64).alias("win"),
    )
    away = played.select(
        pl.col("season"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Float64).alias("win"),
    )
    team_season = (
        pl.concat([home, away])
        .group_by(["season", "team"])
        .agg(
            pl.col("pf").sum().alias("pf"),
            pl.col("pa").sum().alias("pa"),
            pl.col("win").mean().alias("win_pct"),
            pl.len().alias("games"),
        )
        .with_columns(
            ((pl.col("pf") - pl.col("pa")) / pl.col("games").clip(1)).alias("diff_pg")
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    prior = team_season.select(
        "season",
        "team",
        pl.col("diff_pg").alias("prev_diff_pg"),
        pl.col("win_pct").alias("prev_win_pct"),
    )
    home_prior = prior.select(
        "season",
        pl.col("team").alias("home_team"),
        pl.col("prev_diff_pg").alias("home_prev_diff_pg"),
        pl.col("prev_win_pct").alias("home_prev_win_pct"),
    )
    away_prior = prior.select(
        "season",
        pl.col("team").alias("away_team"),
        pl.col("prev_diff_pg").alias("away_prev_diff_pg"),
        pl.col("prev_win_pct").alias("away_prev_win_pct"),
    )

    out = df.join(home_prior, on=["season", "home_team"], how="left").join(
        away_prior, on=["season", "away_team"], how="left"
    )
    out = out.with_columns(
        pl.col("home_prev_diff_pg").fill_null(0.0),
        pl.col("away_prev_diff_pg").fill_null(0.0),
        pl.col("home_prev_win_pct").fill_null(0.5),
        pl.col("away_prev_win_pct").fill_null(0.5),
    )

    k = 2.0
    hg = pl.col("home_prior_games").cast(pl.Float64)
    ag = pl.col("away_prior_games").cast(pl.Float64)
    return out.with_columns(
        ((hg * pl.col("home_diff_pg") + k * pl.col("home_prev_diff_pg")) / (hg + k)).alias(
            "home_blend_diff"
        ),
        ((ag * pl.col("away_diff_pg") + k * pl.col("away_prev_diff_pg")) / (ag + k)).alias(
            "away_blend_diff"
        ),
        ((hg * pl.col("home_win_pct") + k * pl.col("home_prev_win_pct")) / (hg + k)).alias(
            "home_blend_win"
        ),
        ((ag * pl.col("away_win_pct") + k * pl.col("away_prev_win_pct")) / (ag + k)).alias(
            "away_blend_win"
        ),
    ).with_columns(
        (pl.col("home_blend_diff") - pl.col("away_blend_diff")).alias("blend_diff_diff"),
        (pl.col("home_blend_win") - pl.col("away_blend_win")).alias("blend_win_diff"),
    )


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
    group = (
        pl.when(pl.col("position") == "QB").then(pl.lit("QB"))
        .when(pl.col("position").is_in(["T", "G", "C"])).then(pl.lit("OL"))
        .when(pl.col("position").is_in(["WR", "TE", "RB", "FB"])).then(pl.lit("SKILL"))
        .when(pl.col("position").is_in(["CB", "S", "LB", "DE", "DT", "NT"])).then(pl.lit("DEF"))
        .otherwise(pl.lit("ST"))
    )
    injuries = injuries.with_columns(
        (impact * pos_weight).alias("impact"),
        group.alias("inj_group"),
    )
    return (
        injuries.group_by(["season", "week", "team"])
        .agg(
            pl.col("impact").sum().alias("inj_impact"),
            pl.when(pl.col("inj_group") == "QB").then(pl.col("impact")).otherwise(0.0).sum().alias("inj_qb"),
            pl.when(pl.col("inj_group") == "OL").then(pl.col("impact")).otherwise(0.0).sum().alias("inj_ol"),
            pl.when(pl.col("inj_group") == "SKILL").then(pl.col("impact")).otherwise(0.0).sum().alias("inj_skill"),
            pl.when(pl.col("inj_group") == "DEF").then(pl.col("impact")).otherwise(0.0).sum().alias("inj_def"),
            pl.len().alias("inj_players"),
        )
        .sort(["season", "week"])
    )


def add_injury_features(df: pl.DataFrame) -> pl.DataFrame:
    table = injury_table()
    if table is None:
        return df
    out = df
    metrics = ["inj_impact", "inj_qb", "inj_ol", "inj_skill", "inj_def"]
    for side in ("home", "away"):
        side_table = table.rename(
            {"team": f"{side}_team", **{m: f"{side}_{m}" for m in metrics}}
        ).select(["season", "week", f"{side}_team"] + [f"{side}_{m}" for m in metrics])
        out = out.join(side_table, on=["season", "week", f"{side}_team"], how="left")
    out = out.with_columns(
        pl.col("home_inj_impact").is_null().cast(pl.Int8).alias("home_inj_missing"),
        pl.col("away_inj_impact").is_null().cast(pl.Int8).alias("away_inj_missing"),
    ).with_columns(
        [pl.col(f"{side}_{m}").fill_null(0.0) for side in ("home", "away") for m in metrics]
    )
    out = out.with_columns(
        (pl.col("home_inj_impact") - pl.col("away_inj_impact")).alias("inj_impact_diff"),
        (pl.col("home_inj_qb") - pl.col("away_inj_qb")).alias("inj_qb_diff"),
        (pl.col("home_inj_ol") - pl.col("away_inj_ol")).alias("inj_ol_diff"),
        (pl.col("home_inj_skill") - pl.col("away_inj_skill")).alias("inj_skill_diff"),
        (pl.col("home_inj_def") - pl.col("away_inj_def")).alias("inj_def_diff"),
    )
    return out


def roster_table() -> pl.DataFrame | None:
    path = next((p for p in ROSTERS if os.path.exists(p)), None)
    if path is None:
        return None
    rosters = pl.read_parquet(path)
    if not {"season", "week", "team", "position", "status"}.issubset(set(rosters.columns)):
        return None
    active = rosters.filter(pl.col("status") == "ACT")
    position = pl.col("position")
    return (
        active.group_by(["season", "week", "team"])
        .agg(
            (position == "QB").sum().alias("roster_qb"),
            (position == "OL").sum().alias("roster_ol"),
            (position == "DB").sum().alias("roster_db"),
            pl.len().alias("roster_active"),
        )
        .sort(["season", "week"])
    )


def add_roster_features(df: pl.DataFrame) -> pl.DataFrame:
    table = roster_table()
    if table is None:
        return df
    out = df
    for side in ("home", "away"):
        side_table = table.rename(
            {
                "team": f"{side}_team",
                "roster_qb": f"{side}_roster_qb",
                "roster_ol": f"{side}_roster_ol",
                "roster_db": f"{side}_roster_db",
            }
        ).select(
            ["season", "week", f"{side}_team", f"{side}_roster_qb", f"{side}_roster_ol", f"{side}_roster_db"]
        )
        out = out.join(side_table, on=["season", "week", f"{side}_team"], how="left")
    out = out.with_columns(
        pl.col("home_roster_ol").is_null().cast(pl.Int8).alias("home_roster_missing"),
        pl.col("away_roster_ol").is_null().cast(pl.Int8).alias("away_roster_missing"),
    ).with_columns(
        [pl.col(f"{side}_roster_{pos}").fill_null(0) for side in ("home", "away") for pos in ("qb", "ol", "db")]
    )
    out = out.with_columns(
        (pl.col("home_roster_ol") - pl.col("away_roster_ol")).alias("roster_ol_diff"),
        (pl.col("home_roster_qb") - pl.col("away_roster_qb")).alias("roster_qb_diff"),
        (pl.col("home_roster_db") - pl.col("away_roster_db")).alias("roster_db_diff"),
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
    df = add_prior_season_features(df)
    df = add_injury_features(df)
    df = add_roster_features(df)
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
