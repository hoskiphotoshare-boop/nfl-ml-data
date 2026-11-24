"""Feature-engineering module using Polars.

Transforms raw NFL tables (fetched via ``nflreadpy``) into feature-ready
DataFrames for predictive modeling of match outcomes, player game stats, and
playoff outcomes.
"""

from __future__ import annotations

import polars as pl


def main() -> None:
    """Entry point for ``nfl-ml-features`` CLI script."""
    print("nfl-ml-data feature engineering module — use the API.")


def add_epa_wpa(df: pl.DataFrame) -> pl.DataFrame:
    """Add EPA (Expected Points Added) and WPA (Win Probability Added) columns.

    Expects a play-by-play DataFrame with at least the following columns (case‑
    insensitive, Polars‑compatible names):

    - ``posteam`` — team with possession
    - ``defteam`` — defensive team
    - ``quarter`` — quarter number (1‑4, or OT)
    - ``time_left`` — time remaining in the quarter
    - — ``drive`` or scoring context columns

    Returns
    -------
    pl.DataFrame
        Input frame with ``epa`` and ``wpa`` columns appended (if computable).
    """
    # TODO: implement EPA/WPA computation from nflreadpy pbp data
    # Example skeleton:
    # df = df.with_columns(
    #     (pl.col("posteapoints") - pl.col("preteampoints")).alias("epa"),
    #     ...
    # )
    return df


def add_success_rate(df: pl.DataFrame) -> pl.DataFrame:
    """Add success-rate (defined as >50% of expected EPA added) column.

    Parameters
    ----------
    df : pl.DataFrame
        Play-by-play DataFrame typically output from ``fetch_pbp``.

    Returns
    -------
    pl.DataFrame
        Frame with a ``success_rate`` boolean column indicating whether the
        play added >0 EPA (i.e., a "successful" play in the sense of 1st down
        or conversion territory).
    """
    # Placeholder: compute success as EPA > 0
    if "epa" in df.columns:
        return df.with_columns(pl.col("epa").gt(0).alias("success"))
    # If EPA not yet computed, derive from scoring/field-position shifts.
    return df


def engineer_player_features(
    pbp: pl.DataFrame,
    player_stats: pl.DataFrame,
    nextgen: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    rosters_weekly: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a unified player-level feature DataFrame.

    Combines play-by-play, player game stats, next-gen stats, snap counts,
    depth charts, injury reports, and weekly rosters into a single wide
    DataFrame keyed by ``player_id`` and ``game_id`` (or ``gsis_id``).

    The returned frame typically includes (but is not limited to):

    - Target share, air yards share, WOPR (from ``player_stats``)
    - EPA per route, success rate per target (from ``pbp`` + ``player_stats``)
    - Next-gen rushing/receiving/passing metrics (from ``nextgen``)
    - Snap‑percentage and role change (from ``snap_counts``)
    - Depth‑chart position and status (from ``depth_charts``)
    - Injury indicator and days missed (from ``injuries``)
    - Roster status (active/inactive) (from ``rosters_weekly``)

    Parameters
    ----------
    pbp : pl.DataFrame
        Play-by-play DataFrame.
    player_stats : pl.DataFrame
        Player game-statistics DataFrame.
    nextgen : pl.DataFrame | None
        Next-gen statistics DataFrame.
    snap_counts : pl.DataFrame | None
        Snap-count DataFrame.
    depth_charts : pl.DataFrame | None
        Depth‑chart DataFrame.
    injuries : pl.DataFrame | None
        Injury report DataFrame.
    rosters_weekly : pl.DataFrame | None
        Weekly roster DataFrame.

    Returns
    -------
    pl.DataFrame
        Unified player‑level feature set.
    """
    # Start from player_stats as the base
    out = player_stats

    # Example: enrich with next-gen if available
    if nextgen is not None:
        out = out.join(
            nextgen,
            left_on="gsis_id",
            right_on="gsis_id",
            how="left",
        )

    # Example: enrich with snap counts
    if snap_counts is not None:
        out = out.join(
            snap_counts,
            left_on=["gsis_id", "game_id"],
            right_on=["gsis_id", "game_id"],
            how="left",
        )

    # Example: enrich with depth charts
    if depth_charts is not None:
        out = out.join(
            depth_charts,
            left_on=["gsis_id", "game_id"],
            right_on=["gsis_id", "game_id"],
            how="left",
        )

    # Example: enrich with injuries
    if injuries is not None:
        out = out.join(
            injuries,
            left_on=["gsis_id", "game_id"],
            right_on=["gsis_id", "game_id"],
            how="left",
        )

    # Example: enrich with weekly rosters
    if rosters_weekly is not None:
        out = out.join(
            rosters_weekly,
            left_on=["gsis_id", "game_id"],
            right_on=["gsis_id", "game_id"],
            how="left",
        )

    # Derive target share and air yards share
    if "target_share" not in out.columns and "targets" in out.columns:
        # If total attempts are available, compute share; otherwise leave as-is.
        pass

    # Derive WOPR-like metric (placeholder)
    if "wopr" not in out.columns:
        out = out.with_columns(
            (pl.col("receptions") / pl.col("targets").cast(pl.Float64, default=0))
            .alias("target_conversion_rate")
        )

    return out


def build_match_features(schedules: pl.DataFrame) -> pl.DataFrame:
    """Build pre-game match features from a schedules table.

    Expects columns: ``game_id``, ``season``, ``week``, ``home_team``,
    ``away_team``, ``home_score``, ``away_score``.

    The returned frame adds:
    - ``home_win`` (1 if the home team scored more, else 0; ties dropped)
    - ``point_diff`` / ``total_points``
    - rolling pre-game team averages: ``home_avg_points``,
      ``home_avg_allowed``, ``away_avg_points``, ``away_avg_allowed``
    """
    required = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(set(schedules.columns)):
        raise ValueError(f"schedules missing columns: {required - set(schedules.columns)}")

    # Drop unplayed games (null scores) and ties.
    sched = schedules.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("home_score") != pl.col("away_score"))
    )

    if sched.height == 0:
        return sched.with_columns(
            pl.lit(0).alias("home_win"),
            pl.lit(0).alias("point_diff"),
            pl.lit(0).alias("total_points"),
        )

    sched = sched.with_columns(
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"),
        (pl.col("home_score") - pl.col("away_score")).alias("point_diff"),
        (pl.col("home_score") + pl.col("away_score")).alias("total_points"),
    ).sort(["season", "week", "game_id"])

    # Long format: one row per team per game.
    home = sched.select(
        "game_id", "season", "week",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_score").alias("points"),
        pl.col("away_score").alias("points_allowed"),
    )
    away = sched.select(
        "game_id", "season", "week",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_score").alias("points"),
        pl.col("home_score").alias("points_allowed"),
    )
    long = pl.concat([home, away]).sort(["season", "week", "game_id", "team"])

    # Cumulative pre-game averages per team.
    long = long.with_columns(
        pl.col("points").cum_sum().over("team").alias("cum_points"),
        pl.col("points_allowed").cum_sum().over("team").alias("cum_allowed"),
        pl.col("points").cum_count().over("team").alias("game_num"),
    ).with_columns(
        ((pl.col("cum_points") - pl.col("points")) / (pl.col("game_num") - 1).clip(1)).alias("avg_points"),
        ((pl.col("cum_allowed") - pl.col("points_allowed")) / (pl.col("game_num") - 1).clip(1)).alias("avg_allowed"),
    )

    home_feats = long.filter(pl.col("team") == pl.col("team")).join(
        sched.select("game_id", "home_team"), on="game_id"
    ).filter(pl.col("team") == pl.col("home_team")).select(
        "game_id",
        pl.col("avg_points").alias("home_avg_points"),
        pl.col("avg_allowed").alias("home_avg_allowed"),
    ).unique(subset=["game_id"])

    away_feats = long.join(
        sched.select("game_id", "away_team"), on="game_id"
    ).filter(pl.col("team") == pl.col("away_team")).select(
        "game_id",
        pl.col("avg_points").alias("away_avg_points"),
        pl.col("avg_allowed").alias("away_avg_allowed"),
    ).unique(subset=["game_id"])

    out = sched.join(home_feats, on="game_id", how="left").join(away_feats, on="game_id", how="left")
    return out


def build_player_features(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Build player-level features from a player-stats table.

    Expects columns such as ``gsis_id``, ``game_id``, ``team``, ``positions``,
    ``receptions``, ``targets``, ``yards``, ``touchdowns``.

    Adds ``fantasy_points`` using half-PPR scoring and ``target_conversion_rate``.
    """
    if player_stats.height == 0:
        return player_stats

    # Normalise common column names that nflreadpy uses.
    col_map = {
        "receiving_receptions": "receptions",
        "receiving_targets": "targets",
        "receiving_yards": "yards",
        "receiving_tds": "touchdowns",
    }
    for old, new in col_map.items():
        if old in player_stats.columns and new not in player_stats.columns:
            player_stats = player_stats.rename({old: new})

    cols = player_stats.columns
    exprs = []
    if "receptions" in cols:
        exprs.append(pl.col("receptions").fill_null(0).alias("receptions"))
    if "targets" in cols:
        exprs.append(pl.col("targets").fill_null(0).alias("targets"))
    if "yards" in cols:
        exprs.append(pl.col("yards").fill_null(0).alias("yards"))
    if "touchdowns" in cols:
        exprs.append(pl.col("touchdowns").fill_null(0).alias("touchdowns"))

    out = player_stats
    if exprs:
        out = out.with_columns(exprs)

    # Half-PPR fantasy points.
    fp = pl.lit(0.0)
    if "receptions" in out.columns:
        fp = fp + pl.col("receptions").fill_null(0) * 0.5
    if "yards" in out.columns:
        fp = fp + pl.col("yards").fill_null(0) * 0.1
    if "touchdowns" in out.columns:
        fp = fp + pl.col("touchdowns").fill_null(0) * 6.0
    out = out.with_columns(fp.alias("fantasy_points"))

    if "targets" in out.columns and "receptions" in out.columns:
        out = out.with_columns(
            (pl.col("receptions") / pl.col("targets").cast(pl.Float64, strict=False))
            .fill_null(0)
            .alias("target_conversion_rate")
        )
    return out

if __name__ == "__main__":
    main()
