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


def build_match_features(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    team_stats: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build pre-game match features from schedules, PBP, injuries, team stats, and snaps.

    Expects schedules columns: ``game_id``, ``season``, ``week``, ``home_team``,
    ``away_team``, ``home_score``, ``away_score``.

    If ``pbp`` is provided, adds team-level EPA/success features for the home
    and away sides (offensive/defensive EPA, success rate, pass/rush EPA).

    If ``injuries`` is provided, adds per-team weekly injury counts (Out,
    Doubtful, Questionable, total) for both sides.

    If ``team_stats`` is provided, adds team-week numeric stats (EPA, points,
    success rates, etc.) for both sides.

    If ``snap_counts`` is provided, adds team-week snap-count aggregates
    (total snaps, players with ≥50% snap share, mean snap percentage).

    The returned frame adds:
    - ``home_win`` (1 if the home team scored more, else 0; ties dropped)
    - ``point_diff`` / ``total_points``
    - rolling pre-game team averages: ``home_avg_points``,
      ``home_avg_allowed``, ``away_avg_points``, ``away_avg_allowed``
    - optional PBP aggregates: ``home_off_epa``, ``home_def_epa``,
      ``home_success_rate``, ``home_pass_epa``, ``home_rush_epa`` (and away equivalents)
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

    # Rest-day features (days since each team's previous game).
    date_col = "gameday" if "gameday" in sched.columns else ("date" if "date" in sched.columns else None)
    if date_col:
        try:
            sched = sched.with_columns(
                pl.col(date_col).cast(pl.Utf8).str.to_date(strict=False).alias("_game_date")
            )
            home_rest = sched.select(
                "game_id", "season", "week",
                pl.col("home_team").alias("team"),
                pl.col("_game_date").alias("game_date"),
            )
            away_rest = sched.select(
                "game_id", "season", "week",
                pl.col("away_team").alias("team"),
                pl.col("_game_date").alias("game_date"),
            )
            rest_long = (
                pl.concat([home_rest, away_rest])
                .sort(["team", "game_date", "game_id"])
                .with_columns(
                    pl.col("game_date").diff().dt.total_days().over("team").alias("rest_days")
                )
                .with_columns(pl.col("rest_days").fill_null(7).clip(0, 21))
            )
            home_rest_feats = (
                rest_long.join(sched.select("game_id", "home_team"), on="game_id")
                .filter(pl.col("team") == pl.col("home_team"))
                .select("game_id", pl.col("rest_days").alias("home_rest_days"))
                .unique("game_id")
            )
            away_rest_feats = (
                rest_long.join(sched.select("game_id", "away_team"), on="game_id")
                .filter(pl.col("team") == pl.col("away_team"))
                .select("game_id", pl.col("rest_days").alias("away_rest_days"))
                .unique("game_id")
            )
            out = out.join(home_rest_feats, on="game_id", how="left").join(
                away_rest_feats, on="game_id", how="left"
            )
            out = out.with_columns(
                (pl.col("home_rest_days") - pl.col("away_rest_days")).alias("rest_diff"),
                (pl.col("home_rest_days") <= 4).cast(pl.Int8).alias("home_short_week"),
                (pl.col("away_rest_days") <= 4).cast(pl.Int8).alias("away_short_week"),
            )
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not add rest features: {exc}")

    # Optional injury-report counts (per team, per week).
    if injuries is not None and injuries.height > 0:
        try:
            inj = injuries
            required_inj = {"season", "week", "team"}
            if not required_inj.issubset(set(inj.columns)):
                raise ValueError(f"injuries missing {required_inj - set(inj.columns)}")
            status_col = next((c for c in inj.columns if "status" in c.lower()), None)
            if status_col is None:
                raise ValueError("injuries missing a status column")

            inj = inj.with_columns(
                pl.col(status_col).cast(pl.Utf8).str.to_lowercase().alias("_status")
            )
            agg = (
                inj.group_by(["season", "week", "team"])
                .agg(
                    (pl.col("_status") == "out").sum().alias("inj_out"),
                    (pl.col("_status") == "doubtful").sum().alias("inj_doubtful"),
                    (pl.col("_status") == "questionable").sum().alias("inj_questionable"),
                    pl.len().alias("inj_total"),
                )
            )

            home_inj = agg.select(
                "season", "week",
                pl.col("team").alias("home_team"),
                pl.col("inj_out").alias("home_inj_out"),
                pl.col("inj_doubtful").alias("home_inj_doubtful"),
                pl.col("inj_questionable").alias("home_inj_questionable"),
                pl.col("inj_total").alias("home_inj_total"),
            )
            away_inj = agg.select(
                "season", "week",
                pl.col("team").alias("away_team"),
                pl.col("inj_out").alias("away_inj_out"),
                pl.col("inj_doubtful").alias("away_inj_doubtful"),
                pl.col("inj_questionable").alias("away_inj_questionable"),
                pl.col("inj_total").alias("away_inj_total"),
            )

            out = out.join(home_inj, on=["season", "week", "home_team"], how="left").join(
                away_inj, on=["season", "week", "away_team"], how="left"
            )

            inj_cols = [
                "home_inj_out", "home_inj_doubtful", "home_inj_questionable", "home_inj_total",
                "away_inj_out", "away_inj_doubtful", "away_inj_questionable", "away_inj_total",
            ]
            out = out.with_columns([pl.col(c).fill_null(0) for c in inj_cols])
            out = out.with_columns(
                (pl.col("home_inj_out") - pl.col("away_inj_out")).alias("inj_out_diff"),
                (pl.col("home_inj_total") - pl.col("away_inj_total")).alias("inj_total_diff"),
            )
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not add injury features: {exc}")

    # Optional PBP-derived team features (pre-game rolling EPA/success).
    if pbp is not None and pbp.height > 0:
        try:
            p = pbp
            required_pbp = {"game_id", "posteam", "defteam", "epa"}
            if not required_pbp.issubset(set(p.columns)):
                raise ValueError(f"pbp missing {required_pbp - set(p.columns)}")

            if "play_type" in p.columns:
                pass_epa = pl.col("epa").filter(pl.col("play_type") == "pass").mean()
                rush_epa = pl.col("epa").filter(pl.col("play_type") == "run").mean()
            else:
                pass_epa = pl.lit(None, dtype=pl.Float64)
                rush_epa = pl.lit(None, dtype=pl.Float64)

            off = (
                p.filter(pl.col("posteam").is_not_null())
                .group_by(["game_id", "posteam"])
                .agg(
                    pl.col("epa").mean().alias("off_epa"),
                    (pl.col("epa") > 0).mean().alias("success_rate"),
                    pass_epa.alias("pass_epa"),
                    rush_epa.alias("rush_epa"),
                    pl.len().alias("plays"),
                )
                .rename({"posteam": "team"})
            )

            deff = (
                p.filter(pl.col("defteam").is_not_null())
                .group_by(["game_id", "defteam"])
                .agg(pl.col("epa").mean().alias("def_epa"))
                .rename({"defteam": "team"})
            )

            team_game = off.join(deff, on=["game_id", "team"], how="left")
            team_game = team_game.join(
                sched.select("game_id", "season", "week"),
                on="game_id",
                how="left",
            )

            metric_cols = ["off_epa", "def_epa", "success_rate", "pass_epa", "rush_epa"]
            team_game = team_game.with_columns(
                [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in metric_cols]
            ).sort(["season", "week", "game_id", "team"])

            team_game = team_game.with_columns(
                [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in metric_cols]
                + [pl.col("off_epa").cum_count().over("team").alias("game_num")]
            )

            # Pre-game average = (cumulative through current game - current game)
            # divided by the number of prior games.
            for c in metric_cols:
                team_game = team_game.with_columns(
                    ((pl.col(f"cum_{c}") - pl.col(c)) / (pl.col("game_num") - 1).clip(1))
                    .alias(f"pre_{c}")
                )

            home_stats = team_game.select(
                "game_id",
                pl.col("team").alias("home_team"),
                pl.col("pre_off_epa").alias("home_off_epa"),
                pl.col("pre_def_epa").alias("home_def_epa"),
                pl.col("pre_success_rate").alias("home_success_rate"),
                pl.col("pre_pass_epa").alias("home_pass_epa"),
                pl.col("pre_rush_epa").alias("home_rush_epa"),
            )

            away_stats = team_game.select(
                "game_id",
                pl.col("team").alias("away_team"),
                pl.col("pre_off_epa").alias("away_off_epa"),
                pl.col("pre_def_epa").alias("away_def_epa"),
                pl.col("pre_success_rate").alias("away_success_rate"),
                pl.col("pre_pass_epa").alias("away_pass_epa"),
                pl.col("pre_rush_epa").alias("away_rush_epa"),
            )

            out = out.join(home_stats, on=["game_id", "home_team"], how="left").join(
                away_stats, on=["game_id", "away_team"], how="left"
            )
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not add PBP features: {exc}")

    # Optional team-level stats (points, EPA, success rates).
    if team_stats is not None and team_stats.height > 0:
        try:
            ts = team_stats
            team_col = next(
                (c for c in ["team", "posteam", "team_abbr", "club_code"] if c in ts.columns),
                None,
            )
            required_ts = {"season", "week"}
            if team_col is None or not required_ts.issubset(set(ts.columns)):
                raise ValueError("team_stats missing season/week/team columns")

            candidate_cols = [
                "points",
                "points_allowed",
                "offensive_epa",
                "defensive_epa",
                "passing_epa",
                "rushing_epa",
                "special_teams_epa",
                "success_rate",
                "pass_success_rate",
                "rush_success_rate",
                "cpoe",
                "wpa",
                "epa_per_play",
                "pass_epa_per_play",
                "rush_epa_per_play",
            ]
            present_cols = [c for c in candidate_cols if c in ts.columns]
            if not present_cols:
                raise ValueError("team_stats has no numeric stat columns")

            agg = (
                ts.select(["season", "week", pl.col(team_col).alias("team")] + present_cols)
                .group_by(["season", "week", "team"])
                .agg([pl.col(c).mean().alias(c) for c in present_cols])
                .sort(["team", "season", "week"])
            )

            # Convert same-game team stats into pre-game rolling averages.
            agg = agg.with_columns(
                [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in present_cols]
                + [pl.col(present_cols[0]).cum_count().over("team").alias("game_num")]
            )
            for c in present_cols:
                agg = agg.with_columns(
                    ((pl.col(f"cum_{c}") - pl.col(c)) / (pl.col("game_num") - 1).clip(1))
                    .alias(c)
                )

            home_ts = agg.select(
                "season",
                "week",
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"home_{c}") for c in present_cols],
            )
            away_ts = agg.select(
                "season",
                "week",
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"away_{c}") for c in present_cols],
            )

            out = out.join(home_ts, on=["season", "week", "home_team"], how="left").join(
                away_ts, on=["season", "week", "away_team"], how="left"
            )
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not add team-stats features: {exc}")

    # Optional snap-count features (pre-game rolling team aggregates).
    if snap_counts is not None and snap_counts.height > 0:
        try:
            sc = snap_counts
            team_col = next(
                (c for c in ["team", "team_abbr", "club_code"] if c in sc.columns),
                None,
            )
            required_sc = {"season", "week"}
            if team_col is None or not required_sc.issubset(set(sc.columns)):
                raise ValueError("snap_counts missing season/week/team columns")

            agg_exprs = [pl.len().alias("snap_players")]
            if "offense_snaps" in sc.columns:
                agg_exprs.append(pl.col("offense_snaps").sum().alias("snap_off_total"))
            if "defense_snaps" in sc.columns:
                agg_exprs.append(pl.col("defense_snaps").sum().alias("snap_def_total"))
            if "offense_pct" in sc.columns:
                agg_exprs.extend(
                    [
                        pl.col("offense_pct").mean().alias("snap_off_mean_pct"),
                        (pl.col("offense_pct") >= 0.5).sum().alias("snap_off_players_50"),
                    ]
                )
            if "defense_pct" in sc.columns:
                agg_exprs.extend(
                    [
                        pl.col("defense_pct").mean().alias("snap_def_mean_pct"),
                        (pl.col("defense_pct") >= 0.5).sum().alias("snap_def_players_50"),
                    ]
                )

            agg = (
                sc.select(
                    ["season", "week", pl.col(team_col).alias("team")]
                    + [
                        c
                        for c in ["offense_snaps", "defense_snaps", "offense_pct", "defense_pct"]
                        if c in sc.columns
                    ]
                )
                .group_by(["season", "week", "team"])
                .agg(agg_exprs)
                .sort(["team", "season", "week"])
            )

            metric_cols = [c for c in agg.columns if c not in {"season", "week", "team"}]
            agg = agg.with_columns(
                [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in metric_cols]
                + [pl.col(metric_cols[0]).cum_count().over("team").alias("game_num")]
            )
            for c in metric_cols:
                agg = agg.with_columns(
                    ((pl.col(f"cum_{c}") - pl.col(c)) / (pl.col("game_num") - 1).clip(1))
                    .alias(c)
                )

            home_sc = agg.select(
                "season",
                "week",
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"home_{c}") for c in metric_cols],
            )
            away_sc = agg.select(
                "season",
                "week",
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"away_{c}") for c in metric_cols],
            )

            out = out.join(home_sc, on=["season", "week", "home_team"], how="left").join(
                away_sc, on=["season", "week", "away_team"], how="left"
            )
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not add snap-count features: {exc}")

    return out


def add_team_rolling_features(
    match_features: pl.DataFrame,
    df: pl.DataFrame,
    team_col: str,
    stat_cols: list[str],
    prefix: str,
) -> pl.DataFrame:
    """Merge pre-game rolling team-week averages from a player-level dataset.

    ``df`` must contain ``season``, ``week``, ``team_col`` and the requested
    ``stat_cols``.  The resulting columns are named
    ``home_<prefix>_<stat>`` / ``away_<prefix>_<stat>``.
    """
    if df.height == 0 or team_col not in df.columns:
        return match_features

    present = [c for c in stat_cols if c in df.columns]
    if not present:
        return match_features

    agg = (
        df.select(["season", "week", pl.col(team_col).alias("team")] + present)
        .group_by(["season", "week", "team"])
        .agg([pl.col(c).mean().alias(c) for c in present])
        .sort(["team", "season", "week"])
    )

    agg = agg.with_columns(
        [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in present]
        + [pl.col(present[0]).cum_count().over("team").alias("game_num")]
    )
    for c in present:
        agg = agg.with_columns(
            ((pl.col(f"cum_{c}") - pl.col(c)) / (pl.col("game_num") - 1).clip(1)).alias(c)
        )

    home = agg.select(
        "season",
        "week",
        pl.col("team").alias("home_team"),
        *[pl.col(c).alias(f"home_{prefix}_{c}") for c in present],
    )
    away = agg.select(
        "season",
        "week",
        pl.col("team").alias("away_team"),
        *[pl.col(c).alias(f"away_{prefix}_{c}") for c in present],
    )
    return match_features.join(home, on=["season", "week", "home_team"], how="left").join(
        away, on=["season", "week", "away_team"], how="left"
    )


def add_roster_continuity_features(
    match_features: pl.DataFrame,
    rosters_weekly: pl.DataFrame | None,
) -> pl.DataFrame:
    """Add week-over-week roster continuity from weekly rosters.

    ``roster_continuity`` = fraction of a team's current players who were also on
    the same team the previous week.  Week 1 defaults to 1.0.
    """
    if rosters_weekly is None or rosters_weekly.height == 0:
        return match_features

    id_col = next(
        (c for c in ["gsis_id", "player_id", "full_name"] if c in rosters_weekly.columns),
        None,
    )
    required = {"season", "week", "team"}
    if id_col is None or not required.issubset(set(rosters_weekly.columns)):
        print("Could not add roster continuity: missing columns.")
        return match_features

    r = (
        rosters_weekly.select(
            ["season", "week", pl.col("team").alias("team"), pl.col(id_col).alias("player_id")]
        )
        .unique()
    )

    current = r.group_by(["season", "week", "team"]).agg(
        pl.len().alias("roster_size")
    )

    prev = (
        r.with_columns((pl.col("week") - 1).alias("week"))
        .rename({"player_id": "prev_player_id"})
    )

    joined = r.join(prev, on=["season", "week", "team"], how="left")
    continuity = (
        joined.group_by(["season", "week", "team"])
        .agg((pl.col("prev_player_id").is_not_null()).sum().alias("returning_players"))
        .join(current, on=["season", "week", "team"], how="left")
        .with_columns(
            pl.when(pl.col("week") <= 1)
            .then(1.0)
            .otherwise(pl.col("returning_players") / pl.col("roster_size"))
            .clip(0.0, 1.0)
            .alias("roster_continuity")
        )
    )

    home = continuity.select(
        "season",
        "week",
        pl.col("team").alias("home_team"),
        pl.col("roster_continuity").alias("home_roster_continuity"),
    )
    away = continuity.select(
        "season",
        "week",
        pl.col("team").alias("away_team"),
        pl.col("roster_continuity").alias("away_roster_continuity"),
    )

    out = match_features.join(home, on=["season", "week", "home_team"], how="left").join(
        away, on=["season", "week", "away_team"], how="left"
    )
    out = out.with_columns(
        [
            pl.col("home_roster_continuity").fill_null(1.0),
            pl.col("away_roster_continuity").fill_null(1.0),
        ]
    ).with_columns(
        (
            pl.col("home_roster_continuity") - pl.col("away_roster_continuity")
        ).alias("roster_continuity_diff")
    )
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
