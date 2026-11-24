#!/usr/bin/env python3
"""Build feature tables from raw NFL data.

Reads Parquet/CSV files from ``data/`` and writes feature tables to
``features/``.  Designed to run in GitHub Actions or locally.

If the raw data is missing (e.g., the download step failed), the script
prints a warning and exits 0 so the workflow can continue gracefully.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import polars as pl

from src.features import (
    add_ftn_features,
    add_participation_features,
    add_roster_continuity_features,
    add_team_rolling_features,
    build_match_features,
    build_player_features,
)


def load_table(name: str) -> pl.DataFrame | None:
    """Load ``data/<name>.parquet`` if it exists, otherwise ``data/<name>.csv``."""
    parquet = f"data/{name}.parquet"
    csv = f"data/{name}.csv"
    if os.path.exists(parquet):
        return pl.read_parquet(parquet)
    if os.path.exists(csv):
        return pl.read_csv(csv)
    return None


def main() -> None:
    os.makedirs("features", exist_ok=True)

    schedules = load_table("schedules")
    if schedules is None:
        print("No schedules table found; skipping match features.")
    else:
        try:
            pbp = load_table("pbp_latest")
            injuries = load_table("injuries_latest")
            team_stats_raw = load_table("team_stats_latest")
            snap_counts_raw = load_table("snap_counts_latest")
            match_features = build_match_features(
                schedules,
                pbp=pbp,
                injuries=injuries,
                team_stats=team_stats_raw,
                snap_counts=snap_counts_raw,
            )

            ng_pass = load_table("nextgen_passing_latest")
            if ng_pass is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    ng_pass,
                    team_col="team_abbr",
                    stat_cols=[
                        "avg_time_to_throw",
                        "avg_completed_air_yards",
                        "avg_intended_air_yards",
                        "aggressiveness",
                        "completion_percentage_above_expectation",
                        "passer_rating",
                    ],
                    prefix="ng_pass",
                )

            ng_rush = load_table("nextgen_rushing_latest")
            if ng_rush is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    ng_rush,
                    team_col="team_abbr",
                    stat_cols=[
                        "efficiency",
                        "percent_attempts_gte_eight_defenders",
                        "avg_time_to_los",
                        "rush_yards_over_expected_per_att",
                        "rush_pct_over_expected",
                    ],
                    prefix="ng_rush",
                )

            ng_rec = load_table("nextgen_receiving_latest")
            if ng_rec is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    ng_rec,
                    team_col="team_abbr",
                    stat_cols=[
                        "avg_cushion",
                        "avg_separation",
                        "percent_share_of_intended_air_yards",
                        "catch_percentage",
                        "avg_yac_above_expectation",
                    ],
                    prefix="ng_rec",
                )

            pfr_pass = load_table("pfr_pass_latest")
            if pfr_pass is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    pfr_pass,
                    team_col="team",
                    stat_cols=[
                        "passing_drop_pct",
                        "passing_bad_throw_pct",
                        "times_pressured_pct",
                    ],
                    prefix="pfr_pass",
                )

            pfr_rush = load_table("pfr_rush_latest")
            if pfr_rush is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    pfr_rush,
                    team_col="team",
                    stat_cols=[
                        "rushing_yards_before_contact_avg",
                        "rushing_yards_after_contact_avg",
                    ],
                    prefix="pfr_rush",
                )

            pfr_rec = load_table("pfr_rec_latest")
            if pfr_rec is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    pfr_rec,
                    team_col="team",
                    stat_cols=[
                        "receiving_drop_pct",
                        "receiving_rat",
                    ],
                    prefix="pfr_rec",
                )

            pfr_def = load_table("pfr_def_latest")
            if pfr_def is not None:
                match_features = add_team_rolling_features(
                    match_features,
                    pfr_def,
                    team_col="team",
                    stat_cols=[
                        "def_completion_pct",
                        "def_yards_allowed_per_tgt",
                        "def_passer_rating_allowed",
                        "def_missed_tackle_pct",
                        "def_pressures",
                        "def_sacks",
                    ],
                    prefix="pfr_def",
                )

            rosters_weekly_raw = load_table("rosters_weekly_latest")
            match_features = add_roster_continuity_features(
                match_features, rosters_weekly_raw
            )

            participation_raw = load_table("participation_latest")
            match_features = add_participation_features(
                match_features, participation_raw, schedules
            )

            ftn_raw = load_table("ftn_charting_latest")
            match_features = add_ftn_features(match_features, ftn_raw, pbp)

            match_features.write_parquet("features/match_features.parquet")
            print(f"Wrote features/match_features.parquet ({match_features.height} rows)")
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not build match features: {exc}")

    player_stats = load_table("player_stats_latest") or load_table("player_stats_2024_week1")
    if player_stats is None:
        print("No player_stats table found; skipping player features.")
    else:
        try:
            player_features = build_player_features(player_stats)
            player_features.write_parquet("features/player_features.parquet")
            print(f"Wrote features/player_features.parquet ({player_features.height} rows)")
        except Exception as exc:  # pragma: no cover
            print(f"Could not build player features: {exc}")

    print("Feature build complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Feature build error: {exc}")
    sys.exit(0)