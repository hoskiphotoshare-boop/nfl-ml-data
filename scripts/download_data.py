#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the latest NFL data using nflreadpy and write Parquet files.

The script discovers the current NFL season (2026 for the 2026/27 season) and
fetches the most recent schedules, play-by-play, player stats, and team stats.
It writes ``*_latest.parquet`` files under ``data/`` so downstream feature
engineering always uses the freshest data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

from nflreadpy import get_current_season, get_current_week

from src.ingest import (
    fetch_schedules,
    fetch_pbp,
    fetch_player_stats,
    fetch_team_stats,
    fetch_injuries,
    fetch_trades,
    fetch_rosters_weekly,
    fetch_depth_charts,
    fetch_players,
)
import polars as pl


def main() -> None:
    os.makedirs("data", exist_ok=True)

    current_season = get_current_season()
    current_week = get_current_week()
    print(f"Current NFL season: {current_season}, week: {current_week}")

    # 1) Schedules: all seasons (1999-present) so the match model has history.
    print("Fetching schedules (all seasons)…")
    schedules = fetch_schedules()
    schedules.write_parquet("data/schedules.parquet")
    print(f"Wrote schedules: {schedules.shape[0]} rows x {schedules.shape[1]} cols")

    # Use the current season plus four prior seasons for model training.
    seasons_for_model = list(range(max(current_season - 4, 1999), current_season + 1))

    # 2) Play-by-play for the recent seasons (all available weeks).
    print(f"Fetching pbp ({seasons_for_model})…")
    pbp = fetch_pbp(season=seasons_for_model)
    pbp.write_parquet("data/pbp_latest.parquet")
    print(f"Wrote pbp: {pbp.shape[0]} rows x {pbp.shape[1]} cols")

    # 3) Player stats for the recent seasons.
    print(f"Fetching player_stats ({seasons_for_model})…")
    players = fetch_player_stats(season=seasons_for_model)
    players.write_parquet("data/player_stats_latest.parquet")
    print(f"Wrote player_stats: {players.shape[0]} rows x {players.shape[1]} cols")

    # 4) Team stats for the recent seasons.
    print(f"Fetching team_stats ({seasons_for_model})…")
    teams = fetch_team_stats(season=seasons_for_model)
    teams.write_parquet("data/team_stats_latest.parquet")
    print(f"Wrote team_stats: {teams.shape[0]} rows x {teams.shape[1]} cols")

    # 5) Injury reports for the recent seasons.
    print(f"Fetching injuries ({seasons_for_model})…")
    injuries = fetch_injuries(season=seasons_for_model)
    injuries.write_parquet("data/injuries_latest.parquet")
    print(f"Wrote injuries: {injuries.shape[0]} rows x {injuries.shape[1]} cols")

    # 6) Trades, weekly rosters, depth charts, and player metadata.
    print("Fetching trades…")
    trades = fetch_trades()
    trades.write_parquet("data/trades_latest.parquet")
    print(f"Wrote trades: {trades.shape[0]} rows x {trades.shape[1]} cols")

    print(f"Fetching rosters_weekly ({current_season})…")
    rosters = fetch_rosters_weekly(season=current_season)
    rosters.write_parquet("data/rosters_weekly_latest.parquet")
    print(f"Wrote rosters_weekly: {rosters.shape[0]} rows x {rosters.shape[1]} cols")

    print(f"Fetching depth_charts ({current_season})…")
    depth = fetch_depth_charts(season=current_season)
    depth.write_parquet("data/depth_charts_latest.parquet")
    print(f"Wrote depth_charts: {depth.shape[0]} rows x {depth.shape[1]} cols")

    print("Fetching players…")
    players_master = fetch_players()
    players_master.write_parquet("data/players_latest.parquet")
    print(f"Wrote players: {players_master.shape[0]} rows x {players_master.shape[1]} cols")

    print("Latest data downloaded and written to data/.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Error during data download: {exc}")
    sys.exit(0)