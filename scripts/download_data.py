#!/usr/bin/env python3
"""Download NFL data using nflreadpy and write Parquet files.

This script is intended to be run in GitHub Actions (or locally) to fetch
selected nflverse tables and persist them as Parquet files under ./data/.
The ./data/ directory is git‑ignored so raw data never pollutes history.
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is on the path so we can import src.
sys.path.insert(0, os.path.dirname(__file__))  # dirname of scripts -> repo root

from src.ingest import fetch_schedules, fetch_pbp, fetch_player_stats, fetch_team_stats
import polars as pl

def main() -> None:
    # Create the data directory (git‑ignored, safe to mkdir –p)
    os.makedirs("data", exist_ok=True)

    # Schedules (all seasons)
    print("Fetching schedules…")
    df = fetch_schedules()
    df.write_parquet("data/schedules.parquet")
    print(f"Wrote schedules: {df.shape[0]} rows x {df.shape[1]} cols")

    # PBP for 2024 week 1
    print("Fetching pbp (2024 wk1)…")
    df2 = fetch_pbp(season=2024, week=1)
    df2.write_parquet("data/pbp_2024_week1.parquet")
    print(f"Wrote pbp: {df2.shape[0]} rows x {df2.shape[1]} cols")

    # Player stats for 2024 week 1
    print("Fetching player_stats (2024 wk1)…")
    df3 = fetch_player_stats(season=2024, week=1)
    df3.write_parquet("data/player_stats_2024_week1.parquet")
    print(f"Wrote player_stats: {df3.shape[0]} rows x {df3.shape[1]} cols")

    # Team stats for 2024 week 1
    print("Fetching team_stats (2024 wk1)…")
    df4 = fetch_team_stats(season=2024, week=1)
    df4.write_parquet("data/team_stats_2024_week1.parquet")
    print(f"Wrote team_stats: {df4.shape[0]} rows x {df4.shape[1]} cols")

    print("All data downloaded and written to data/.")


if __name__ == "__main__":
    main()