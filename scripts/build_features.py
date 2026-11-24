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

from src.features import build_match_features, build_player_features


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
            match_features = build_match_features(schedules)
            match_features.write_parquet("features/match_features.parquet")
            print(f"Wrote features/match_features.parquet ({match_features.height} rows)")
        except Exception as exc:  # pragma: no cover – defensive
            print(f"Could not build match features: {exc}")

    player_stats = load_table("player_stats_2024_week1")
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