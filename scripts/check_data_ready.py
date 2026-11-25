#!/usr/bin/env python3
"""Check whether the latest NFL week is complete before retraining.

Exit 0 when every game in the most recent week with results has a score;
exit 1 when a game is still missing a result (for example, MNF data has not
landed yet). The Tuesday workflow retries this check before training.
"""

from __future__ import annotations

import os
import sys

import polars as pl


def main() -> int:
    path = "data/schedules.parquet"
    if not os.path.exists(path):
        print("No schedules found; cannot verify data readiness.")
        return 1
    frame = pl.read_parquet(path)
    required = {"season", "week", "home_score", "away_score"}
    if not required.issubset(set(frame.columns)):
        print("Schedules missing score columns; cannot verify readiness.")
        return 1

    played = frame.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if played.is_empty():
        print("No completed games found.")
        return 1

    latest_season = int(played["season"].max())
    season = frame.filter(pl.col("season") == latest_season)
    played_season = played.filter(pl.col("season") == latest_season)
    latest_week = int(played_season["week"].max())
    week_games = season.filter(pl.col("week") == latest_week)
    missing = week_games.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    )

    if missing.height:
        print(
            f"Latest week incomplete: {latest_season} week {latest_week} "
            f"has {missing.height} game(s) without scores."
        )
        for row in missing.select(
            ["away_team", "home_team", "gameday"]
        ).iter_rows(named=True):
            print(
                f"  missing: {row.get('away_team')} @ {row.get('home_team')} "
                f"({row.get('gameday')})"
            )
        return 1

    print(
        f"Latest week complete: {latest_season} week {latest_week} "
        f"({week_games.height} games)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
