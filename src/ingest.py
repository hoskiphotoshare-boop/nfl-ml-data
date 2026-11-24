"""Data ingestion module using nflreadpy.


.. NOTE: This module is intended to be used programmatically or as a CLI
    entry point via ``nfl-ml-ingest``.
"""


def main() -> None:
    """Entry point for ``nfl-ml-ingest`` CLI script."""
    print("nfl-ml-data ingestion module — use the API or run ``nfl-ml-ingest``.")


if __name__ == "__main__":
    main()

Fetches NFL open datasets dynamically at runtime. No raw data is committed
to Git history; all data is fetched fresh on each CI run or local execution.
"""

from __future__ import annotations

import polars as pl
from nflreadpy import (
    get_schedules,
    get_pbp,
    get_player_stats,
    get_team_stats,
    get_nextgen_stats,
    get_snap_counts,
    get_depth_charts,
    get_injuries,
    get_rosters_weekly,
    get_rosters,
)


def fetch_schedules(season: int | None = None) -> pl.DataFrame:
    """Fetch NFL schedules.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the most recent available
        season.

    Returns
    -------
    pl.DataFrame
        Schedule table with games, dates, results, etc.
    """
    df = get_schedules(season=season)
    return pl.from_pandas(df)


def fetch_pbp(
    season: int | None = None,
    week: int | None = None,
    franchise: str | None = None,
) -> pl.DataFrame:
    """Fetch play-by-play data.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the most recent available
        season.
    week: int | None
        Specific week number. If None, returns all weeks for the given season.
    franchise: str | None
        Franchise abbreviation filter (e.g. "NYJ").

    Returns
    -------
    pl.DataFrame
        Play-by-play table with down, distance, yard line, result, EPA, WPA, etc.
    """
    df = get_pbp(season=season, week=week, franchise=franchise)
    return pl.from_pandas(df)


def fetch_player_stats(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch player game statistics.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the most recent available
        season.
    week: int | None
        Specific week number. If None, returns all weeks for the given season.

    Returns
    -------
    pl.DataFrame
        Player game stats table with targets, receptions, yards, touchdowns, etc.
    """
    df = get_player_stats(season=season, week=week)
    return pl.from_pandas(df)


def fetch_team_stats(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch team statistics.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the most recent available
        season.
    week: int | None
        Specific week number. If None, returns all weeks for the given season.

    Returns
    -------
    pl.DataFrame
        Team stat table (offensive/defensive metrics, etc.).
    """
    df = get_team_stats(season=season, week=week)
    return pl.from_pandas(df)


def fetch_nextgen_stats(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch next-gen stats (passing, rushing, receiving).

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the most recent available
        season.
    week: int | None
        Specific week number. If None, returns all weeks for the given season.

    Returns
    -------
    pl.DataFrame
        Next-gen stats table with expected metrics, air yards, etc.
    """
    df = get_nextgen_stats(season=season, week=week)
    return pl.from_pandas(df)


def fetch_snap_counts(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch snap counts per player per game.

    Parameters
    ----------
    season: int | None
        Four-digit season year.
    week: int | None
        Specific week number.

    Returns
    -------
    pl.DataFrame
        Snap count table.
    """
    df = get_snap_counts(season=season, week=week)
    return pl.from_pandas(df)


def fetch_depth_charts(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch depth charts.

    Parameters
    ----------
    season: int | None
        Four-digit season year.
    week: int | None
        Specific week number.

    Returns
    -------
    pl.DataFrame
        Depth chart table.
    """
    df = get_depth_charts(season=season, week=week)
    return pl.from_pandas(df)


def fetch_injuries(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch injury reports.

    Parameters
    ----------
    season: int | None
        Four-digit season year.
    week: int | None
        Specific week number.

    Returns
    -------
    pl.DataFrame
        Injury report table.
    """
    df = get_injuries(season=season, week=week)
    return pl.from_pandas(df)


def fetch_rosters_weekly(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch weekly rosters.

    Parameters
    ----------
    season: int | None
        Four-digit season year.
    week: int | None
        Specific week number.

    Returns
    -------
    pl.DataFrame
        Weekly roster table with status, position, etc.
    """
    df = get_rosters_weekly(season=season, week=week)
    return pl.from_pandas(df)


def fetch_rosters(
    season: int | None = None,
) -> pl.DataFrame:
    """Fetch final rosters.

    Parameters
    ----------
    season: int | None
        Four-digit season year.

    Returns
    -------
    pl.DataFrame
        Roster table.
    """
    df = get_rosters(season=season)
    return pl.from_pandas(df)