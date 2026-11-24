"""Data ingestion module using nflreadpy.

Fetches NFL open datasets dynamically at runtime. No raw data is committed to
Git history; all data is fetched fresh on each CI run or local execution.

The wrappers below normalise the public ``nflreadpy`` API (``load_*`` functions)
and support optional week/franchise filtering for convenience.
"""

from __future__ import annotations

import polars as pl
from nflreadpy import (
    load_schedules,
    load_pbp,
    load_player_stats,
    load_team_stats,
    load_nextgen_stats,
    load_snap_counts,
    load_depth_charts,
    load_injuries,
    load_rosters_weekly,
    load_rosters,
)

try:  # optional tables
    from nflreadpy import load_ftn_charting, load_participation
    FTN_CHARTING_AVAILABLE = True
    PARTICIPATION_AVAILABLE = True
except Exception:  # pragma: no cover
    FTN_CHARTING_AVAILABLE = False
    PARTICIPATION_AVAILABLE = False


def fetch_schedules(season: int | None = None) -> pl.DataFrame:
    """Fetch NFL schedules.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns all available seasons.

    Returns
    -------
    pl.DataFrame
        Schedule table with games, dates, results, etc.
    """
    return load_schedules(seasons=True if season is None else season)


def fetch_pbp(
    season: int | None = None,
    week: int | None = None,
    franchise: str | None = None,
) -> pl.DataFrame:
    """Fetch play-by-play data.

    Parameters
    ----------
    season: int | None
        Four-digit season year. If None, returns the current season.
    week: int | None
        Specific week number. If None, returns all weeks.
    franchise: str | None
        Franchise abbreviation filter (e.g. "NYJ").

    Returns
    -------
    pl.DataFrame
        Play-by-play table with down, distance, yard line, result, EPA, WPA, etc.
    """
    df = load_pbp(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    if franchise is not None:
        clauses = []
        if "posteam" in df.columns:
            clauses.append(pl.col("posteam") == franchise)
        if "defteam" in df.columns:
            clauses.append(pl.col("defteam") == franchise)
        if clauses:
            expr = clauses[0]
            for clause in clauses[1:]:
                expr = expr | clause
            df = df.filter(expr)
    return df


def fetch_player_stats(
    season: int | None = None,
    week: int | None = None,
    summary_level: str = "week",
) -> pl.DataFrame:
    """Fetch player game statistics."""
    df = load_player_stats(
        seasons=season,
        summary_level=summary_level,  # type: ignore[arg-type]
    )
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_team_stats(
    season: int | None = None,
    week: int | None = None,
    summary_level: str = "week",
) -> pl.DataFrame:
    """Fetch team statistics."""
    df = load_team_stats(
        seasons=season,
        summary_level=summary_level,  # type: ignore[arg-type]
    )
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_nextgen_stats(
    season: int | None = None,
    week: int | None = None,
    stat_type: str = "passing",
) -> pl.DataFrame:
    """Fetch next-gen stats (passing, rushing, receiving)."""
    df = load_nextgen_stats(seasons=season, stat_type=stat_type)  # type: ignore[arg-type]
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_snap_counts(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch snap counts per player per game."""
    df = load_snap_counts(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_depth_charts(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch depth charts."""
    df = load_depth_charts(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_injuries(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch injury reports."""
    df = load_injuries(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_rosters_weekly(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch weekly rosters."""
    df = load_rosters_weekly(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_rosters(season: int | None = None) -> pl.DataFrame:
    """Fetch final rosters."""
    return load_rosters(seasons=season)


def fetch_ftn_charting(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch FTN charting data (CC-BY-SA 4.0; attribution required)."""
    if not FTN_CHARTING_AVAILABLE:
        raise RuntimeError("nflreadpy ftn_charting not available")
    df = load_ftn_charting(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def fetch_participation(
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Fetch participation records (CC-BY 4.0)."""
    if not PARTICIPATION_AVAILABLE:
        raise RuntimeError("nflreadpy participation not available")
    df = load_participation(seasons=season)
    if week is not None and "week" in df.columns:
        df = df.filter(pl.col("week") == week)
    return df


def main() -> None:
    """Entry point for the ingestion module."""
    print("nfl-ml-data ingestion module — use the API.")


if __name__ == "__main__":
    main()