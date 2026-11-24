"""Feature-engineering module using Polars.


.. NOTE: This module is intended to be used programmatically or as a CLI
    entry point via ``nfl-ml-features``.
"""


def main() -> None:
    """Entry point for ``nfl-ml-features`` CLI script."""
    print("nfl-ml-data feature engineering module — use the API.")


if __name__ == "__main__":
    main()

Transforms raw NFL tables (fetched via ``nflreadpy``) into feature-ready
DataFrames for predictive modeling of match outcomes, player game stats, and
playoff outcomes.
"""

from __future__ import annotations

import polars as pl


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