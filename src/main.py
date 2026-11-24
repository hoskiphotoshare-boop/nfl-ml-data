#!/usr/bin/env python3
"""CLI entry point for the nfl-ml-data package.

Install with `pip install -e .` and run commands like:

    nfl-ml-data pbp --season 2024 --week 1
    nfl-ml-data schedules --season 2024
    nfl-ml-data players --season 2024 --week 1
    nfl-ml-data teams --season 2024 --week 1
"""

from __future__ import annotations

import sys

import typer

from src.ingest import fetch_schedules, fetch_pbp, fetch_player_stats, fetch_team_stats

app = typer.Typer()


@app.command()
def schedules(season: int | None = None) -> None:
    """Fetch and print schedule shape."""
    df = fetch_schedules(season=season)
    typer.echo(f"schedule rows={df.shape[0]} cols={df.shape[1]}")


@app.command()
def pbp(season: int = 2024, week: int | None = None) -> None:
    """Fetch and print pbp shape."""
    df = fetch_pbp(season=season, week=week)
    typer.echo(f"pbp rows={df.shape[0]} cols={df.shape[1]}")


@app.command()
def players(season: int = 2024, week: int | None = None) -> None:
    """Fetch and print player-stats shape."""
    df = fetch_player_stats(season=season, week=week)
    typer.echo(f"player_stats rows={df.shape[0]} cols={df.shape[1]}")


@app.command()
def teams(season: int = 2024, week: int | None = None) -> None:
    """Fetch and print team-stats shape."""
    df = fetch_team_stats(season=season, week=week)
    typer.echo(f"team_stats rows={df.shape[0]} cols={df.shape[1]}")


def main() -> None:
    """Allow running as ``python -m src.main`` or ``nfl-ml-data``."""
    app()


if __name__ == "__main__":
    main()