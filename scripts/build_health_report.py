#!/usr/bin/env python3
"""Build ``reports/model_health.json`` for the admin diagnostics page."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def main() -> None:
    market = load("reports/market_comparison.json", {})
    backtest = load("reports/backtest.json", {})
    futures = load("reports/futures_comparison.json", {})
    movement = load("reports/line_movement.json", {})
    playoff = load("reports/playoff_backtest.json", {})
    player_backtest = load("reports/player_backtest.json", {})
    calls = load("reports/calls_scoreboard.json", {})
    players = load("predictions/players_week.json", [])
    upcoming = load("predictions/upcoming_games.json", [])
    platt = load("models/platt.json", {})
    odds_meta = load("market/odds_meta.json", {})
    futures_meta = load("market/futures_meta.json", {})

    market_summary = market.get("summary", {})
    outliers = [g for g in market.get("games", []) if g.get("outlier")]
    futures_summary = futures.get("summary", {})

    sigma = {}
    if os.path.exists("models/score_models.joblib"):
        try:
            import joblib

            bundle = joblib.load("models/score_models.joblib")
            sigma = {
                "margin": bundle.get("sigma_margin"),
                "total": bundle.get("sigma_total"),
            }
        except Exception:
            sigma = {}

    injuries = {"seasons": [], "latest_week": None}
    for path in ("data/injuries_history.parquet", "data/injuries_latest.parquet"):
        if not os.path.exists(path):
            continue
        try:
            import polars as pl

            frame = pl.read_parquet(path)
            latest_season = int(frame["season"].max())
            latest_week = (
                int(frame.filter(pl.col("season") == latest_season)["week"].max())
                if frame.height
                else None
            )
            injuries = {
                "seasons": sorted(frame["season"].unique().to_list()),
                "latest_season": latest_season,
                "latest_week": latest_week,
                "rows": frame.height,
            }
            break
        except Exception:
            pass

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_backtest": backtest.get("pooled", {}),
        "calibration": {
            "enabled": platt.get("enabled"),
            "a": platt.get("a"),
            "b": platt.get("b"),
            "raw_log_loss": platt.get("raw_log_loss"),
            "calibrated_log_loss": platt.get("calibrated_log_loss"),
            "n": platt.get("n"),
        },
        "market": {
            "games": market_summary.get("games_compared"),
            "outliers": len(outliers),
            "outlier_games": [g.get("game_id") for g in outliers],
            "mean_abs_edge": market_summary.get("mean_abs_edge"),
            "max_abs_edge": market_summary.get("max_abs_edge"),
        },
        "futures": {
            "mean_abs_edge": futures_summary.get("mean_abs_edge"),
            "max_abs_edge": futures_summary.get("max_abs_edge"),
            "teams_matched": futures_summary.get("teams_matched"),
        },
        "calls": {
            "graded": calls.get("graded"),
            "open": calls.get("open"),
            "wins": calls.get("wins"),
            "losses": calls.get("losses"),
            "pushes": calls.get("pushes"),
            "win_rate": calls.get("win_rate"),
            "roi": calls.get("roi"),
            "profit": calls.get("profit"),
            "by_market": calls.get("by_market", {}),
            "by_edge": calls.get("by_edge", {}),
        },
        "score_models": sigma,
        "playoff_backtest": playoff.get("pooled", {}),
        "player_intervals": {
            group: {
                "coverage_80": metrics.get("coverage_80"),
                "mae": metrics.get("mae"),
            }
            for group, metrics in player_backtest.items()
            if metrics
        },
        "players": {
            "count": len(players),
            "with_ranges": sum(
                1 for player in players if player.get("floor_fantasy_points") is not None
            ),
            "active": sum(
                1 for player in players if player.get("roster_status") == "ACT"
            ),
        },
        "weather": {
            "games_with_forecast": sum(
                1 for game in upcoming if game.get("temp") is not None
            ),
            "games": len(upcoming),
        },
        "injuries": injuries,
        "odds": {
            "fetched_at": odds_meta.get("fetched_at"),
            "events": odds_meta.get("events"),
            "credits": odds_meta.get("credits"),
            "requests_used": odds_meta.get("requests_used"),
            "requests_remaining": odds_meta.get("requests_remaining"),
        },
        "futures_api": {
            "fetched_at": futures_meta.get("fetched_at"),
            "teams": futures_meta.get("teams"),
            "credits": futures_meta.get("credits"),
            "requests_used": futures_meta.get("requests_used"),
            "requests_remaining": futures_meta.get("requests_remaining"),
        },
        "line_movement": {
            "snapshots": len(movement.get("snapshots", [])),
            "games": movement.get("games"),
            "summary": movement.get("summary", {}),
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_health.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Model health: {len(outliers)} outliers, "
        f"calibration enabled={payload['calibration']['enabled']}, "
        f"injury weeks through {injuries.get('latest_week')}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Health report error: {exc}")
