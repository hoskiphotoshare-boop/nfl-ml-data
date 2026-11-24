#!/usr/bin/env python3
"""Compare model predictions with market odds.

Reads ``predictions/games_week.json`` and ``market/odds_latest.json`` and writes
``reports/market_comparison.json`` with per-game edges and summary statistics.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys

try:
    from nflreadpy import load_teams
    TEAMS_AVAILABLE = True
except Exception:  # pragma: no cover
    TEAMS_AVAILABLE = False


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def team_name_map() -> dict[str, str]:
    if not TEAMS_AVAILABLE:
        return {}
    try:
        teams = load_teams()
        return {row["team_abbr"]: row["team_name"] for row in teams.iter_rows(named=True)}
    except Exception as exc:  # pragma: no cover
        print(f"Could not load team names: {exc}")
        return {}


def main() -> None:
    predictions = load_json("predictions/upcoming_games.json") or load_json(
        "predictions/games_week.json"
    )
    odds = load_json("market/odds_latest.json")
    if predictions is None:
        print("No predictions found; skipping market comparison.")
        return
    if not odds:
        print("No odds found; skipping market comparison.")
        return

    names = team_name_map()
    # Index odds by (home_full, away_full)
    odds_index = {}
    for event in odds:
        odds_index[(event.get("home_team"), event.get("away_team"))] = event

    comparisons = []
    for game in predictions:
        home_abbr = game.get("home_team")
        away_abbr = game.get("away_team")
        home_full = names.get(home_abbr, home_abbr)
        away_full = names.get(away_abbr, away_abbr)

        event = odds_index.get((home_full, away_full))
        if event is None:
            event = odds_index.get((away_full, home_full))
        if event is None:
            continue

        model_prob = game.get("home_win_prob")
        market_prob = event.get("market_home_win_prob")
        if model_prob is None or market_prob is None:
            continue

        # If the odds event is reversed relative to our prediction, flip.
        if event.get("home_team") == away_full and event.get("away_team") == home_full:
            market_prob = 1.0 - market_prob
            market_spread = event.get("away_spread")
            market_total = event.get("total")
        else:
            market_spread = event.get("home_spread")
            market_total = event.get("total")

        edge = float(model_prob) - float(market_prob)
        comparisons.append(
            {
                "game_id": game.get("game_id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "home_team": home_abbr,
                "away_team": away_abbr,
                "model_home_win_prob": round(float(model_prob), 4),
                "market_home_win_prob": round(float(market_prob), 4),
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "model_spread": game.get("spread"),
                "market_spread": market_spread,
                "model_total": game.get("total"),
                "market_total": market_total,
            }
        )

    if not comparisons:
        print("No matching games between model and market.")
        return

    edges = [c["edge"] for c in comparisons]
    abs_edges = [c["abs_edge"] for c in comparisons]
    summary = {
        "games_compared": len(comparisons),
        "mean_edge": round(statistics.mean(edges), 4),
        "mean_abs_edge": round(statistics.mean(abs_edges), 4),
        "max_abs_edge": round(max(abs_edges), 4),
    }

    report = {"summary": summary, "games": comparisons}
    os.makedirs("reports", exist_ok=True)
    with open("reports/market_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"Market comparison: {summary['games_compared']} games, "
        f"mean edge {summary['mean_edge']:+.3f}, "
        f"mean abs edge {summary['mean_abs_edge']:.3f}, "
        f"max abs edge {summary['max_abs_edge']:.3f}"
    )
    print("Wrote reports/market_comparison.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Market comparison error: {exc}")
    sys.exit(0)