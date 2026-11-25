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
from datetime import datetime

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
    # Index odds by (home_full, away_full) -> list of events.
    odds_index = {}
    for event in odds:
        key = (event.get("home_team"), event.get("away_team"))
        odds_index.setdefault(key, []).append(event)

    def event_date(event):
        try:
            return datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            ).date()
        except Exception:
            return None

    def game_date(game):
        try:
            if game.get("gameday"):
                return datetime.strptime(game["gameday"], "%Y-%m-%d").date()
        except Exception:
            pass
        return None

    def choose_event(candidates, game):
        if not candidates:
            return None
        gd = game_date(game)
        if gd is None:
            return candidates[0]
        best = None
        best_diff = None
        for ev in candidates:
            ed = event_date(ev)
            if ed is None:
                continue
            diff = abs((ed - gd).days)
            if best_diff is None or diff < best_diff:
                best = ev
                best_diff = diff
        if best is None:
            return candidates[0]
        if best_diff is not None and best_diff > 2:
            return None
        return best

    comparisons = []
    for game in predictions:
        home_abbr = game.get("home_team")
        away_abbr = game.get("away_team")
        home_full = names.get(home_abbr, home_abbr)
        away_full = names.get(away_abbr, away_abbr)

        candidates = list(odds_index.get((home_full, away_full), [])) + list(
            odds_index.get((away_full, home_full), [])
        )
        event = choose_event(candidates, game)
        if event is None:
            continue

        model_prob = game.get("home_win_prob")
        market_prob = event.get("market_home_win_prob")
        if model_prob is None or market_prob is None:
            continue

        reversed_event = (
            event.get("home_team") == away_full
            and event.get("away_team") == home_full
        )
        if reversed_event:
            market_prob = 1.0 - market_prob
            market_spread = event.get("away_spread")
            market_total = event.get("total")
        else:
            market_spread = event.get("home_spread")
            market_total = event.get("total")

        best_home_ml = event.get("best_home_ml")
        best_away_ml = event.get("best_away_ml")
        best_home_spread = event.get("best_home_spread")
        best_away_spread = event.get("best_away_spread")
        if reversed_event:
            best_home_ml, best_away_ml = best_away_ml, best_home_ml
            best_home_spread, best_away_spread = best_away_spread, best_home_spread

        edge = float(model_prob) - float(market_prob)

        ev_home = None
        ev_away = None
        if best_home_ml and best_home_ml.get("price"):
            ev_home = (float(model_prob) * float(best_home_ml["price"]) - 1.0) * 100.0
        if best_away_ml and best_away_ml.get("price"):
            ev_away = (
                (1.0 - float(model_prob)) * float(best_away_ml["price"]) - 1.0
            ) * 100.0
        ev_values = [v for v in (ev_home, ev_away) if v is not None]
        best_ev = max(ev_values) if ev_values else None
        best_ev_side = None
        if best_ev is not None:
            best_ev_side = "home" if ev_home == best_ev else "away"

        spread_edge = None
        if game.get("spread") is not None and market_spread is not None:
            spread_edge = float(market_spread) - float(game["spread"])

        total_edge = None
        if game.get("total") is not None and market_total is not None:
            total_edge = float(game["total"]) - float(market_total)

        comparisons.append(
            {
                "game_id": game.get("game_id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "commence_time": event.get("commence_time"),
                "home_team": home_abbr,
                "away_team": away_abbr,
                "model_home_win_prob": round(float(model_prob), 4),
                "market_home_win_prob": round(float(market_prob), 4),
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "ev_home_ml": round(ev_home, 3) if ev_home is not None else None,
                "ev_away_ml": round(ev_away, 3) if ev_away is not None else None,
                "best_ev": round(best_ev, 3) if best_ev is not None else None,
                "best_ev_side": best_ev_side,
                "model_spread": game.get("spread"),
                "market_spread": market_spread,
                "spread_edge": round(spread_edge, 2) if spread_edge is not None else None,
                "model_total": game.get("total"),
                "market_total": market_total,
                "total_edge": round(total_edge, 2) if total_edge is not None else None,
                "best_home_ml": best_home_ml,
                "best_away_ml": best_away_ml,
                "best_home_spread": best_home_spread,
                "best_away_spread": best_away_spread,
                "best_over": event.get("best_over"),
                "best_under": event.get("best_under"),
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

    odds_meta = {}
    if os.path.exists("market/odds_meta.json"):
        try:
            with open("market/odds_meta.json", encoding="utf-8") as f:
                odds_meta = json.load(f)
        except Exception:
            odds_meta = {}

    report = {
        "summary": summary,
        "odds_fetched_at": odds_meta.get("fetched_at"),
        "odds_events": odds_meta.get("events"),
        "odds_requests_remaining": odds_meta.get("requests_remaining"),
        "games": comparisons,
    }
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