#!/usr/bin/env python3
"""Compare model predictions with market odds.

Reads ``predictions/upcoming_games.json`` (falling back to
``predictions/games_week.json``) and ``market/odds_latest.json`` and writes
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

# Normal approximation used to turn line differences into cover probabilities.
# NFL margins have a standard deviation of roughly two touchdowns.
SIGMA = 13.5
SPREAD_SIGMA = 13.5
TOTAL_SIGMA = 13.5
try:
    import joblib

    _score_bundle = joblib.load("models/score_models.joblib")
    SPREAD_SIGMA = float(_score_bundle.get("sigma_margin", 13.5))
    TOTAL_SIGMA = float(_score_bundle.get("sigma_total", 13.5))
except Exception:
    pass


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# The Odds API returns full team names. Keep a static fallback so the
# comparison works in a bare Python environment (e.g. the odds workflow).
FALLBACK_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def team_name_map() -> dict[str, str]:
    names = dict(FALLBACK_TEAM_NAMES)
    if not TEAMS_AVAILABLE:
        return names
    try:
        teams = load_teams()
        for row in teams.iter_rows(named=True):
            abbr = row.get("team_abbr")
            name = row.get("team_name")
            if abbr and name:
                names[abbr] = name
    except Exception as exc:  # pragma: no cover
        print(f"Could not load team names from nflreadpy, using fallback: {exc}")
    return names


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

        # ---- per-market model edges -------------------------------------
        # Every edge is expressed as a probability advantage so ML, spread
        # and total are directly comparable.
        model_spread = float(game["spread"]) if game.get("spread") is not None else None
        model_total = float(game["total"]) if game.get("total") is not None else None
        markets: dict[str, dict] = {}

        if float(model_prob) >= float(market_prob):
            ml_side, ml_team = "home", home_abbr
            ml_model, ml_market = float(model_prob), float(market_prob)
            ml_price = (best_home_ml or {}).get("price")
            ml_book = (best_home_ml or {}).get("bookmaker")
        else:
            ml_side, ml_team = "away", away_abbr
            ml_model, ml_market = 1.0 - float(model_prob), 1.0 - float(market_prob)
            ml_price = (best_away_ml or {}).get("price")
            ml_book = (best_away_ml or {}).get("bookmaker")
        markets["ml"] = {
            "side": ml_side,
            "team": ml_team,
            "line": None,
            "model_prob": round(ml_model, 4),
            "market_prob": round(ml_market, 4),
            "edge": round(ml_model - ml_market, 4),
            "price": ml_price,
            "bookmaker": ml_book,
        }

        if model_spread is not None:
            spread_candidates = []
            if best_home_spread and best_home_spread.get("point") is not None:
                line = float(best_home_spread["point"])
                cover = normal_cdf((model_spread + line) / SPREAD_SIGMA)
                spread_candidates.append(
                    {
                        "side": "home",
                        "team": home_abbr,
                        "line": line,
                        "model_prob": round(cover, 4),
                        "market_prob": 0.5,
                        "edge": round(cover - 0.5, 4),
                        "price": best_home_spread.get("price"),
                        "bookmaker": best_home_spread.get("bookmaker"),
                    }
                )
            if best_away_spread and best_away_spread.get("point") is not None:
                line = float(best_away_spread["point"])
                cover = normal_cdf((-model_spread + line) / SPREAD_SIGMA)
                spread_candidates.append(
                    {
                        "side": "away",
                        "team": away_abbr,
                        "line": line,
                        "model_prob": round(cover, 4),
                        "market_prob": 0.5,
                        "edge": round(cover - 0.5, 4),
                        "price": best_away_spread.get("price"),
                        "bookmaker": best_away_spread.get("bookmaker"),
                    }
                )
            if spread_candidates:
                markets["spread"] = max(spread_candidates, key=lambda m: m["edge"])

        if model_total is not None:
            total_candidates = []
            best_over = event.get("best_over")
            best_under = event.get("best_under")
            if best_over and best_over.get("point") is not None:
                line = float(best_over["point"])
                over = normal_cdf((model_total - line) / TOTAL_SIGMA)
                total_candidates.append(
                    {
                        "side": "over",
                        "team": "Over",
                        "line": line,
                        "model_prob": round(over, 4),
                        "market_prob": 0.5,
                        "edge": round(over - 0.5, 4),
                        "price": best_over.get("price"),
                        "bookmaker": best_over.get("bookmaker"),
                    }
                )
            if best_under and best_under.get("point") is not None:
                line = float(best_under["point"])
                under = normal_cdf((line - model_total) / TOTAL_SIGMA)
                total_candidates.append(
                    {
                        "side": "under",
                        "team": "Under",
                        "line": line,
                        "model_prob": round(under, 4),
                        "market_prob": 0.5,
                        "edge": round(under - 0.5, 4),
                        "price": best_under.get("price"),
                        "bookmaker": best_under.get("bookmaker"),
                    }
                )
            if total_candidates:
                markets["total"] = max(total_candidates, key=lambda m: m["edge"])

        for market in markets.values():
            price = market.get("price")
            market["ev"] = (
                round((float(market["model_prob"]) * float(price) - 1.0) * 100.0, 2)
                if price
                else None
            )

        best_market = None
        if markets:
            best_market = max(markets, key=lambda key: markets[key].get("edge", -1))
        best_edge = markets[best_market]["edge"] if best_market else None
        market_evs = [m["ev"] for m in markets.values() if m.get("ev") is not None]
        best_market_ev = max(market_evs) if market_evs else None

        # Flag any market where the model and the price disagree by 20+ points,
        # so large spread/total edges are not presented as clean picks either.
        market_edges = [
            (key, abs(m.get("edge", 0)))
            for key, m in markets.items()
            if m.get("edge") is not None
        ]
        outlier_market = max(market_edges, key=lambda item: item[1])[0] if market_edges else None
        outlier_edge = max((value for _, value in market_edges), default=0.0)
        outlier = bool(market_edges) and outlier_edge >= 0.20

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
                "best_market_ev": round(best_market_ev, 2) if best_market_ev is not None else None,
                "markets": markets,
                "best_market": best_market,
                "best_edge": best_edge,
                "outlier": outlier,
                "outlier_edge": round(outlier_edge, 4),
                "outlier_market": outlier_market,
                "best_team": markets[best_market]["team"] if best_market else None,
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
    best_edges = [c["best_edge"] for c in comparisons if c.get("best_edge") is not None]
    summary = {
        "games_compared": len(comparisons),
        "mean_edge": round(statistics.mean(edges), 4),
        "mean_abs_edge": round(statistics.mean(abs_edges), 4),
        "max_abs_edge": round(max(abs_edges), 4),
        "mean_best_edge": round(statistics.mean(best_edges), 4) if best_edges else None,
        "max_best_edge": round(max(best_edges), 4) if best_edges else None,
        "strong_edges": sum(1 for e in best_edges if e >= 0.05),
        "best_market_counts": {
            key: sum(1 for c in comparisons if c.get("best_market") == key)
            for key in ("ml", "spread", "total")
        },
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