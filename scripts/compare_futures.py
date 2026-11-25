#!/usr/bin/env python3
"""Compare model playoff odds with Super Bowl futures markets.

Reads ``predictions/playoffs.json`` and ``market/futures_latest.json`` and
writes ``reports/futures_comparison.json`` with per-team market probability,
best price and the model edge in probability points.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))

from compare_market import FALLBACK_TEAM_NAMES  # noqa: E402


def load(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    playoffs = load("predictions/playoffs.json", [])
    futures = load("market/futures_latest.json", {})
    if not playoffs:
        print("No playoff predictions; skipping futures comparison.")
        return
    if not futures or not futures.get("teams"):
        print("No futures market data; skipping futures comparison.")
        return

    by_abbr = {abbr: name for abbr, name in FALLBACK_TEAM_NAMES.items()}
    market_by_abbr = {}
    for row in futures["teams"]:
        name = row.get("team")
        abbr = next((key for key, full in by_abbr.items() if full == name), None)
        if abbr:
            market_by_abbr[abbr] = row

    teams = []
    for team in playoffs:
        abbr = team.get("team")
        model_prob = team.get("super_bowl_prob")
        market = market_by_abbr.get(abbr, {})
        market_prob = market.get("market_prob")
        edge = None
        if model_prob is not None and market_prob is not None:
            edge = round(float(model_prob) - float(market_prob), 4)
        teams.append(
            {
                "team": abbr,
                "conference": team.get("conference"),
                "model_sb_prob": model_prob,
                "market_sb_prob": market_prob,
                "edge": edge,
                "best_price": market.get("best_price"),
                "best_bookmaker": market.get("best_bookmaker"),
                "median_price": market.get("median_price"),
                "books": market.get("books"),
            }
        )

    matched = [t for t in teams if t["market_sb_prob"] is not None]
    edges = [abs(t["edge"]) for t in matched if t["edge"] is not None]
    best = max(matched, key=lambda t: t["edge"] or -9, default=None)
    market_fav = max(matched, key=lambda t: t["market_sb_prob"] or 0, default=None)

    report = {
        "updated_at": futures.get("fetched_at"),
        "sport": futures.get("sport"),
        "market_teams": len(market_by_abbr),
        "summary": {
            "teams_matched": len(matched),
            "mean_abs_edge": round(statistics.mean(edges), 4) if edges else None,
            "max_abs_edge": round(max(edges), 4) if edges else None,
            "best_edge_team": best.get("team") if best else None,
            "best_edge": best.get("edge") if best else None,
            "market_favourite": market_fav.get("team") if market_fav else None,
            "market_favourite_prob": market_fav.get("market_sb_prob") if market_fav else None,
            "overround": futures.get("overround"),
        },
        "teams": teams,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/futures_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = report["summary"]
    print(
        f"Futures comparison: {summary['teams_matched']} teams, "
        f"mean abs edge {summary['mean_abs_edge']}, "
        f"best {summary['best_edge_team']} {summary['best_edge']}"
    )
    print("Wrote reports/futures_comparison.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Futures comparison error: {exc}")
    sys.exit(0)
