#!/usr/bin/env python3
"""Summarise opening-to-latest line movement from stored snapshots."""

from __future__ import annotations

import glob
import json
import os
import statistics

from compare_market import FALLBACK_TEAM_NAMES

OUTPUT = "reports/line_movement.json"


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    files = sorted(glob.glob("market/history/*.json"))
    name_to_abbr = {name: abbr for abbr, name in FALLBACK_TEAM_NAMES.items()}
    series: dict[tuple, list[dict]] = {}

    for path in files:
        for event in load(path):
            key = (event.get("home_team"), event.get("away_team"), event.get("commence_time"))
            series.setdefault(key, []).append(event)

    games = []
    for key, points in series.items():
        if len(points) < 2:
            continue
        first, last = points[0], points[-1]
        home_abbr = name_to_abbr.get(key[0], key[0])
        away_abbr = name_to_abbr.get(key[1], key[1])
        row = {
            "home_team": home_abbr,
            "away_team": away_abbr,
            "commence_time": key[2],
            "snapshots": len(points),
            "first_spread": first.get("home_spread"),
            "latest_spread": last.get("home_spread"),
            "first_total": first.get("total"),
            "latest_total": last.get("total"),
            "first_home_ml": first.get("home_moneyline"),
            "latest_home_ml": last.get("home_moneyline"),
        }
        if row["first_spread"] is not None and row["latest_spread"] is not None:
            row["spread_move"] = round(row["latest_spread"] - row["first_spread"], 2)
        if row["first_total"] is not None and row["latest_total"] is not None:
            row["total_move"] = round(row["latest_total"] - row["first_total"], 2)
        if row["first_home_ml"] is not None and row["latest_home_ml"] is not None:
            row["home_ml_move"] = round(row["latest_home_ml"] - row["first_home_ml"], 3)
        games.append(row)

    spread_moves = [abs(g["spread_move"]) for g in games if "spread_move" in g]
    total_moves = [abs(g["total_move"]) for g in games if "total_move" in g]
    payload = {
        "snapshots": [os.path.basename(f) for f in files],
        "games": len(games),
        "summary": {
            "games_with_movement": len(games),
            "mean_abs_spread_move": round(statistics.mean(spread_moves), 3) if spread_moves else None,
            "max_abs_spread_move": round(max(spread_moves), 3) if spread_moves else None,
            "mean_abs_total_move": round(statistics.mean(total_moves), 3) if total_moves else None,
            "max_abs_total_move": round(max(total_moves), 3) if total_moves else None,
        },
        "games_detail": games,
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Line movement: {len(games)} games with ≥2 snapshots; "
        f"mean |spread| {payload['summary']['mean_abs_spread_move']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Line movement error: {exc}")
