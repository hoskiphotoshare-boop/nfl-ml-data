#!/usr/bin/env python3
"""Summarise line movement from stored odds snapshots.

Reads every ``market/history/*.json`` snapshot (daily, pre-game and
timestamped runs), builds an opening-to-latest series per game and writes
``reports/market_backtest.json``.

Unlike a simple first-vs-last file comparison, this keeps a game’s own
timeline, so new snapshots of the next slate do not erase movement measured
for games that have already kicked off.
"""

from __future__ import annotations

import glob
import json
import os
import statistics

HISTORY_GLOB = "market/history/*.json"


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def movement(first, last, key):
    a, b = num(first.get(key)), num(last.get(key))
    if a is None or b is None:
        return None
    return round(b - a, 2)


def main() -> None:
    files = sorted(glob.glob(HISTORY_GLOB))
    payload = {
        "snapshots": [os.path.basename(f) for f in files],
        "games": 0,
        "summary": {},
        "movements": [],
    }

    if not files:
        payload["summary"] = {"note": "No odds history yet; the next fetch starts it."}
        write(payload)
        return

    # game key -> ordered list of (snapshot name, event)
    series: dict[tuple, list[tuple[str, dict]]] = {}
    for path in files:
        name = os.path.basename(path)
        for event in load(path):
            key = (
                event.get("home_team"),
                event.get("away_team"),
                event.get("commence_time"),
            )
            series.setdefault(key, []).append((name, event))

    movements = []
    for key, points in series.items():
        first_name, first = points[0]
        last_name, last = points[-1]
        row = {
            "home_team": key[0],
            "away_team": key[1],
            "commence_time": key[2],
            "snapshots": len(points),
            "first_snapshot": first_name,
            "last_snapshot": last_name,
            "opening_spread": num(first.get("home_spread")),
            "latest_spread": num(last.get("home_spread")),
            "opening_total": num(first.get("total")),
            "latest_total": num(last.get("total")),
            "opening_home_ml": num(first.get("home_moneyline")),
            "latest_home_ml": num(last.get("home_moneyline")),
        }
        row["spread_move"] = movement(first, last, "home_spread")
        row["total_move"] = movement(first, last, "total")
        row["home_ml_move"] = movement(first, last, "home_moneyline")
        movements.append(row)

    movements.sort(
        key=lambda r: abs(r["spread_move"]) if r["spread_move"] is not None else -1,
        reverse=True,
    )

    spread_moves = [abs(m["spread_move"]) for m in movements if m["spread_move"] is not None]
    total_moves = [abs(m["total_move"]) for m in movements if m["total_move"] is not None]
    live_games = [m for m in movements if m["snapshots"] >= 2]

    payload.update(
        {
            "games": len(movements),
            "games_with_history": len(live_games),
            "summary": {
                "games_with_history": len(live_games),
                "mean_abs_spread_move": round(statistics.mean(spread_moves), 3)
                if spread_moves
                else None,
                "max_abs_spread_move": round(max(spread_moves), 3) if spread_moves else None,
                "mean_abs_total_move": round(statistics.mean(total_moves), 3)
                if total_moves
                else None,
                "max_abs_total_move": round(max(total_moves), 3) if total_moves else None,
            },
            "movements": movements,
        }
    )
    write(payload)


def write(payload: dict) -> None:
    os.makedirs("reports", exist_ok=True)
    with open("reports/market_backtest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    summary = payload.get("summary", {})
    print(
        f"Market backtest: {payload.get('games', 0)} games across "
        f"{len(payload.get('snapshots', []))} snapshots; wrote reports/market_backtest.json"
    )
    if summary.get("mean_abs_spread_move") is not None:
        print(
            f"  mean |spread move| {summary['mean_abs_spread_move']}, "
            f"mean |total move| {summary.get('mean_abs_total_move')}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Market backtest error: {exc}")
