#!/usr/bin/env python3
"""Analyze stored odds snapshots for closing-line movement.

Reads ``market/history/*.json`` and writes ``reports/market_backtest.json``.
With fewer than two snapshots it simply reports that history is accumulating.
"""

from __future__ import annotations

import glob
import json
import os
import statistics


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    files = sorted(glob.glob("market/history/*.json"))
    if len(files) < 2:
        payload = {
            "snapshots": [os.path.basename(f) for f in files],
            "games": 0,
            "summary": {"note": "Need at least two snapshots for line movement."},
        }
        os.makedirs("reports", exist_ok=True)
        with open("reports/market_backtest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Market backtest: insufficient history (need ≥2 snapshots).")
        return

    snapshots = [(os.path.basename(f), load(f)) for f in files]
    first_date, first_events = snapshots[0]
    last_date, last_events = snapshots[-1]

    def index(events):
        out = {}
        for e in events:
            key = (e.get("home_team"), e.get("away_team"), e.get("commence_time"))
            out[key] = e
        return out

    first = index(first_events)
    last = index(last_events)

    movements = []
    for key in sorted(set(first) & set(last)):
        a = first[key]
        b = last[key]
        row = {
            "home_team": key[0],
            "away_team": key[1],
            "commence_time": key[2],
            "first_spread": a.get("home_spread"),
            "last_spread": b.get("home_spread"),
            "first_total": a.get("total"),
            "last_total": b.get("total"),
            "first_home_ml": a.get("home_moneyline"),
            "last_home_ml": b.get("home_moneyline"),
        }
        if row["first_spread"] is not None and row["last_spread"] is not None:
            row["spread_move"] = round(row["last_spread"] - row["first_spread"], 2)
        if row["first_total"] is not None and row["last_total"] is not None:
            row["total_move"] = round(row["last_total"] - row["first_total"], 2)
        movements.append(row)

    spread_moves = [abs(m["spread_move"]) for m in movements if "spread_move" in m]
    total_moves = [abs(m["total_move"]) for m in movements if "total_move" in m]

    payload = {
        "snapshots": [d for d, _ in snapshots],
        "first": first_date,
        "last": last_date,
        "games": len(movements),
        "summary": {
            "mean_abs_spread_move": round(statistics.mean(spread_moves), 3) if spread_moves else None,
            "max_abs_spread_move": round(max(spread_moves), 3) if spread_moves else None,
            "mean_abs_total_move": round(statistics.mean(total_moves), 3) if total_moves else None,
            "max_abs_total_move": round(max(total_moves), 3) if total_moves else None,
        },
        "movements": movements,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/market_backtest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Market backtest: {len(movements)} games across "
        f"{len(snapshots)} snapshots; wrote reports/market_backtest.json"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Market backtest error: {exc}")
