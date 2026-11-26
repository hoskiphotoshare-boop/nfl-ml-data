#!/usr/bin/env python3
"""Maintain the model-call ledger and grade completed calls.

One primary call per game is snapshotted from the best available market edge
before kickoff. Closing prices/lines are refreshed while the call is open, and
the call is graded once final scores are available.

Outputs:
  predictions/calls_history.json
  reports/calls_scoreboard.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import polars as pl

MARKETS = "reports/market_comparison.json"
UPCOMING = "predictions/upcoming_games.json"
SCHEDULES = "data/schedules.parquet"
CALLS = "predictions/calls_history.json"
SCOREBOARD = "reports/calls_scoreboard.json"
MIN_EDGE = 0.02


def load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def call_id(game_id: str) -> str:
    return f"{game_id}_best"


def side_matches(call: dict, entry: dict) -> bool:
    if not entry:
        return False
    if call["market"] == "ml":
        return entry.get("side") == call["side"]
    if call["market"] == "total":
        return entry.get("side") == call["side"]
    return entry.get("side") == call["side"] and entry.get("team") == call.get("team")


def snapshot_calls(games: list[dict], existing: dict, now: datetime) -> int:
    added = 0
    for game in games:
        game_id = game.get("game_id")
        if not game_id or call_id(game_id) in existing:
            continue
        kickoff = parse_time(game.get("commence_time"))
        if kickoff and kickoff <= now:
            continue
        best_key = game.get("best_market")
        best_entry = (game.get("markets") or {}).get(best_key)
        if not best_key or not best_entry:
            continue
        edge = best_entry.get("edge")
        price = best_entry.get("price")
        if edge is None or edge < MIN_EDGE or not price:
            continue
        if best_entry.get("line") is None and best_key != "ml":
            continue
        entry = {
            "call_id": call_id(game_id),
            "game_id": game_id,
            "season": game.get("season"),
            "week": game.get("week"),
            "away_team": game.get("away_team"),
            "home_team": game.get("home_team"),
            "market": best_key,
            "side": best_entry.get("side"),
            "team": best_entry.get("team"),
            "line": best_entry.get("line"),
            "price": price,
            "bookmaker": best_entry.get("bookmaker"),
            "model_prob": best_entry.get("model_prob"),
            "market_prob": best_entry.get("market_prob"),
            "edge": edge,
            "kickoff": game.get("commence_time"),
            "snapshot_at": now.isoformat(),
            "closing_line": None,
            "closing_price": None,
            "closing_at": None,
            "status": "open",
            "result": None,
            "profit": None,
            "graded_at": None,
        }
        existing[entry["call_id"]] = entry
        added += 1
    return added


def update_closing(games: list[dict], existing: dict, now: datetime) -> int:
    updated = 0
    by_game = {game.get("game_id"): game for game in games}
    for call in existing.values():
        if call.get("status") != "open":
            continue
        kickoff = parse_time(call.get("kickoff"))
        if kickoff and kickoff <= now:
            continue
        game = by_game.get(call.get("game_id"))
        entry = ((game or {}).get("markets") or {}).get(call.get("market"))
        if not side_matches(call, entry):
            continue
        if entry.get("line") is not None:
            call["closing_line"] = entry.get("line")
        if entry.get("price") is not None:
            call["closing_price"] = entry.get("price")
            call["closing_at"] = now.isoformat()
            updated += 1
    return updated


def final_scores() -> dict[tuple, dict]:
    if not os.path.exists(SCHEDULES):
        return {}
    try:
        frame = pl.read_parquet(SCHEDULES)
    except Exception:
        return {}
    required = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(set(frame.columns)):
        return {}
    played = frame.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    return {
        (
            int(row["season"]),
            int(row["week"]),
            row["away_team"],
            row["home_team"],
        ): row
        for row in played.iter_rows(named=True)
    }


def grade_calls(existing: dict, now: datetime) -> int:
    scores = final_scores()
    graded = 0
    for call in existing.values():
        if call.get("status") != "open":
            continue
        key = (
            int(call.get("season") or 0),
            int(call.get("week") or 0),
            call.get("away_team"),
            call.get("home_team"),
        )
        row = scores.get(key)
        if not row:
            continue
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        total = home_score + away_score
        result = None
        if call["market"] == "ml":
            team_score = home_score if call["side"] == "home" else away_score
            opp_score = away_score if call["side"] == "home" else home_score
            result = "push" if team_score == opp_score else (
                "win" if team_score > opp_score else "loss"
            )
        elif call["market"] == "spread":
            team_score = home_score if call["side"] == "home" else away_score
            opp_score = away_score if call["side"] == "home" else home_score
            margin = team_score - opp_score + float(call.get("line") or 0)
            result = "push" if margin == 0 else ("win" if margin > 0 else "loss")
        elif call["market"] == "total":
            line = float(call.get("line") or 0)
            if total == line:
                result = "push"
            elif call["side"] == "over":
                result = "win" if total > line else "loss"
            else:
                result = "win" if total < line else "loss"
        if result is None:
            continue
        price = float(call.get("price") or 0)
        call["result"] = result
        call["profit"] = 0.0 if result == "push" else (
            round(price - 1.0, 4) if result == "win" else -1.0
        )
        call["status"] = "graded"
        call["graded_at"] = now.isoformat()
        graded += 1
    return graded


def summarize(calls: list[dict]) -> dict:
    graded = [call for call in calls if call.get("status") == "graded"]
    wins = sum(1 for call in graded if call["result"] == "win")
    losses = sum(1 for call in graded if call["result"] == "loss")
    pushes = sum(1 for call in graded if call["result"] == "push")
    profit = sum(float(call.get("profit") or 0.0) for call in graded)
    decided = wins + losses

    def bucket(edge):
        if edge is None:
            return "unknown"
        if edge >= 0.10:
            return "10%+"
        if edge >= 0.05:
            return "5–10%"
        return "2–5%"

    by_market = {}
    by_edge = {}
    for call in graded:
        for target, key in (
            (by_market, call.get("market")),
            (by_edge, bucket(call.get("edge"))),
        ):
            row = target.setdefault(key, {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "profit": 0.0})
            row["n"] += 1
            row["wins"] += 1 if call["result"] == "win" else 0
            row["losses"] += 1 if call["result"] == "loss" else 0
            row["pushes"] += 1 if call["result"] == "push" else 0
            row["profit"] += float(call.get("profit") or 0.0)

    for table in (by_market, by_edge):
        for row in table.values():
            row["roi"] = round(row["profit"] / row["n"], 4) if row["n"] else None
            decisions = row["wins"] + row["losses"]
            row["win_rate"] = round(row["wins"] / decisions, 4) if decisions else None
            row["profit"] = round(row["profit"], 4)

    recent = sorted(
        graded, key=lambda call: call.get("graded_at") or "", reverse=True
    )[:12]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calls": len(calls),
        "graded": len(graded),
        "open": sum(1 for call in calls if call.get("status") == "open"),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi": round(profit / len(graded), 4) if graded else None,
        "profit": round(profit, 4),
        "by_market": by_market,
        "by_edge": by_edge,
        "recent": recent,
    }


def main() -> None:
    games = load(MARKETS, {}).get("games", [])
    upcoming = load(UPCOMING, [])
    upcoming_by_id = {game.get("game_id"): game for game in upcoming}
    # Ensure the market report carries kickoff times where possible.
    for game in games:
        if not game.get("commence_time"):
            source = upcoming_by_id.get(game.get("game_id"))
            if source:
                game["commence_time"] = source.get("gameday")

    history = load(CALLS, [])
    if not isinstance(history, list):
        history = []
    existing = {call.get("call_id"): call for call in history}
    now = datetime.now(timezone.utc)

    added = snapshot_calls(games, existing, now)
    updated = update_closing(games, existing, now)
    graded = grade_calls(existing, now)

    calls = sorted(
        existing.values(),
        key=lambda call: (call.get("kickoff") or "", call.get("call_id") or ""),
    )
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open(CALLS, "w", encoding="utf-8") as handle:
        json.dump(calls, handle, indent=2)
    scoreboard = summarize(calls)
    with open(SCOREBOARD, "w", encoding="utf-8") as handle:
        json.dump(scoreboard, handle, indent=2)
    print(
        f"Calls: {scoreboard['calls']} total · {scoreboard['open']} open · "
        f"{scoreboard['graded']} graded · {added} new · {updated} closing updates · "
        f"{graded} newly graded"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Call ledger error: {exc}")
