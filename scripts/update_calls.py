#!/usr/bin/env python3
"""Maintain the model-call ledger and grade completed calls.

One primary call per game is snapshotted from the best available market edge
before kickoff. Closing prices/lines are refreshed while the call is open, and
the call is graded once final scores are available.

Also archives every priced game's three model picks (ml/spread/total) into
predictions/game_picks.json so grades stay durable after a game leaves the
upcoming list.

Outputs:
  predictions/calls_history.json
  predictions/game_picks.json
  reports/calls_scoreboard.json
"""

from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime, timezone

import polars as pl

MARKETS = "reports/market_comparison.json"
UPCOMING = "predictions/upcoming_games.json"
SCHEDULES = "data/schedules.parquet"
CALLS = "predictions/calls_history.json"
SCOREBOARD = "reports/calls_scoreboard.json"
PICKS = "predictions/game_picks.json"
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


def grade_market(market, side, line, home, away):
    """Return 'win' | 'loss' | 'push' | None for a market pick.

    Shared by the call ledger and the archived pick grading so rules cannot
    diverge. Moneyline ties are pushes; spread uses the line; total uses
    over/under; a missing line yields None.
    """
    if market == "ml":
        if home == away:
            return "push"
        picked_home = side == "home"
        home_won = home > away
        return "win" if picked_home == home_won else "loss"
    if line is None:
        return None
    try:
        line_value = float(line)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(line_value):
        return None
    if market == "spread":
        team_score = home if side == "home" else away
        opp_score = away if side == "home" else home
        margin = team_score - opp_score + line_value
        if margin == 0:
            return "push"
        return "win" if margin > 0 else "loss"
    if market == "total":
        total = home + away
        if total == line_value:
            return "push"
        over = total > line_value
        if side == "over":
            return "win" if over else "loss"
        return "win" if not over else "loss"
    return None


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


def grade_calls(existing: dict, now: datetime, scores=None) -> int:
    if scores is None:
        scores = final_scores()
    graded = 0
    for call in existing.values():
        if not isinstance(call, dict):
            continue
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
        result = grade_market(
            call.get("market"), call.get("side"), call.get("line"), home_score, away_score
        )
        if result is None:
            continue
        price = float(call.get("price") or 0)
        call["result"] = result
        call["profit"] = 0.0 if result == "push" else (
            round(price - 1.0, 4) if result == "win" else -1.0
        )
        call["status"] = "graded"
        call["graded_at"] = now.isoformat()
        call["final_home_score"] = home_score
        call["final_away_score"] = away_score
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


def _copy(value):
    return copy.deepcopy(value)


def _upcoming_source(upcoming_by_id, game_id):
    source = upcoming_by_id.get(game_id)
    return source if isinstance(source, dict) else None


def build_pick_entry(game, upcoming_by_id, now, odds_fetched_at):
    source = _upcoming_source(upcoming_by_id, game.get("game_id"))
    season = game.get("season")
    if source and source.get("season") is not None:
        season = source.get("season")
    week = game.get("week")
    if source and source.get("week") is not None:
        week = source.get("week")
    away_team = game.get("away_team")
    if source and source.get("away_team"):
        away_team = source.get("away_team")
    home_team = game.get("home_team")
    if source and source.get("home_team"):
        home_team = source.get("home_team")
    commence = game.get("commence_time")
    if source and source.get("commence_time"):
        commence = source.get("commence_time")

    if source:
        gameday = source.get("gameday")
        gametime = source.get("gametime")
    else:
        gameday = None
        gametime = None
    if gameday is None and commence:
        gameday = str(commence)[:10]
    if gametime is None and commence and len(str(commence)) >= 16:
        gametime = str(commence)[11:16]

    if source and source.get("home_win_prob") is not None:
        home_win_prob = source.get("home_win_prob")
    elif game.get("model_home_win_prob") is not None:
        home_win_prob = game.get("model_home_win_prob")
    else:
        home_win_prob = None

    if source and source.get("spread") is not None:
        spread = source.get("spread")
    elif game.get("model_spread") is not None:
        spread = game.get("model_spread")
    else:
        spread = None

    if source and source.get("total") is not None:
        total = source.get("total")
    elif game.get("model_total") is not None:
        total = game.get("model_total")
    else:
        total = None

    def _src(key):
        return source.get(key) if source else None

    entry = {
        "game_id": game.get("game_id"),
        "season": season,
        "week": week,
        "away_team": away_team,
        "home_team": home_team,
        "gameday": gameday,
        "gametime": gametime,
        "kickoff": commence,
        "home_win_prob": home_win_prob,
        "spread": spread,
        "total": total,
        "home_record": _src("home_record"),
        "away_record": _src("away_record"),
        "home_form": _src("home_form"),
        "away_form": _src("away_form"),
        "stadium": _src("stadium"),
        "roof": _src("roof"),
        "surface": _src("surface"),
        "location": _src("location"),
        "div_game": _src("div_game"),
        "temp": _src("temp"),
        "wind": _src("wind"),
        "markets": _copy(game.get("markets") or {}),
        "source_fetched_at": odds_fetched_at,
        "snapshot_at": now.isoformat(),
        "final": None,
        "grades": None,
        "graded_at": None,
    }
    return entry


def main() -> None:
    market_doc = load(MARKETS, {})
    if not isinstance(market_doc, dict):
        market_doc = {}
    games = market_doc.get("games", [])
    if not isinstance(games, list):
        games = []
    upcoming = load(UPCOMING, [])
    if not isinstance(upcoming, list):
        upcoming = []
    upcoming_by_id = {g.get("game_id"): g for g in upcoming if isinstance(g, dict)}

    # Ensure the market report carries kickoff times where possible.
    for game in games:
        if isinstance(game, dict) and not game.get("commence_time"):
            source = upcoming_by_id.get(game.get("game_id"))
            if source:
                game["commence_time"] = source.get("gameday")

    history = load(CALLS, [])
    if not isinstance(history, list):
        history = []
    existing_calls = {call.get("call_id"): call for call in history if isinstance(call, dict)}
    now = datetime.now(timezone.utc)

    added_calls = snapshot_calls(games, existing_calls, now)
    updated_calls = update_closing(games, existing_calls, now)
    scores = final_scores()
    graded_calls = grade_calls(existing_calls, now, scores)

    # Archive picks.
    picks_doc = load(PICKS, {"version": 1, "generated_at": None, "games": []})
    if not isinstance(picks_doc, dict):
        picks_doc = {"version": 1, "generated_at": None, "games": []}
    if not isinstance(picks_doc.get("games"), list):
        picks_doc["games"] = []
    existing_picks = {}
    for g in picks_doc["games"]:
        if isinstance(g, dict) and g.get("game_id"):
            existing_picks[g["game_id"]] = g

    new_archived = 0
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = game.get("game_id")
        if not game_id:
            continue
        markets = game.get("markets")
        if not isinstance(markets, dict) or not markets:
            continue

        pick = existing_picks.get(game_id)
        if pick is None:
            pick = build_pick_entry(game, upcoming_by_id, now, market_doc.get("odds_fetched_at"))
            existing_picks[game_id] = pick
            new_archived += 1
        else:
            if not isinstance(pick.get("markets"), dict):
                pick["markets"] = {}
            for key, entry in markets.items():
                if key not in pick["markets"]:
                    pick["markets"][key] = _copy(entry)

        season = pick.get("season")
        week = pick.get("week")
        away_team = pick.get("away_team")
        home_team = pick.get("home_team")

        if pick.get("final") is None:
            key = (int(season or 0), int(week or 0), away_team, home_team)
            row = scores.get(key)
            if row:
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                pick["final"] = {"home": home_score, "away": away_score}
                grades = {}
                for mkey, entry in (pick.get("markets") or {}).items():
                    grades[mkey] = grade_market(
                        mkey,
                        (entry or {}).get("side"),
                        (entry or {}).get("line"),
                        home_score,
                        away_score,
                    )
                pick["grades"] = grades
                pick["graded_at"] = now.isoformat()
        else:
            if not isinstance(pick.get("grades"), dict):
                final = pick.get("final") or {}
                home_score = int(final.get("home"))
                away_score = int(final.get("away"))
                grades = {}
                for mkey, entry in (pick.get("markets") or {}).items():
                    grades[mkey] = grade_market(
                        mkey,
                        (entry or {}).get("side"),
                        (entry or {}).get("line"),
                        home_score,
                        away_score,
                    )
                pick["grades"] = grades
                if not pick.get("graded_at"):
                    pick["graded_at"] = now.isoformat()

    all_picks = list(existing_picks.values())
    seasons = [
        int(p["season"])
        for p in all_picks
        if isinstance(p.get("season"), (int, float))
    ]
    for game in games:
        if isinstance(game, dict) and isinstance(game.get("season"), (int, float)):
            seasons.append(int(game["season"]))
    latest_season = max(seasons, default=datetime.now(timezone.utc).year)
    allowed_seasons = {latest_season, latest_season - 1}
    filtered = [p for p in all_picks if p.get("season") in allowed_seasons]
    filtered.sort(key=lambda p: (p.get("kickoff") or "", p.get("game_id") or ""))

    picks_doc["version"] = 1
    picks_doc["generated_at"] = now.isoformat()
    picks_doc["games"] = filtered

    os.makedirs("predictions", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open(PICKS, "w", encoding="utf-8") as handle:
        json.dump(picks_doc, handle, indent=2)

    calls = sorted(
        existing_calls.values(),
        key=lambda call: (call.get("kickoff") or "", call.get("call_id") or ""),
    )
    with open(CALLS, "w", encoding="utf-8") as handle:
        json.dump(calls, handle, indent=2)
    scoreboard = summarize(calls)
    with open(SCOREBOARD, "w", encoding="utf-8") as handle:
        json.dump(scoreboard, handle, indent=2)
    graded_total = sum(
        1 for p in filtered if p.get("final") and isinstance(p.get("grades"), dict)
    )
    print(
        f"Calls: {scoreboard['calls']} total · {scoreboard['open']} open · "
        f"{scoreboard['graded']} graded · {added_calls} new · {updated_calls} closing updates · "
        f"{graded_calls} newly graded"
    )
    print(f"Picks: {len(filtered)} archived · {graded_total} graded · {new_archived} new")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Call ledger error: {exc}")
