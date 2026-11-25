#!/usr/bin/env python3
"""Attach this week's opponent/matchup to each player projection.

Reads ``predictions/upcoming_games.json`` (written by ``predict_upcoming.py``)
and ``predictions/players_week.json`` and adds ``opponent``, ``matchup``,
``home_away``, ``gameday`` and ``gametime`` so the frontend can show a
fantasy-style matchup column.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def load(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    games = load("predictions/upcoming_games.json", [])
    players = load("predictions/players_week.json", [])
    if not games or not players:
        print("Missing games or player projections; skipping matchup enrichment.")
        return

    weeks = sorted({g.get("week") for g in games if g.get("week") is not None})
    current_week = weeks[0] if weeks else None
    slate = [g for g in games if g.get("week") == current_week]

    by_team: dict[str, dict] = {}
    for g in slate:
        home, away = g.get("home_team"), g.get("away_team")
        if not home or not away:
            continue
        by_team[home] = {
            "opponent": away,
            "home_away": "home",
            "matchup": f"vs {away}",
            "game_id": g.get("game_id"),
            "gameday": g.get("gameday"),
            "gametime": g.get("gametime"),
            "stadium": g.get("stadium"),
            "home_win_prob": g.get("home_win_prob"),
        }
        by_team[away] = {
            "opponent": home,
            "home_away": "away",
            "matchup": f"@ {home}",
            "game_id": g.get("game_id"),
            "gameday": g.get("gameday"),
            "gametime": g.get("gametime"),
            "stadium": g.get("stadium"),
            "home_win_prob": g.get("home_win_prob"),
        }

    enriched = 0
    for p in players:
        info = by_team.get(p.get("team"))
        if not info:
            p["opponent"] = None
            p["matchup"] = None
            continue
        p["opponent"] = info["opponent"]
        p["matchup"] = info["matchup"]
        p["home_away"] = info["home_away"]
        p["game_id"] = info["game_id"]
        p["gameday"] = info["gameday"]
        p["gametime"] = info["gametime"]
        enriched += 1

    with open("predictions/players_week.json", "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    with open("predictions/meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "games": len(games),
                "season": games[0].get("season"),
                "week": current_week,
                "players": len(players),
            },
            f,
            indent=2,
        )

    print(
        f"Added matchups to {enriched}/{len(players)} players "
        f"(week {current_week})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Matchup enrichment error: {exc}")
