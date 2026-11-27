#!/usr/bin/env python3
"""Build per-game QB and injury context for the Games cards.

Reads the current player projections and the latest weekly injury report and
writes reports/game_context.json. Only the current week has projections, so
future-week cards simply have no context block.

Output: reports/game_context.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import polars as pl

PLAYERS = "predictions/players_week.json"
UPCOMING = "predictions/upcoming_games.json"
META = "predictions/meta.json"
INJURIES = "data/injuries_latest.parquet"
OUT = "reports/game_context.json"

SEVERITY = {
    "out": 0,
    "doubtful": 1,
    "questionable": 2,
    "dnp": 3,
    "limited": 4,
    "full": 5,
}


def load(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def clean(value) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def status_label(report, practice) -> tuple[str, str, int] | None:
    report = clean(report)
    practice = clean(practice)
    report_l = report.lower()
    practice_l = practice.lower()
    if report_l in ("out", "doubtful", "questionable"):
        return report, report, SEVERITY[report_l]
    if "did not participate" in practice_l:
        return "DNP", "DNP", SEVERITY["dnp"]
    if "limited participation" in practice_l:
        return "Limited", "Limited", SEVERITY["limited"]
    return None


def injury_rows(season: int, week: int) -> dict[str, list[dict]]:
    if not os.path.exists(INJURIES):
        return {}
    frame = pl.read_parquet(INJURIES).filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    if frame.height == 0:
        return {}
    # One row per player, latest first.
    frame = frame.sort("date_modified", nulls_last=True).group_by(
        ["gsis_id"], maintain_order=True
    ).agg(pl.all().last())
    by_team: dict[str, list[dict]] = {}
    for row in frame.iter_rows(named=True):
        labelled = status_label(row.get("report_status"), row.get("practice_status"))
        injury = clean(row.get("report_primary_injury")) or clean(
            row.get("practice_primary_injury")
        )
        if labelled is None:
            # A full participant with a listed body part is still an injury-
            # report entry; keep it as "Full" so the card shows everything.
            if not injury:
                continue
            status, kind, severity = "Full", "full", SEVERITY["full"]
        else:
            status, kind, severity = labelled
        entry = {
            "player_id": row.get("gsis_id"),
            "name": clean(row.get("full_name")) or clean(row.get("gsis_id")),
            "position": clean(row.get("position")),
            "status": status,
            "kind": kind,
            "severity": severity,
            "injury": injury or None,
            "report_status": clean(row.get("report_status")) or None,
            "practice_status": clean(row.get("practice_status")) or None,
        }
        by_team.setdefault(clean(row.get("team")), []).append(entry)
    for team in by_team:
        by_team[team].sort(
            key=lambda item: (
                item["severity"],
                item["position"] or "ZZ",
                item["name"],
            )
        )
    return by_team


def starting_qb(players: list[dict], team: str) -> dict | None:
    candidates = [
        player
        for player in players
        if player.get("team") == team and player.get("position") == "QB"
    ]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda player: (
            player.get("depth_rank") if player.get("depth_rank") is not None else 99,
            -(player.get("projected_fantasy_points") or 0),
        ),
    )
    qb = ranked[0]
    return {
        "player_id": qb.get("player_id"),
        "name": qb.get("name"),
        "status": clean(qb.get("injury_status")) or None,
        "depth_rank": qb.get("depth_rank"),
        "projected": qb.get("projected_fantasy_points"),
    }


def main() -> None:
    players = load(PLAYERS, [])
    upcoming = load(UPCOMING, [])
    meta = load(META, {})
    if not isinstance(players, list) or not isinstance(upcoming, list):
        print("Missing players or upcoming games; skipping game context.")
        return
    season = int(meta.get("season") or 2026)
    week = int(meta.get("week") or 0)
    if not week:
        weeks = [game.get("week") for game in upcoming if game.get("week") is not None]
        week = min(weeks) if weeks else 0

    injuries = injury_rows(season, week)
    by_id = {
        str(player.get("player_id")): player
        for player in players
        if player.get("player_id")
    }
    games = {}
    for game in upcoming:
        if int(game.get("week") or 0) != week:
            continue
        game_id = game.get("game_id")
        if not game_id:
            continue
        home, away = game.get("home_team"), game.get("away_team")
        sides = {}
        for side, team in (("home", home), ("away", away)):
            team_injuries = injuries.get(team, []) if team else []
            qb = starting_qb(players, team) if team else None
            if qb and qb.get("player_id") in by_id:
                injured = next(
                    (
                        row
                        for row in team_injuries
                        if str(row.get("player_id")) == str(qb["player_id"])
                    ),
                    None,
                )
                if injured:
                    qb["status"] = injured["status"]
                    qb["injury"] = injured.get("injury")
            sides[side] = {
                "team": team,
                "qb": qb,
                "injuries": team_injuries,
            }
        weather = {
            key: game.get(key)
            for key in (
                "temp",
                "wind",
                "gust",
                "precip",
                "precip_mm",
                "snow",
                "weather_code",
                "humidity",
            )
            if game.get(key) is not None
        }
        games[str(game_id)] = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "weather": weather,
            **sides,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "week": week,
        "games": games,
        "note": "Current-week player projections and the latest weekly injury report; future weeks have no context.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"Wrote {OUT}: {len(games)} games, "
        f"{sum(len(s['injuries']) for g in games.values() for s in (g['home'], g['away']))} injury entries"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Game context error: {exc}")
