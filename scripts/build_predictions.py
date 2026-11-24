#!/usr/bin/env python3
"""Build prediction JSON files for the frontend.

This script generates small, static JSON files under ``predictions/`` that the
Cloudflare Pages frontend can fetch via jsDelivr.  The values are placeholders
until the real models are trained; the pipeline is wired so that once
``scripts/train_models.py`` writes real predictions, this script can be replaced
with the real inference logic.
"""

from __future__ import annotations

import json
import os

os.makedirs("predictions", exist_ok=True)


def write_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def build_games() -> None:
    games = [
        {
            "game_id": 0,
            "season": 2024,
            "week": 1,
            "home_team": "Team A",
            "away_team": "Team C",
            "home_win_prob": 0.62,
            "spread": -3.5,
            "total": 45.5,
        },
        {
            "game_id": 1,
            "season": 2024,
            "week": 2,
            "home_team": "Team B",
            "away_team": "Team D",
            "home_win_prob": 0.48,
            "spread": 1.5,
            "total": 42.0,
        },
    ]
    write_json(games, "predictions/games_week.json")


def build_players() -> None:
    players = [
        {
            "player_id": "0000",
            "name": "Player One",
            "team": "Team A",
            "position": "WR",
            "projected_targets": 8.2,
            "projected_receptions": 5.4,
            "projected_yards": 72.5,
            "projected_tds": 0.6,
        },
        {
            "player_id": "0001",
            "name": "Player Two",
            "team": "Team B",
            "position": "QB",
            "projected_targets": 0.0,
            "projected_receptions": 0.0,
            "projected_yards": 0.0,
            "projected_tds": 0.0,
            "projected_pass_yards": 245.0,
            "projected_pass_tds": 1.8,
        },
    ]
    write_json(players, "predictions/players_week.json")


def build_playoffs() -> None:
    playoff_data = [
        {"team": "Team A", "playoff_prob": 0.78, "super_bowl_prob": 0.12},
        {"team": "Team B", "playoff_prob": 0.65, "super_bowl_prob": 0.08},
        {"team": "Team C", "playoff_prob": 0.42, "super_bowl_prob": 0.03},
        {"team": "Team D", "playoff_prob": 0.31, "super_bowl_prob": 0.01},
    ]
    write_json(playoff_data, "predictions/playoffs.json")


def main() -> None:
    build_games()
    build_players()
    build_playoffs()
    print("All prediction files written to predictions/.")


if __name__ == "__main__":
    main()