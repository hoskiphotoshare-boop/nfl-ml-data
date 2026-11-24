#!/usr/bin/env python3
"""Monte Carlo NFL playoff / Super Bowl simulator.

The simulator:

1. Reads the current season's schedules.
2. Builds current standings from completed games.
3. Simulates the remaining regular-season games (null scores) using team ratings.
4. Seeds the top 7 teams per conference.
5. Simulates the playoff bracket 10,000 times.

Outputs ``predictions/playoffs.json`` with playoff, conference, and Super Bowl
probabilities for every team.  Degrades gracefully if data is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import polars as pl

try:
    from nflreadpy import load_teams
    TEAMS_AVAILABLE = True
except Exception:  # pragma: no cover
    TEAMS_AVAILABLE = False


N_SIMS = 10_000


def load_schedules() -> pl.DataFrame | None:
    if os.path.exists("data/schedules.parquet"):
        return pl.read_parquet("data/schedules.parquet")
    if os.path.exists("data/schedules.csv"):
        return pl.read_csv("data/schedules.csv")
    return None


def conference_map() -> dict[str, str]:
    if not TEAMS_AVAILABLE:
        return {}
    try:
        teams = load_teams()
        abbr_col = next((c for c in teams.columns if "abbr" in c.lower()), None)
        conf_col = next((c for c in teams.columns if "conf" in c.lower()), None)
        if abbr_col and conf_col:
            return {row[abbr_col]: row[conf_col] for row in teams.iter_rows(named=True)}
    except Exception as exc:  # pragma: no cover
        print(f"Could not load team conferences: {exc}")
    return {}


def build_current_state(schedules: pl.DataFrame):
    """Return (records_dict, remaining_games, season)."""
    df = schedules.filter(pl.col("season") == schedules["season"].max())
    season = int(df["season"].max())

    played = df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    remaining = df.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    )

    if played.height == 0:
        return None, [], season

    home = played.select(
        pl.col("home_team").alias("team"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("win"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
    )
    away = played.select(
        pl.col("away_team").alias("team"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Int8).alias("win"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
    )
    long = pl.concat([home, away])

    agg = (
        long.group_by("team")
        .agg(
            pl.col("win").sum().alias("wins"),
            pl.len().alias("games"),
            pl.col("pf").sum().alias("pf"),
            pl.col("pa").sum().alias("pa"),
        )
        .filter(pl.col("games") > 0)
    )

    records: dict[str, dict] = {}
    for row in agg.iter_rows(named=True):
        records[row["team"]] = {
            "wins": int(row["wins"]),
            "games": int(row["games"]),
            "pf": int(row["pf"]),
            "pa": int(row["pa"]),
        }

    remaining_games = [
        (row["home_team"], row["away_team"])
        for row in remaining.iter_rows(named=True)
    ]
    return records, remaining_games, season


def team_rating(rec: dict) -> float:
    games = max(rec["games"], 1)
    win_pct = rec["wins"] / games
    point_diff_per_game = (rec["pf"] - rec["pa"]) / games
    return win_pct * 2.0 + point_diff_per_game / 10.0


def simulate_game(team_a, team_b, home_advantage: float = 0.0):
    """Return the winning (team_name, rating) tuple."""
    rating_a = team_a[1] + home_advantage
    rating_b = team_b[1]
    prob_a = 1.0 / (1.0 + math.exp(-(rating_a - rating_b) * 3.0))
    prob_a = min(max(prob_a, 0.05), 0.95)
    return team_a if random.random() < prob_a else team_b


def simulate(records: dict, remaining_games: list[tuple[str, str]], conf_map: dict, n_sims: int = N_SIMS):
    if not records:
        return []

    counts = {team: {"playoffs": 0, "conf": 0, "sb": 0} for team in records}

    for _ in range(n_sims):
        # Start from current records and simulate the remaining schedule.
        sim = {team: rec.copy() for team, rec in records.items()}
        for home, away in remaining_games:
            if home not in sim or away not in sim:
                continue
            home_rating = team_rating(sim[home])
            away_rating = team_rating(sim[away])
            prob_home = 1.0 / (1.0 + math.exp(-(home_rating - away_rating + 0.10) * 3.0))
            prob_home = min(max(prob_home, 0.05), 0.95)
            if random.random() < prob_home:
                sim[home]["wins"] += 1
            else:
                sim[away]["wins"] += 1
            sim[home]["games"] += 1
            sim[away]["games"] += 1

        # Group by conference.
        confs: dict[str, list[tuple[str, float]]] = {}
        if conf_map:
            for team in sim:
                conf = conf_map.get(team, "OTHER")
                confs.setdefault(conf, []).append((team, team_rating(sim[team])))
        if len(confs) < 2:
            ordered = sorted(sim.keys())
            mid = len(ordered) // 2
            confs = {
                "AFC": [(t, team_rating(sim[t])) for t in ordered[:mid]],
                "NFC": [(t, team_rating(sim[t])) for t in ordered[mid:]],
            }

        # Keep only conferences with enough teams.
        confs = {k: v for k, v in confs.items() if len(v) >= 7}
        if len(confs) < 2:
            ordered = sorted(sim.keys())
            mid = max(7, len(ordered) // 2)
            confs = {
                "AFC": [(t, team_rating(sim[t])) for t in ordered[:mid]],
                "NFC": [(t, team_rating(sim[t])) for t in ordered[mid:]],
            }

        conf_seeds: dict[str, list[tuple[str, float]]] = {}
        for conf, teams in confs.items():
            ranked = sorted(teams, key=lambda x: (-x[1], x[0]))[:7]
            conf_seeds[conf] = ranked
            for team, _ in ranked:
                counts[team]["playoffs"] += 1

        conf_champs: dict[str, tuple[str, float]] = {}

        for conf, seeds in conf_seeds.items():
            if len(seeds) < 7:
                continue

            # Wild card: 2v7, 3v6, 4v5 (zero-based indices 1/6, 2/5, 3/4)
            wc = [
                simulate_game(seeds[1], seeds[6], home_advantage=0.10),
                simulate_game(seeds[2], seeds[5], home_advantage=0.10),
                simulate_game(seeds[3], seeds[4], home_advantage=0.10),
            ]
            # Divisional: seed 1 vs lowest remaining; other two play.
            wc_sorted = sorted(wc, key=lambda x: seeds.index(x))
            div = [
                simulate_game(seeds[0], wc_sorted[-1], home_advantage=0.10),
                simulate_game(wc_sorted[0], wc_sorted[1], home_advantage=0.10),
            ]
            champ = simulate_game(div[0], div[1], home_advantage=0.05)
            conf_champs[conf] = champ
            counts[champ[0]]["conf"] += 1

        if "AFC" in conf_champs and "NFC" in conf_champs:
            sb = simulate_game(conf_champs["AFC"], conf_champs["NFC"], home_advantage=0.0)
            counts[sb[0]]["sb"] += 1

    results = []
    for team in records:
        c = counts[team]
        results.append(
            {
                "team": team,
                "conference": conf_map.get(team, ""),
                "playoff_prob": round(c["playoffs"] / n_sims, 4),
                "conference_prob": round(c["conf"] / n_sims, 4),
                "super_bowl_prob": round(c["sb"] / n_sims, 4),
            }
        )
    results.sort(key=lambda x: (-x["super_bowl_prob"], -x["playoff_prob"]))
    return results


def main() -> None:
    schedules = load_schedules()
    if schedules is None:
        print("No schedules found; leaving playoff predictions unchanged.")
        return

    records, remaining_games, season = build_current_state(schedules)
    if records is None:
        print("No completed games found; leaving playoff predictions unchanged.")
        return

    print(
        f"Season {season}: {len(records)} teams, "
        f"{len(remaining_games)} remaining games to simulate"
    )

    results = simulate(records, remaining_games, conference_map(), n_sims=N_SIMS)

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/playoffs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote predictions/playoffs.json ({len(results)} teams)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Playoff simulation error: {exc}")
    sys.exit(0)