#!/usr/bin/env python3
"""Monte Carlo NFL playoff / Super Bowl simulator.

Reads the latest schedules, computes team ratings from the current season, then
simulates the remainder of the season and the playoff bracket many times to
produce:

- ``playoff_prob``     – probability of making the playoffs
- ``conference_prob``  – probability of winning the conference
- ``super_bowl_prob``  – probability of winning the Super Bowl

Writes ``predictions/playoffs.json``.  Degrades gracefully if data or team
metadata is unavailable.
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


def load_schedules() -> pl.DataFrame | None:
    if os.path.exists("data/schedules.parquet"):
        return pl.read_parquet("data/schedules.parquet")
    if os.path.exists("data/schedules.csv"):
        return pl.read_csv("data/schedules.csv")
    return None


def compute_standings(schedules: pl.DataFrame) -> tuple[pl.DataFrame | None, int | None]:
    df = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if df.height == 0:
        return None, None

    season = int(df["season"].max())
    df = df.filter(pl.col("season") == season)

    home = df.select(
        pl.col("home_team").alias("team"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("win"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
    )
    away = df.select(
        pl.col("away_team").alias("team"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Int8).alias("win"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
    )
    long = pl.concat([home, away])

    standings = (
        long.group_by("team")
        .agg(
            pl.col("win").sum().alias("wins"),
            pl.len().alias("games"),
            pl.col("pf").sum().alias("pf"),
            pl.col("pa").sum().alias("pa"),
        )
        .with_columns(
            (pl.col("wins") / pl.col("games")).alias("win_pct"),
            (pl.col("pf") - pl.col("pa")).alias("point_diff"),
        )
        .with_columns(
            (
                pl.col("win_pct") * 2.0
                + ((pl.col("point_diff") / pl.col("games")) / 10.0)
            ).alias("rating")
        )
    )
    return standings, season


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


def simulate_game(team_a, team_b, home_advantage: float = 0.0):
    """Return the winning (team, rating) tuple."""
    rating_a = team_a[1] + home_advantage
    rating_b = team_b[1]
    prob_a = 1.0 / (1.0 + math.exp(-(rating_a - rating_b) * 3.0))
    prob_a = min(max(prob_a, 0.05), 0.95)
    return team_a if random.random() < prob_a else team_b


def simulate(standings: pl.DataFrame, conf_map: dict[str, str], n_sims: int = 10_000) -> list[dict]:
    teams = standings.to_dicts()
    if not teams:
        return []

    # Group by conference.  If metadata is missing, split alphabetically.
    confs: dict[str, list[dict]] = {}
    if conf_map:
        for t in teams:
            conf = conf_map.get(t["team"], "OTHER")
            confs.setdefault(conf, []).append(t)
    if len(confs) < 2:
        ordered = sorted(teams, key=lambda x: x["team"])
        mid = len(ordered) // 2
        confs = {"AFC": ordered[:mid], "NFC": ordered[mid:]}

    # Keep only conferences with at least 7 teams; otherwise fall back to a
    # single 14-team pool split in half.
    confs = {k: v for k, v in confs.items() if len(v) >= 7}
    if len(confs) < 2:
        ordered = sorted(teams, key=lambda x: x["team"])
        mid = max(7, len(ordered) // 2)
        confs = {"AFC": ordered[:mid], "NFC": ordered[mid:]}

    counts = {t["team"]: {"playoffs": 0, "conf": 0, "sb": 0} for t in teams}

    for _ in range(n_sims):
        conf_seeds: dict[str, list[tuple[dict, float]]] = {}

        for conf, team_list in confs.items():
            sim_list = []
            for t in team_list:
                sim_rating = float(t["rating"]) + random.gauss(0, 0.08)
                sim_list.append((t, sim_rating))
            sim_list.sort(key=lambda x: (-x[1], -float(x[0]["point_diff"])))
            seeds = sim_list[:7]
            conf_seeds[conf] = seeds
            for t, _ in seeds:
                counts[t["team"]]["playoffs"] += 1

        conf_champs: dict[str, tuple[dict, float]] = {}

        for conf, seeds in conf_seeds.items():
            if len(seeds) < 7:
                continue

            # Wild card round: 2v7, 3v6, 4v5
            wc = []
            for a, b in [(1, 6), (2, 5), (3, 4)]:
                wc.append(simulate_game(seeds[a], seeds[b], home_advantage=0.10))

            # Divisional round: seed 1 vs lowest remaining; other two play.
            wc_sorted = sorted(wc, key=lambda x: seeds.index(x))
            div = [
                simulate_game(seeds[0], wc_sorted[-1], home_advantage=0.10),
                simulate_game(wc_sorted[0], wc_sorted[1], home_advantage=0.10),
            ]

            # Conference championship
            champ = simulate_game(div[0], div[1], home_advantage=0.05)
            conf_champs[conf] = champ
            counts[champ[0]["team"]]["conf"] += 1

        # Super Bowl (neutral site)
        if "AFC" in conf_champs and "NFC" in conf_champs:
            sb = simulate_game(conf_champs["AFC"], conf_champs["NFC"], home_advantage=0.0)
            counts[sb[0]["team"]]["sb"] += 1

    results = []
    for t in teams:
        c = counts[t["team"]]
        results.append(
            {
                "team": t["team"],
                "conference": conf_map.get(t["team"], ""),
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

    standings, season = compute_standings(schedules)
    if standings is None:
        print("No completed games found; leaving playoff predictions unchanged.")
        return

    conf_map = conference_map()
    results = simulate(standings, conf_map, n_sims=10_000)

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/playoffs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote predictions/playoffs.json for {season} ({len(results)} teams)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Playoff simulation error: {exc}")
    sys.exit(0)