#!/usr/bin/env python3
"""Backtest the full playoff bracket, including conference and Super Bowl.

For each season since 2020 (7-team bracket era), freeze at Week 12, rate teams
with an opponent-adjusted point-differential (SRS) model, simulate the rest of
the regular season and the entire postseason, and score:

- playoff qualification (top 7 per conference)
- conference champion
- Super Bowl champion

Actual outcomes come from nflverse postseason games (WC/DIV/CON/SB). The live
simulator additionally blends in the match model; this validates the bracket
structure and the SRS prior.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics

import polars as pl

try:
    from nflreadpy import load_teams
    TEAMS_AVAILABLE = True
except Exception:  # pragma: no cover
    TEAMS_AVAILABLE = False

CUTOFF_WEEK = 12
N_SIMS = 2000
SIGMA = 13.5
HFA = 2.5
NOISE = 0.8
OUTPUT = "reports/playoff_backtest.json"


def load_schedules() -> pl.DataFrame:
    return pl.read_parquet("data/schedules.parquet")


def conference_map() -> dict[str, str]:
    if not TEAMS_AVAILABLE:
        return {}
    try:
        teams = load_teams()
        return {row["team_abbr"]: row["team_conf"] for row in teams.iter_rows(named=True)}
    except Exception:
        return {}


def srs_ratings(games: list[tuple[str, str, int]]) -> dict[str, float]:
    teams = {team for home, away, _ in games for team in (home, away)}
    rating = {team: 0.0 for team in teams}
    for _ in range(50):
        buckets = {team: [] for team in teams}
        for home, away, margin in games:
            buckets[home].append(margin + rating[away])
            buckets[away].append(-margin + rating[home])
        updated = {
            team: (sum(values) / len(values) if values else 0.0)
            for team, values in buckets.items()
        }
        if max(abs(updated[t] - rating[t]) for t in teams) < 1e-6:
            return updated
        rating = updated
    return rating


def standings(games: list[tuple[str, str, int]]) -> dict[str, dict]:
    records = {}
    for home, away, margin in games:
        for team, value in ((home, margin), (away, -margin)):
            rec = records.setdefault(team, {"wins": 0, "games": 0, "diff": 0})
            rec["games"] += 1
            rec["diff"] += value
            if value > 0:
                rec["wins"] += 1
    return records


def seed_key(records: dict[str, dict], team: str):
    games = max(records[team]["games"], 1)
    return (-records[team]["wins"], -(records[team]["diff"] / games), team)


def play(home: str, away: str, ratings: dict, noise: dict, neutral: bool = False) -> str:
    gap = ratings.get(home, 0) - ratings.get(away, 0)
    if not neutral:
        gap += HFA
    gap += noise.get(home, 0) - noise.get(away, 0)
    prob = 0.5 * (1 + math.erf(gap / (SIGMA * math.sqrt(2))))
    prob = min(max(prob, 0.03), 0.97)
    return home if random.random() < prob else away


def simulate_season(
    records: dict,
    remaining: list[tuple[str, str]],
    ratings: dict,
    conf: dict,
    n_sims: int,
) -> dict[str, dict]:
    counts = {
        team: {"playoff": 0, "conf": 0, "sb": 0} for team in records
    }
    random.seed(42)
    for _ in range(n_sims):
        sim = {team: rec.copy() for team, rec in records.items()}
        noise = {team: random.gauss(0.0, NOISE) for team in records}
        for home, away in remaining:
            if home not in sim or away not in sim:
                continue
            winner = play(home, away, ratings, noise, neutral=False)
            sim[winner]["wins"] += 1
            sim[home]["games"] += 1
            sim[away]["games"] += 1

        seeds = {}
        for group in ("AFC", "NFC"):
            teams = [t for t in sim if conf.get(t) == group]
            teams.sort(key=lambda t: seed_key(sim, t))
            seeds[group] = teams[:7]
            for team in seeds[group]:
                counts[team]["playoff"] += 1

        champs = {}
        for group, seeded in seeds.items():
            if len(seeded) < 7:
                continue
            wild = [
                play(seeded[1], seeded[6], ratings, noise),
                play(seeded[2], seeded[5], ratings, noise),
                play(seeded[3], seeded[4], ratings, noise),
            ]
            remaining_wc = sorted(wild, key=lambda t: seeded.index(t))
            div_one = play(seeded[0], remaining_wc[-1], ratings, noise)
            div_two = play(remaining_wc[0], remaining_wc[1], ratings, noise)
            finalists = sorted([div_one, div_two], key=lambda t: seeded.index(t))
            champ = play(finalists[0], finalists[1], ratings, noise)
            champs[group] = champ
            counts[champ]["conf"] += 1

        if "AFC" in champs and "NFC" in champs:
            sb = play(champs["AFC"], champs["NFC"], ratings, noise, neutral=True)
            counts[sb]["sb"] += 1

    return {
        team: {key: value / n_sims for key, value in record.items()}
        for team, record in counts.items()
    }


def score(probs: dict[str, dict], key: str, actual: set[str]) -> tuple[float, float]:
    brier = 0.0
    log_loss = 0.0
    for team, record in probs.items():
        p = min(max(record[key], 0.005), 0.995)
        y = 1.0 if team in actual else 0.0
        brier += (p - y) ** 2
        log_loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
    n = max(len(probs), 1)
    return brier / n, log_loss / n


def main() -> None:
    schedules = load_schedules()
    current = int(schedules["season"].max())
    conf = conference_map()
    seasons = [season for season in range(2020, current)]

    rows = []
    pooled = {"playoff": ([], []), "conf": ([], []), "sb": ([], [])}

    for season in seasons:
        season_df = schedules.filter(pl.col("season") == season)
        regular = season_df.filter(pl.col("game_type") == "REG")
        played = regular.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )
        pre = played.filter(pl.col("week") < CUTOFF_WEEK)
        if pre.height < 80:
            continue

        pre_games = [
            (r["home_team"], r["away_team"], int(r["home_score"]) - int(r["away_score"]))
            for r in pre.iter_rows(named=True)
        ]
        ratings = srs_ratings(pre_games)
        records = standings(pre_games)
        remaining = [
            (r["home_team"], r["away_team"])
            for r in regular.filter(pl.col("week") >= CUTOFF_WEEK).iter_rows(named=True)
        ]
        probs = simulate_season(records, remaining, ratings, conf, N_SIMS)

        post = season_df.filter(pl.col("game_type") != "REG")
        actual_playoff = set(post["home_team"].to_list()) | set(post["away_team"].to_list())
        actual_conf = set()
        actual_sb = set()
        for row in post.iter_rows(named=True):
            if row["home_score"] is None or row["away_score"] is None:
                continue
            winner = (
                row["home_team"]
                if int(row["home_score"]) > int(row["away_score"])
                else row["away_team"]
            )
            if row["game_type"] == "CON":
                actual_conf.add(winner)
            elif row["game_type"] == "SB":
                actual_sb.add(winner)

        playoff_brier, playoff_ll = score(probs, "playoff", actual_playoff)
        conf_brier, conf_ll = score(probs, "conf", actual_conf)
        sb_brier, sb_ll = score(probs, "sb", actual_sb)

        rows.append(
            {
                "season": season,
                "teams": len(probs),
                "playoff_brier": round(playoff_brier, 4),
                "playoff_log_loss": round(playoff_ll, 4),
                "conf_brier": round(conf_brier, 4),
                "conf_log_loss": round(conf_ll, 4),
                "sb_brier": round(sb_brier, 4),
                "sb_log_loss": round(sb_ll, 4),
                "actual_sb_winner": sorted(actual_sb),
            }
        )

        for key, actual in (
            ("playoff", actual_playoff),
            ("conf", actual_conf),
            ("sb", actual_sb),
        ):
            for team, record in probs.items():
                p = min(max(record[key], 0.005), 0.995)
                pooled[key][0].append(p)
                pooled[key][1].append(1.0 if team in actual else 0.0)

    if not rows:
        print("No seasons available for playoff backtest.")
        return

    payload = {
        "cutoff_week": CUTOFF_WEEK,
        "simulations": N_SIMS,
        "seasons": rows,
        "pooled": {
            key: {
                "brier": round(
                    statistics.mean((p - y) ** 2 for p, y in zip(*pairs)), 4
                ),
                "log_loss": round(
                    statistics.mean(
                        -(y * math.log(p) + (1 - y) * math.log(1 - p))
                        for p, y in zip(*pairs)
                    ),
                    4,
                ),
                "observations": len(pairs[0]),
            }
            for key, pairs in pooled.items()
            if pairs[0]
        },
        "note": "SRS-only postseason simulation; live sim blends the match model.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Playoff backtest: {len(rows)} seasons | "
        f"playoff Brier {payload['pooled']['playoff']['brier']}, "
        f"conf {payload['pooled']['conf']['brier']}, "
        f"SB {payload['pooled']['sb']['brier']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Playoff backtest error: {exc}")
