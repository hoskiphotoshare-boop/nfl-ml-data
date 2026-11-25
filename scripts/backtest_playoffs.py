#!/usr/bin/env python3
"""Backtest playoff-qualification probabilities with the SRS simulator.

For each season, freeze at a cutoff week, rate teams with an
opponent-adjusted point-differential (SRS) model, simulate the rest of the
regular season, and score the probability of finishing top-7 in each
conference. This validates the simulator's structural calibration; the live
simulator additionally blends in the match model.

Bracket/Super Bowl validation needs playoff results and is not covered here.
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
        for team, opp_margin in ((home, margin), (away, -margin)):
            rec = records.setdefault(team, {"wins": 0, "games": 0, "diff": 0})
            rec["games"] += 1
            rec["diff"] += opp_margin
            if opp_margin > 0:
                rec["wins"] += 1
    return records


def top_seven(records: dict[str, dict], conf: dict[str, str]) -> set[str]:
    result = set()
    for group in ("AFC", "NFC"):
        teams = [t for t in records if conf.get(t) == group]
        teams.sort(
            key=lambda t: (
                -records[t]["wins"],
                -records[t]["diff"],
                t,
            )
        )
        result.update(teams[:7])
    return result


def simulate_season(
    records: dict[str, dict],
    remaining: list[tuple[str, str]],
    ratings: dict[str, float],
    conf: dict[str, str],
    n_sims: int,
) -> dict[str, float]:
    counts = {team: 0 for team in records}
    random.seed(42)
    for _ in range(n_sims):
        sim = {team: rec.copy() for team, rec in records.items()}
        noise = {team: random.gauss(0.0, NOISE) for team in records}
        for home, away in remaining:
            if home not in sim or away not in sim:
                continue
            gap = ratings.get(home, 0) - ratings.get(away, 0) + HFA
            prob = 0.5 * (1 + math.erf((gap + noise[home] - noise[away]) / (SIGMA * math.sqrt(2))))
            prob = min(max(prob, 0.03), 0.97)
            if random.random() < prob:
                sim[home]["wins"] += 1
            else:
                sim[away]["wins"] += 1
            sim[home]["games"] += 1
            sim[away]["games"] += 1
        for team in top_seven(sim, conf):
            counts[team] += 1
    return {team: count / n_sims for team, count in counts.items()}


def main() -> None:
    schedules = load_schedules()
    current = int(schedules["season"].max())
    conf = conference_map()
    # Exclude the in-progress season: its "actual" playoff field is not final.
    seasons = [season for season in range(2022, current)]
    rows = []
    pooled_actual = []
    pooled_prob = []

    for season in seasons:
        season_df = schedules.filter(pl.col("season") == season)
        played = season_df.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )
        if played.height < 100:
            continue
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
            for r in season_df.filter(pl.col("week") >= CUTOFF_WEEK).iter_rows(named=True)
        ]
        probs = simulate_season(records, remaining, ratings, conf, N_SIMS)

        final_games = [
            (r["home_team"], r["away_team"], int(r["home_score"]) - int(r["away_score"]))
            for r in played.iter_rows(named=True)
        ]
        actual = top_seven(standings(final_games), conf)

        y = [1 if team in actual else 0 for team in probs]
        p = [min(max(probs[team], 0.01), 0.99) for team in probs]
        brier = statistics.mean((pi - yi) ** 2 for pi, yi in zip(p, y))
        logloss = statistics.mean(
            -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi)) for pi, yi in zip(p, y)
        )
        rows.append(
            {
                "season": season,
                "teams": len(p),
                "brier": round(brier, 4),
                "log_loss": round(logloss, 4),
                "actual_playoff_teams": len(actual),
            }
        )
        pooled_actual.extend(y)
        pooled_prob.extend(p)

    if not rows:
        print("No seasons available for playoff backtest.")
        return

    pooled_brier = statistics.mean(
        (pi - yi) ** 2 for pi, yi in zip(pooled_prob, pooled_actual)
    )
    pooled_logloss = statistics.mean(
        -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
        for pi, yi in zip(pooled_prob, pooled_actual)
    )
    payload = {
        "cutoff_week": CUTOFF_WEEK,
        "simulations": N_SIMS,
        "seasons": rows,
        "pooled": {
            "brier": round(pooled_brier, 4),
            "log_loss": round(pooled_logloss, 4),
            "teams": len(pooled_actual),
        },
        "note": "SRS-only simulation; live sim also blends the match model.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Playoff backtest: {len(rows)} seasons, "
        f"pooled Brier {pooled_brier:.4f}, log loss {pooled_logloss:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Playoff backtest error: {exc}")
