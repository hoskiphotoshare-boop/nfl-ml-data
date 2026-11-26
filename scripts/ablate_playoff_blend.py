#!/usr/bin/env python3
"""Research ablation: match-model vs SRS blend weight for the playoff sim.

The live simulator blends the match model 50/50 (in logit space) with an
opponent-adjusted point-differential rating. This backtest replays 2022–2025
from Week 12 with walk-forward match models and sweeps the blend weight, so the
constant is validated instead of assumed.

Writes ``reports/playoff_blend.json``. Nothing changes unless a weight beats the
current 0.5 consistently.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

import backtest_playoffs as B
import simulate_playoffs as S

MATCH = "features/match_features.parquet"
BUNDLE = "models/match_model.joblib"
OUT = "reports/playoff_blend.json"
SEASONS = (2022, 2023, 2024, 2025)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_SIMS = 1500


def clamp(p: float, lo: float = 0.03, hi: float = 0.97) -> float:
    return min(max(p, lo), hi)


def team_stats_before(
    frames: pl.DataFrame, season: int, cutoff: int
) -> dict[str, dict]:
    sub = frames.filter((pl.col("season") == season) & (pl.col("week") < cutoff))
    stats = {}
    for team in set(sub["home_team"].unique().to_list()) | set(
        sub["away_team"].unique().to_list()
    ):
        rows = sub.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        ).sort("week")
        if rows.height == 0:
            continue
        row = rows.tail(1).to_dicts()[0]
        prefix = "home_" if row.get("home_team") == team else "away_"
        values = {
            key[len(prefix):]: value
            for key, value in row.items()
            if key.startswith(prefix)
        }
        values["rest_days"] = 7
        values["short_week"] = 0
        values["div_game"] = 0
        stats[team] = values
    return stats


def train_model(
    frames: pl.DataFrame, season: int, features: list[str]
):
    train = frames.filter(
        (pl.col("season") < season) & pl.col("home_win").is_not_null()
    )
    X = np.nan_to_num(
        train.select(features).to_numpy().astype(float), nan=0.0
    )
    y = train["home_win"].to_numpy()
    model = LogisticRegression(max_iter=4000)
    model.fit(X, y)
    return model


class Blend:
    def __init__(
        self,
        model,
        features: list[str],
        stats: dict[str, dict],
        season: int,
    ):
        self.model = model
        self.features = features
        self.stats = stats
        self.season = season
        self.cache: dict[tuple[str, str, int], float] = {}

    def _raw(self, home: str, away: str, div: int) -> float | None:
        if home not in self.stats or away not in self.stats:
            return None
        key = (home, away, div)
        if key in self.cache:
            return self.cache[key]
        home_stats = dict(self.stats[home])
        home_stats["div_game"] = div
        vector = S.build_feature_vector(
            self.features, home_stats, self.stats[away], B.CUTOFF_WEEK, self.season
        )
        prob = float(self.model.predict_proba([vector])[0, 1])
        prob = clamp(prob, 0.02, 0.98)
        self.cache[key] = prob
        return prob

    def prob(self, home: str, away: str, div: int = 0, neutral: bool = False) -> float | None:
        if neutral:
            forward = self._raw(home, away, 0)
            reverse = self._raw(away, home, 0)
            if forward is None or reverse is None:
                return None
            return 0.5 * (forward + (1.0 - reverse))
        return self._raw(home, away, div)


def simulate(
    records: dict,
    remaining: list[tuple[str, str, int]],
    conf: dict,
    blend: Blend | None,
    weight: float,
    rating: dict,
    n_sims: int,
) -> dict[str, dict]:
    counts = {team: {"playoff": 0, "conf": 0, "sb": 0} for team in records}

    def win_prob(home: str, away: str, div: int, noise: dict, neutral: bool = False) -> float:
        p_rating = S.rating_probability(rating, home, away, neutral)
        if blend is not None and weight > 0:
            p_model = blend.prob(home, away, div, neutral)
        else:
            p_model = None
        if p_model is None:
            value = S.logit(p_rating)
        else:
            value = weight * S.logit(p_model) + (1.0 - weight) * S.logit(p_rating)
        value += noise.get(home, 0.0) - noise.get(away, 0.0)
        return clamp(S.sigmoid(value))

    random.seed(42)
    for _ in range(n_sims):
        sim = {team: rec.copy() for team, rec in records.items()}
        noise = {team: random.gauss(0.0, B.NOISE) for team in records}
        for home, away, div in remaining:
            if home not in sim or away not in sim:
                continue
            if random.random() < win_prob(home, away, div, noise):
                sim[home]["wins"] += 1
            else:
                sim[away]["wins"] += 1
            sim[home]["games"] += 1
            sim[away]["games"] += 1

        seeds = {}
        for group in ("AFC", "NFC"):
            teams = [t for t in sim if conf.get(t) == group]
            teams.sort(key=lambda t: B.seed_key(sim, t))
            seeds[group] = teams[:7]
            for team in seeds[group]:
                counts[team]["playoff"] += 1

        champs = {}
        for group, seeded in seeds.items():
            if len(seeded) < 7:
                continue
            wild = [
                seeded[1] if random.random() < win_prob(seeded[1], seeded[6], 0, noise) else seeded[6],
                seeded[2] if random.random() < win_prob(seeded[2], seeded[5], 0, noise) else seeded[5],
                seeded[3] if random.random() < win_prob(seeded[3], seeded[4], 0, noise) else seeded[4],
            ]
            remaining_wc = sorted(wild, key=lambda t: seeded.index(t))
            div_one = (
                seeded[0]
                if random.random() < win_prob(seeded[0], remaining_wc[-1], 0, noise)
                else remaining_wc[-1]
            )
            div_two = (
                remaining_wc[0]
                if random.random() < win_prob(remaining_wc[0], remaining_wc[1], 0, noise)
                else remaining_wc[1]
            )
            finalists = sorted([div_one, div_two], key=lambda t: seeded.index(t))
            champ = (
                finalists[0]
                if random.random() < win_prob(finalists[0], finalists[1], 0, noise)
                else finalists[1]
            )
            champs[group] = champ
            counts[champ]["conf"] += 1

        if "AFC" in champs and "NFC" in champs:
            sb = (
                champs["AFC"]
                if random.random() < win_prob(champs["AFC"], champs["NFC"], 0, noise, neutral=True)
                else champs["NFC"]
            )
            counts[sb]["sb"] += 1

    return {
        team: {key: value / n_sims for key, value in record.items()}
        for team, record in counts.items()
    }


def main() -> None:
    frames = pl.read_parquet(MATCH)
    bundle = joblib.load(BUNDLE)
    features = [c for c in bundle["features"] if c in frames.columns]
    schedules = B.load_schedules()
    conf = B.conference_map()
    division = {}
    try:
        from nflreadpy import load_teams

        division = {
            row["team_abbr"]: row["team_division"]
            for row in load_teams().iter_rows(named=True)
        }
    except Exception:
        division = {}

    pooled = {
        f"w={weight}": {"playoff": ([], []), "conf": ([], []), "sb": ([], [])}
        for weight in WEIGHTS
    }
    per_season = {f"w={weight}": [] for weight in WEIGHTS}

    for season in SEASONS:
        season_df = schedules.filter(pl.col("season") == season)
        regular = season_df.filter(pl.col("game_type") == "REG")
        played = regular.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )
        pre = played.filter(pl.col("week") < B.CUTOFF_WEEK)
        if pre.height < 80:
            continue
        pre_games = [
            (r["home_team"], r["away_team"], int(r["home_score"]) - int(r["away_score"]))
            for r in pre.iter_rows(named=True)
        ]
        ratings = B.srs_ratings(pre_games)
        records = B.standings(pre_games)
        remaining = []
        for row in regular.filter(pl.col("week") >= B.CUTOFF_WEEK).iter_rows(named=True):
            home, away = row["home_team"], row["away_team"]
            div = 1 if division.get(home) and division.get(home) == division.get(away) else 0
            remaining.append((home, away, div))

        model = train_model(frames, season, features)
        stats = team_stats_before(frames, season, B.CUTOFF_WEEK)
        blend = Blend(model, features, stats, season)

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

        for weight in WEIGHTS:
            probs = simulate(
                records, remaining, conf, blend, weight, ratings, N_SIMS
            )
            row = {"season": season}
            for key, actual in (
                ("playoff", actual_playoff),
                ("conf", actual_conf),
                ("sb", actual_sb),
            ):
                brier, ll = B.score(probs, key, actual)
                row[key] = {"brier": round(brier, 4), "log_loss": round(ll, 4)}
                for team, record in probs.items():
                    p = clamp(record[key], 0.005, 0.995)
                    pooled[f"w={weight}"][key][0].append(p)
                    pooled[f"w={weight}"][key][1].append(1.0 if team in actual else 0.0)
            per_season[f"w={weight}"].append(row)

    summary = {}
    for weight, rounds in pooled.items():
        summary[weight] = {}
        for key, (probs, outcomes) in rounds.items():
            summary[weight][key] = {
                "brier": round(
                    statistics.mean((p - y) ** 2 for p, y in zip(probs, outcomes)), 4
                ),
                "log_loss": round(
                    statistics.mean(
                        -(y * math.log(p) + (1 - y) * math.log(1 - p))
                        for p, y in zip(probs, outcomes)
                    ),
                    4,
                ),
            }
        print(weight, summary[weight])

    current = summary.get("w=0.5", {})
    best = min(
        summary,
        key=lambda w: sum(summary[w][k]["log_loss"] for k in ("playoff", "conf", "sb")),
    )
    print(f"current=w=0.5 best={best}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(
            {"seasons": SEASONS, "simulations": N_SIMS, "pooled": summary, "per_season": per_season},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
