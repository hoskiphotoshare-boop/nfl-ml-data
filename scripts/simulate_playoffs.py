#!/usr/bin/env python3
"""Monte Carlo NFL playoff / Super Bowl simulator.

The simulator:

1. Reads the current season's schedules.
2. Builds current standings from completed games.
3. Simulates the remaining regular-season games using the trained match model.
4. Seeds the top 7 teams per conference (wins, then point differential).
5. Simulates the playoff bracket 10,000 times.

If the match model or its feature table is unavailable it falls back to a
simple win-percentage / point-differential rating, so the pipeline still works.

Outputs ``predictions/playoffs.json`` with playoff, conference, and Super Bowl
probabilities for every team.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

try:
    from nflreadpy import load_teams
    TEAMS_AVAILABLE = True
except Exception:  # pragma: no cover
    TEAMS_AVAILABLE = False

try:
    import joblib
    from predict_upcoming import (
        build_feature_vector,
        team_latest_row,
        team_stats_from_row,
    )

    MODEL_IMPORTS = True
except Exception:  # pragma: no cover
    MODEL_IMPORTS = False


N_SIMS = 10_000
PLAYOFF_WEEK = 20
HOME_ADVANTAGE = 0.10

# Playoff simulation settings.
# The match model is blended with an opponent-adjusted point-differential
# rating: the model's NGS-driven team ranking is unstable for hypothetical
# matchups (e.g. it ranked a +6.5 point-differential team first), while the
# rating alone ignores matchup detail. A 50/50 logit blend is more robust.
MODEL_WEIGHT = 0.5
# One logit unit of season-long team uncertainty (injuries, form, coaching).
TEAM_NOISE = 0.8
# NFL margin standard deviation, used to turn rating gaps into probabilities.
MARGIN_SIGMA = 13.5
RATING_HOME_ADVANTAGE = 2.5


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
        (
            row["home_team"],
            row["away_team"],
            int(row["week"]),
            int(row.get("div_game") or 0),
        )
        for row in remaining.iter_rows(named=True)
    ]
    return records, remaining_games, season


def team_rating(rec: dict) -> float:
    games = max(rec["games"], 1)
    win_pct = rec["wins"] / games
    point_diff_per_game = (rec["pf"] - rec["pa"]) / games
    return win_pct * 2.0 + point_diff_per_game / 10.0


def rating_prob(home_rating: float, away_rating: float) -> float:
    prob = 1.0 / (1.0 + math.exp(-(home_rating - away_rating + HOME_ADVANTAGE) * 3.0))
    return min(max(prob, 0.05), 0.95)


def logit(prob: float) -> float:
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return math.log(prob / (1.0 - prob))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def srs_ratings(schedules: pl.DataFrame, season: int) -> dict[str, float]:
    """Opponent-adjusted point differential (simple rating system)."""
    played = schedules.filter(
        (pl.col("season") == season)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    games = [
        (row["home_team"], row["away_team"], int(row["home_score"]) - int(row["away_score"]))
        for row in played.iter_rows(named=True)
    ]
    teams = {team for home, away, _ in games for team in (home, away)}
    rating = {team: 0.0 for team in teams}
    for _ in range(50):
        buckets: dict[str, list[float]] = {team: [] for team in teams}
        for home, away, margin in games:
            buckets[home].append(margin + rating[away])
            buckets[away].append(-margin + rating[home])
        updated = {
            team: (sum(values) / len(values) if values else 0.0)
            for team, values in buckets.items()
        }
        if max(abs(updated[t] - rating[t]) for t in teams) < 1e-6:
            rating = updated
            break
        rating = updated
    return rating


def rating_probability(rating: dict, home: str, away: str, neutral: bool = False) -> float:
    gap = rating.get(home, 0.0) - rating.get(away, 0.0)
    if not neutral:
        gap += RATING_HOME_ADVANTAGE
    return 0.5 * (1.0 + math.erf(gap / (MARGIN_SIGMA * math.sqrt(2.0))))


class ModelProbabilities:
    """Match-model probabilities for any pairing of the current rosters."""

    def __init__(self, season: int):
        self.bundle = joblib.load("models/match_model.joblib")
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]
        self.season = season
        frames = pl.read_parquet("features/match_features.parquet")
        teams = set(frames["home_team"].unique().to_list()) | set(
            frames["away_team"].unique().to_list()
        )
        self.stats = {
            team: team_stats_from_row(team_latest_row(frames, team), team)
            for team in teams
        }
        # Simulated games are future weeks; the last played game's rest and
        # short-week flags are stale. Use a neutral week for every team.
        for stats in self.stats.values():
            stats["rest_days"] = 7
            stats["short_week"] = 0
        self.cache: dict[tuple[str, str, int], float] = {}
        self.platt = None
        if os.path.exists("models/platt.json"):
            try:
                with open("models/platt.json", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("enabled"):
                    self.platt = data
            except Exception:
                pass

    def prob(self, home: str, away: str, week: int, div_game: int = 0) -> float | None:
        if home not in self.stats or away not in self.stats:
            return None
        key = (home, away, week, div_game)
        if key in self.cache:
            return self.cache[key]
        home_stats = self.stats[home]
        previous = home_stats.get("div_game")
        home_stats["div_game"] = div_game
        try:
            vector = build_feature_vector(
                self.features, home_stats, self.stats[away], week, self.season
            )
        finally:
            if previous is None:
                home_stats.pop("div_game", None)
            else:
                home_stats["div_game"] = previous
        prob = float(self.model.predict_proba([vector])[0, 1])
        if self.platt:
            p = min(max(prob, 1e-6), 1 - 1e-6)
            value = self.platt["a"] * math.log(p / (1 - p)) + self.platt["b"]
            prob = 1.0 / (1.0 + math.exp(-value))
        prob = min(max(prob, 0.02), 0.98)
        self.cache[key] = prob
        return prob

    def neutral_prob(self, team_a: str, team_b: str, week: int) -> float | None:
        forward = self.prob(team_a, team_b, week, 0)
        reverse = self.prob(team_b, team_a, week, 0)
        if forward is None or reverse is None:
            return None
        return 0.5 * (forward + (1.0 - reverse))


def make_win_prob(model: ModelProbabilities | None, sim: dict, rating: dict, noise: dict):
    """Blend match-model and SRS probabilities, then apply team uncertainty."""

    def win_prob(
        home: str, away: str, week: int, neutral: bool = False, div_game: int = 0
    ) -> float:
        p_rating = rating_probability(rating, home, away, neutral)
        if model is not None:
            p_model = (
                model.neutral_prob(home, away, week)
                if neutral
                else model.prob(home, away, week, div_game)
            )
            if p_model is None:
                value = logit(p_rating)
            else:
                value = MODEL_WEIGHT * logit(p_model) + (1.0 - MODEL_WEIGHT) * logit(p_rating)
        else:
            value = logit(p_rating)
        value += noise.get(home, 0.0) - noise.get(away, 0.0)
        return min(max(sigmoid(value), 0.03), 0.97)

    return win_prob


def simulate(
    records: dict,
    remaining_games: list[tuple[str, str, int, int]],
    conf_map: dict,
    n_sims: int = N_SIMS,
    model: ModelProbabilities | None = None,
    rating: dict | None = None,
):
    if not records:
        return []

    counts = {team: {"playoffs": 0, "conf": 0, "sb": 0} for team in records}
    rating = rating or {}
    random.seed(42)  # reproducible Monte Carlo output between runs

    for _ in range(n_sims):
        sim = {team: rec.copy() for team, rec in records.items()}
        noise = {team: random.gauss(0.0, TEAM_NOISE) for team in records}
        win_prob = make_win_prob(model, sim, rating, noise)

        # Remaining regular season.
        for home, away, week, div_game in remaining_games:
            if home not in sim or away not in sim:
                continue
            if random.random() < win_prob(home, away, week, div_game=div_game):
                sim[home]["wins"] += 1
            else:
                sim[away]["wins"] += 1
            sim[home]["games"] += 1
            sim[away]["games"] += 1

        # Seeds: wins first, then point differential (static), then team name.
        confs: dict[str, list[str]] = {}
        for team in sim:
            conf = conf_map.get(team, "OTHER")
            confs.setdefault(conf, []).append(team)

        confs = {k: v for k, v in confs.items() if len(v) >= 7}
        if len(confs) < 2:
            ordered = sorted(sim.keys())
            mid = max(7, len(ordered) // 2)
            confs = {"AFC": ordered[:mid], "NFC": ordered[mid:]}

        conf_seeds: dict[str, list[str]] = {}
        for conf, teams in confs.items():
            def seed_key(team: str):
                diff = (sim[team]["pf"] - sim[team]["pa"]) / max(sim[team]["games"], 1)
                return (-sim[team]["wins"], -diff, team)

            ranked = sorted(teams, key=seed_key)[:7]
            conf_seeds[conf] = ranked
            for team in ranked:
                counts[team]["playoffs"] += 1

        conf_champs: dict[str, str] = {}
        for conf, seeds in conf_seeds.items():
            if len(seeds) < 7:
                continue

            # Wild card: 2v7, 3v6, 4v5.
            wild_card = [
                seeds[1] if random.random() < win_prob(seeds[1], seeds[6], PLAYOFF_WEEK) else seeds[6],
                seeds[2] if random.random() < win_prob(seeds[2], seeds[5], PLAYOFF_WEEK) else seeds[5],
                seeds[3] if random.random() < win_prob(seeds[3], seeds[4], PLAYOFF_WEEK) else seeds[4],
            ]
            remaining = sorted(wild_card, key=lambda team: seeds.index(team))
            # Divisional: 1 seed vs lowest remaining, other two play.
            div_one = (
                seeds[0]
                if random.random() < win_prob(seeds[0], remaining[-1], PLAYOFF_WEEK)
                else remaining[-1]
            )
            div_two = (
                remaining[0]
                if random.random() < win_prob(remaining[0], remaining[1], PLAYOFF_WEEK)
                else remaining[1]
            )
            champ = (
                div_one
                if random.random() < win_prob(div_one, div_two, PLAYOFF_WEEK)
                else div_two
            )
            conf_champs[conf] = champ
            counts[champ]["conf"] += 1

        if "AFC" in conf_champs and "NFC" in conf_champs:
            afc, nfc = conf_champs["AFC"], conf_champs["NFC"]
            prob = win_prob(afc, nfc, PLAYOFF_WEEK, neutral=True)
            winner = afc if random.random() < prob else nfc
            counts[winner]["sb"] += 1

    results = []
    for team in records:
        record = counts[team]
        results.append(
            {
                "team": team,
                "conference": conf_map.get(team, ""),
                "playoff_prob": round(record["playoffs"] / n_sims, 4),
                "conference_prob": round(record["conf"] / n_sims, 4),
                "super_bowl_prob": round(record["sb"] / n_sims, 4),
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

    model = None
    if (
        MODEL_IMPORTS
        and os.path.exists("models/match_model.joblib")
        and os.path.exists("features/match_features.parquet")
    ):
        try:
            model = ModelProbabilities(season)
            print("Using the trained match model for game probabilities.")
        except Exception as exc:  # pragma: no cover
            print(f"Match model unavailable, using ratings instead: {exc}")
    else:
        print("Match model not found; using ratings instead.")

    rating = srs_ratings(schedules, season)
    results = simulate(
        records,
        remaining_games,
        conference_map(),
        n_sims=N_SIMS,
        model=model,
        rating=rating,
    )

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/playoffs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    top = ", ".join(f"{r['team']} {r['super_bowl_prob'] * 100:.1f}%" for r in results[:5])
    print(f"Wrote predictions/playoffs.json ({len(results)} teams)")
    print(f"Top Super Bowl odds: {top}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Playoff simulation error: {exc}")
    sys.exit(0)
