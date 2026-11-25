#!/usr/bin/env python3
"""Predict upcoming NFL games using the trained match model.

Builds a feature vector for each unplayed game from the most recent
team-level feature row, then writes ``predictions/upcoming_games.json``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import joblib
import polars as pl


def load_schedules() -> pl.DataFrame | None:
    for path in ["data/schedules.parquet", "data/schedules.csv"]:
        if os.path.exists(path):
            return pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
    return None


def load_match_features() -> pl.DataFrame | None:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        return None
    return pl.read_parquet(path)


def team_latest_row(df: pl.DataFrame, team: str) -> dict:
    rows = df.filter((pl.col("home_team") == team) | (pl.col("away_team") == team)).sort(
        ["season", "week"]
    )
    if rows.height == 0:
        return {}
    return rows.tail(1).to_dicts()[0]


def team_stats_from_row(row: dict, team: str) -> dict:
    if not row:
        return {}
    prefix = "home_" if row.get("home_team") == team else "away_"
    stats = {}
    for key, value in row.items():
        if key.startswith(prefix):
            stats[key[len(prefix):]] = value
    return stats


def build_feature_vector(features: list[str], home_stats: dict, away_stats: dict, week: int, season: int) -> list[float]:
    values = []
    for feat in features:
        if feat == "week":
            values.append(float(week))
        elif feat == "season":
            values.append(float(season))
        elif feat == "rest_diff":
            values.append(
                float(home_stats.get("rest_days") or 0) - float(away_stats.get("rest_days") or 0)
            )
        elif feat.startswith("home_"):
            values.append(float(home_stats.get(feat[5:]) or 0.0))
        elif feat.startswith("away_"):
            values.append(float(away_stats.get(feat[5:]) or 0.0))
        else:
            values.append(0.0)
    return values


def main() -> None:
    schedules = load_schedules()
    features_df = load_match_features()
    if schedules is None or features_df is None:
        print("Missing schedules or features; skipping upcoming predictions.")
        return
    if not os.path.exists("models/match_model.joblib"):
        print("No trained model found; skipping upcoming predictions.")
        return

    bundle = joblib.load("models/match_model.joblib")
    model = bundle["model"]
    feature_names = bundle["features"]

    current_season = int(schedules["season"].max())
    upcoming = schedules.filter(
        (pl.col("season") == current_season)
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    ).sort(["week", "gameday"] if "gameday" in schedules.columns else ["week"])

    if upcoming.height == 0:
        print("No upcoming games found.")
        return

    stats_cache: dict[str, dict] = {}
    records = []
    for row in upcoming.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        week = int(row["week"])

        if home not in stats_cache:
            stats_cache[home] = team_stats_from_row(
                team_latest_row(features_df, home), home
            )
        if away not in stats_cache:
            stats_cache[away] = team_stats_from_row(
                team_latest_row(features_df, away), away
            )

        x = build_feature_vector(
            feature_names, stats_cache[home], stats_cache[away], week, current_season
        )
        proba = float(model.predict_proba([x])[0, 1])

        home_avg = float(stats_cache[home].get("avg_points") or 0.0)
        away_avg = float(stats_cache[away].get("avg_points") or 0.0)
        home_allowed = float(stats_cache[home].get("avg_allowed") or 0.0)
        away_allowed = float(stats_cache[away].get("avg_allowed") or 0.0)
        proj_home = (home_avg + away_allowed) / 2.0 if home_avg and away_allowed else None
        proj_away = (away_avg + home_allowed) / 2.0 if away_avg and home_allowed else None
        spread = (proj_home - proj_away) if proj_home is not None and proj_away is not None else None
        total = (proj_home + proj_away) if proj_home is not None and proj_away is not None else None

        records.append(
            {
                "game_id": row.get("game_id"),
                "season": current_season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(proba, 3),
                "spread": round(spread, 1) if spread is not None else None,
                "total": round(total, 1) if total is not None else None,
                "gameday": str(row.get("gameday")) if row.get("gameday") is not None else None,
                "gametime": str(row.get("gametime")) if row.get("gametime") is not None else None,
                "location": row.get("location"),
                "stadium": row.get("stadium"),
                "roof": row.get("roof"),
                "surface": row.get("surface"),
                "div_game": row.get("div_game"),
            }
        )

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/upcoming_games.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open("predictions/meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "games": len(records),
                "season": current_season,
            },
            f,
            indent=2,
        )
    print(f"Wrote predictions/upcoming_games.json ({len(records)} games)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Upcoming prediction error: {exc}")
    sys.exit(0)