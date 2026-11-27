#!/usr/bin/env python3
"""Research ablation: pace features for the total (and margin) regressors.

The margin/total models share the match feature set, which is built for the
winner market. Totals should care about pace, not just efficiency. This script
builds pre-game rolling plays and pass rate from team stats and measures total
MAE walk-forward.

Writes ``reports/totals_ablation.json``; nothing is promoted unless it wins.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

MATCH = "features/match_features.parquet"
TEAM_STATS = "data/team_stats_latest.parquet"
BUNDLE = "models/score_models.joblib"
OUT = "reports/totals_ablation.json"
TEST_SEASONS = (2024, 2025, 2026)


def pace_features() -> pl.DataFrame:
    ts = pl.read_parquet(TEAM_STATS)
    ts = ts.filter(pl.col("season") >= 2022)
    ts = ts.with_columns(
        (pl.col("attempts") + pl.col("carries")).alias("plays"),
        (pl.col("attempts") / (pl.col("attempts") + pl.col("carries")).clip(1)).alias(
            "pass_rate"
        ),
        pl.col("fumbles_lost_total").alias("turnovers"),
    ).sort(["team", "season", "week"])

    cols = ["plays", "pass_rate", "turnovers"]
    ts = ts.with_columns(
        [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in cols]
        + [pl.col("plays").cum_count().over("team").alias("game_num")]
    )
    for c in cols:
        ts = ts.with_columns(
            ((pl.col(f"cum_{c}") - pl.col(c)) / (pl.col("game_num") - 1).clip(1)).alias(c)
        )

    home = ts.select(
        "game_id",
        pl.col("team").alias("home_team"),
        *[pl.col(c).alias(f"home_{c}") for c in cols],
    )
    away = ts.select(
        "game_id",
        pl.col("team").alias("away_team"),
        *[pl.col(c).alias(f"away_{c}") for c in cols],
    )
    return home.join(away, on="game_id", how="full", coalesce=True)


def main() -> None:
    import joblib

    bundle = joblib.load(BUNDLE)
    features = bundle.get("total_features", bundle.get("features"))
    match = pl.read_parquet(MATCH).filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    pace = pace_features()
    match = match.join(pace, on="game_id", how="left")
    match = match.with_columns(
        *[
            pl.col(c).fill_null(pl.col(c).mean())
            for c in match.columns
            if c in ("home_plays", "away_plays", "home_pass_rate", "away_pass_rate",
                     "home_turnovers", "away_turnovers")
        ]
    )

    pace_cols = ["home_plays", "away_plays"]
    rate_cols = ["home_pass_rate", "away_pass_rate"]
    turn_cols = ["home_turnovers", "away_turnovers"]
    variants = {
        "base": features,
        "base+plays": features + pace_cols,
        "base+pass_rate": features + rate_cols,
        "base+plays+rate": features + pace_cols + rate_cols,
        "base+plays+rate+turnovers": features + pace_cols + rate_cols + turn_cols,
    }

    seasons = match["season"].to_numpy()
    y_total = (match["home_score"] + match["away_score"]).to_numpy()
    y_margin = (match["home_score"] - match["away_score"]).to_numpy()

    results = {}
    for name, cols in variants.items():
        present = [c for c in cols if c in match.columns]
        X = match.select(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in present]
        ).to_numpy()
        total_mae = []
        margin_mae = []
        for test_season in TEST_SEASONS:
            train = seasons < test_season
            test = seasons == test_season
            total_model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
            ).fit(X[train], y_total[train])
            margin_model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
            ).fit(X[train], y_margin[train])
            total_mae.append(
                (mean_absolute_error(total_model.predict(X[test]), y_total[test]), int(test.sum()))
            )
            margin_mae.append(
                (mean_absolute_error(margin_model.predict(X[test]), y_margin[test]), int(test.sum()))
            )
        total_pooled = sum(m * n for m, n in total_mae) / sum(n for _, n in total_mae)
        margin_pooled = sum(m * n for m, n in margin_mae) / sum(n for _, n in margin_mae)
        results[name] = {
            "total_mae": round(float(total_pooled), 4),
            "margin_mae": round(float(margin_pooled), 4),
            "per_season_total": [round(m, 3) for m, _ in total_mae],
        }
        print(
            f"{name:28s} total MAE={total_pooled:.3f} "
            f"({results[name]['per_season_total']}) margin MAE={margin_pooled:.3f}"
        )

    best = min(results, key=lambda k: results[k]["total_mae"])
    print(f"baseline={results['base']['total_mae']:.3f} best={best}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
