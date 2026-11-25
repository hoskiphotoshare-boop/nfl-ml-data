#!/usr/bin/env python3
"""Walk-forward backtest for the NFL match model.

For each test season from 2022 onward:

- Train on all prior seasons.
- Test on the target season.
- Report accuracy, log loss, Brier score, AUC, and the "always pick home"
  baseline.

Writes ``reports/backtest.json`` so results are versioned with the repo.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

FEATURE_COLS = [
    "home_avg_points",
    "home_avg_allowed",
    "away_avg_points",
    "away_avg_allowed",
    "home_off_epa",
    "home_def_epa",
    "home_success_rate",
    "home_pass_epa",
    "home_rush_epa",
    "away_off_epa",
    "away_def_epa",
    "away_success_rate",
    "away_pass_epa",
    "away_rush_epa",
    "home_passing_epa",
    "away_passing_epa",
    "home_rushing_epa",
    "away_rushing_epa",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_short_week",
    "away_short_week",
    "week",
    "season",
]

NG_PASS = [
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "aggressiveness",
    "completion_percentage_above_expectation",
    "passer_rating",
]
NG_RUSH = [
    "efficiency",
    "percent_attempts_gte_eight_defenders",
    "avg_time_to_los",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
]
NG_REC = [
    "avg_cushion",
    "avg_separation",
    "percent_share_of_intended_air_yards",
    "catch_percentage",
    "avg_yac_above_expectation",
]
NG_FEATURES = (
    [f"home_ng_pass_{c}" for c in NG_PASS]
    + [f"away_ng_pass_{c}" for c in NG_PASS]
    + [f"home_ng_rush_{c}" for c in NG_RUSH]
    + [f"away_ng_rush_{c}" for c in NG_RUSH]
    + [f"home_ng_rec_{c}" for c in NG_REC]
    + [f"away_ng_rec_{c}" for c in NG_REC]
)
FEATURE_COLS += NG_FEATURES


def load_features() -> pl.DataFrame:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError("features/match_features.parquet not found")
    return pl.read_parquet(path)


def prepare(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    df = df.filter(pl.col("home_win").is_not_null())
    for col in features:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in features]
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
    }


def main() -> None:
    features = FEATURE_COLS
    try:
        import joblib

        if os.path.exists("models/match_model.joblib"):
            features = joblib.load("models/match_model.joblib")["features"]
    except Exception:
        pass
    df = prepare(load_features(), features)
    seasons = sorted(int(s) for s in df["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= 2022]

    results = []
    pooled_y = []
    pooled_proba = []

    for test_season in test_seasons:
        train = df.filter(pl.col("season") < test_season)
        test = df.filter(pl.col("season") == test_season)
        if train.height < 100 or test.height < 20:
            continue

        X_train = train.select(features).to_numpy()
        y_train = train["home_win"].to_numpy()
        X_test = test.select(features).to_numpy()
        y_test = test["home_win"].to_numpy()

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, proba)
        home_rate = float(y_train.mean())
        baseline_proba = np.full_like(y_test, home_rate, dtype=float)
        baseline = evaluate(y_test, baseline_proba)

        row = {
            "season": test_season,
            "games": int(len(y_test)),
            **metrics,
            "home_baseline_accuracy": baseline["accuracy"],
            "home_baseline_log_loss": baseline["log_loss"],
            "home_baseline_brier": baseline["brier"],
        }
        results.append(row)

        pooled_y.extend(y_test.tolist())
        pooled_proba.extend(proba.tolist())

        print(
            f"{test_season}: n={row['games']} "
            f"acc={row['accuracy']:.3f} logloss={row['log_loss']:.3f} "
            f"brier={row['brier']:.3f} auc={row['auc']:.3f} "
            f"| baseline acc={baseline['accuracy']:.3f} "
            f"logloss={baseline['log_loss']:.3f}"
        )

    pooled = {}
    if pooled_y:
        pooled = evaluate(np.array(pooled_y), np.array(pooled_proba))

    report = {
        "feature_set": list(features),
        "per_season": results,
        "pooled": pooled,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/backtest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote reports/backtest.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Backtest error: {exc}")
    sys.exit(0)