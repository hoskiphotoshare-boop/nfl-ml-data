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
    "week",
    "season",
]


def load_features() -> pl.DataFrame:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError("features/match_features.parquet not found")
    return pl.read_parquet(path)


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("home_win").is_not_null())
    for col in FEATURE_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in FEATURE_COLS]
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
    df = prepare(load_features())
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

        X_train = train.select(FEATURE_COLS).to_numpy()
        y_train = train["home_win"].to_numpy()
        X_test = test.select(FEATURE_COLS).to_numpy()
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
        "feature_set": FEATURE_COLS,
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