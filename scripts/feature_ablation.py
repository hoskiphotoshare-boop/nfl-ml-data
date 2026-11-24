#!/usr/bin/env python3
"""Feature-group ablation study for the NFL match model.

Runs the same walk-forward backtest (train < season, test = season) for
several feature sets and writes a comparison report to
``reports/feature_ablation.json``.

Groups:

- base      : rolling points averages + week/season
- epa       : base + pre-game rolling EPA/success features
- rest      : rest-day features
- injuries  : raw injury counts (previously rejected)

The script is deliberately independent of the training pipeline so it can be
run manually or in CI.
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

BASE = [
    "home_avg_points",
    "home_avg_allowed",
    "away_avg_points",
    "away_avg_allowed",
    "week",
    "season",
]
EPA = [
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
]
REST = [
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_short_week",
    "away_short_week",
]
INJURIES = [
    "home_inj_out",
    "home_inj_doubtful",
    "home_inj_questionable",
    "home_inj_total",
    "away_inj_out",
    "away_inj_doubtful",
    "away_inj_questionable",
    "away_inj_total",
    "inj_out_diff",
    "inj_total_diff",
]
TEAM_STATS = [
    "home_points",
    "away_points",
    "home_points_allowed",
    "away_points_allowed",
    "home_offensive_epa",
    "away_offensive_epa",
    "home_defensive_epa",
    "away_defensive_epa",
    "home_passing_epa",
    "away_passing_epa",
    "home_rushing_epa",
    "away_rushing_epa",
    "home_special_teams_epa",
    "away_special_teams_epa",
    "home_success_rate",
    "away_success_rate",
    "home_pass_success_rate",
    "away_pass_success_rate",
    "home_rush_success_rate",
    "away_rush_success_rate",
    "home_cpoe",
    "away_cpoe",
    "home_wpa",
    "away_wpa",
    "home_epa_per_play",
    "away_epa_per_play",
    "home_pass_epa_per_play",
    "away_pass_epa_per_play",
    "home_rush_epa_per_play",
    "away_rush_epa_per_play",
]


def uniq(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_features() -> pl.DataFrame:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


def backtest(df: pl.DataFrame, features: list[str]) -> dict:
    seasons = sorted(int(s) for s in df["season"].unique().to_list())
    test_seasons = [s for s in seasons if s >= 2022]
    pooled_y: list[int] = []
    pooled_proba: list[float] = []
    per_season = []

    for season in test_seasons:
        train = df.filter(pl.col("season") < season)
        test = df.filter(pl.col("season") == season)
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
        per_season.append({"season": season, "games": int(len(y_test)), **metrics})
        pooled_y.extend(y_test.tolist())
        pooled_proba.extend(proba.tolist())

    pooled = {}
    if pooled_y:
        pooled = evaluate(np.array(pooled_y), np.array(pooled_proba))

    return {"features": features, "per_season": per_season, "pooled": pooled}


def main() -> None:
    raw = load_features()
    already_used = set(BASE + EPA + REST)
    team_stats_cols = [
        c for c in TEAM_STATS if c in raw.columns and c not in already_used
    ]

    experiments = {
        "base": uniq(BASE),
        "base+epa": uniq(BASE + EPA),
        "base+rest": uniq(BASE + REST),
        "base+epa+rest": uniq(BASE + EPA + REST),
        "base+epa+rest+injuries": uniq(BASE + EPA + REST + INJURIES),
    }
    if team_stats_cols:
        experiments["base+epa+rest+team_stats"] = uniq(BASE + EPA + REST + team_stats_cols)

    results = {}
    for name, features in experiments.items():
        df = prepare(raw, features)
        # Only keep feature columns that actually exist in the frame.
        present = [c for c in features if c in df.columns]
        if not present:
            continue
        report = backtest(df, present)
        results[name] = report
        pooled = report["pooled"]
        if pooled:
            print(
                f"{name:28s} acc={pooled['accuracy']:.4f} "
                f"logloss={pooled['log_loss']:.4f} "
                f"brier={pooled['brier']:.4f} "
                f"auc={pooled['auc']:.4f}"
            )

    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote reports/feature_ablation.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Feature ablation error: {exc}")
    sys.exit(0)