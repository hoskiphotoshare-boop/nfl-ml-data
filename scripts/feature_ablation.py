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
SNAP = [
    "home_snap_players",
    "away_snap_players",
    "home_snap_off_total",
    "away_snap_off_total",
    "home_snap_def_total",
    "away_snap_def_total",
    "home_snap_off_mean_pct",
    "away_snap_off_mean_pct",
    "home_snap_off_players_50",
    "away_snap_off_players_50",
    "home_snap_def_mean_pct",
    "away_snap_def_mean_pct",
    "home_snap_def_players_50",
    "away_snap_def_players_50",
]

_NG_PASS = [
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "aggressiveness",
    "completion_percentage_above_expectation",
    "passer_rating",
]
_NG_RUSH = [
    "efficiency",
    "percent_attempts_gte_eight_defenders",
    "avg_time_to_los",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
]
_NG_REC = [
    "avg_cushion",
    "avg_separation",
    "percent_share_of_intended_air_yards",
    "catch_percentage",
    "avg_yac_above_expectation",
]

NEXTGEN = (
    [f"home_ng_pass_{c}" for c in _NG_PASS]
    + [f"away_ng_pass_{c}" for c in _NG_PASS]
    + [f"home_ng_rush_{c}" for c in _NG_RUSH]
    + [f"away_ng_rush_{c}" for c in _NG_RUSH]
    + [f"home_ng_rec_{c}" for c in _NG_REC]
    + [f"away_ng_rec_{c}" for c in _NG_REC]
)

_PFR_PASS = ["passing_drop_pct", "passing_bad_throw_pct", "times_pressured_pct"]
_PFR_RUSH = [
    "rushing_yards_before_contact_avg",
    "rushing_yards_after_contact_avg",
]
_PFR_REC = ["receiving_drop_pct", "receiving_rat"]
_PFR_DEF = [
    "def_completion_pct",
    "def_yards_allowed_per_tgt",
    "def_passer_rating_allowed",
    "def_missed_tackle_pct",
    "def_pressures",
    "def_sacks",
]

PFR = (
    [f"home_pfr_pass_{c}" for c in _PFR_PASS]
    + [f"away_pfr_pass_{c}" for c in _PFR_PASS]
    + [f"home_pfr_rush_{c}" for c in _PFR_RUSH]
    + [f"away_pfr_rush_{c}" for c in _PFR_RUSH]
    + [f"home_pfr_rec_{c}" for c in _PFR_REC]
    + [f"away_pfr_rec_{c}" for c in _PFR_REC]
    + [f"home_pfr_def_{c}" for c in _PFR_DEF]
    + [f"away_pfr_def_{c}" for c in _PFR_DEF]
)


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

    snap_cols = [c for c in SNAP if c in raw.columns and c not in already_used]
    if snap_cols:
        base_with_snaps = BASE + EPA + REST + (team_stats_cols or []) + snap_cols
        experiments["base+epa+rest+team_stats+snap_counts"] = uniq(base_with_snaps)

    nextgen_cols = [
        c for c in NEXTGEN if c in raw.columns and c not in already_used
    ]
    if nextgen_cols:
        experiments["base+epa+rest+team_stats+nextgen"] = uniq(
            BASE + EPA + REST + (team_stats_cols or []) + nextgen_cols
        )

    pfr_cols = [c for c in PFR if c in raw.columns and c not in already_used]
    if pfr_cols:
        experiments["base+epa+rest+team_stats+nextgen+pfr"] = uniq(
            BASE
            + EPA
            + REST
            + (team_stats_cols or [])
            + (nextgen_cols or [])
            + pfr_cols
        )

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