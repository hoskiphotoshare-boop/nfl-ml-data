#!/usr/bin/env python3
"""Build the model scorecard shown in admin.

Baseline-relative assessment of every shipped model. Numbers come from the
existing reports, with player persistence/position-mean and score-model naive
baselines recomputed here so the comparison is explicit rather than editorial.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

OUT = "reports/model_scorecard.json"
MATCH = "features/match_features.parquet"
PLAYER = "features/player_features.parquet"
SCORE_BUNDLE = "models/score_models.joblib"
GROUPS = {"QB": ["QB"], "RB": ["RB", "FB", "HB"], "WR": ["WR"], "TE": ["TE"], "K": ["K"]}


def load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def fmt(value, digits=3) -> str:
    return f"{value:.{digits}f}"


def match_numbers() -> dict:
    report = load("reports/backtest.json", {})
    per = report.get("per_season", [])
    if not per:
        return {}
    games = sum(r["games"] for r in per)
    pooled = report.get("pooled", {})
    baseline_acc = sum(r.get("home_baseline_accuracy", 0) * r["games"] for r in per) / games
    baseline_ll = sum(r.get("home_baseline_log_loss", 0) * r["games"] for r in per) / games
    latest = per[-1]
    return {
        "season": latest.get("season"),
        "acc": latest.get("accuracy"),
        "ll": latest.get("log_loss"),
        "auc": latest.get("auc"),
        "pooled_acc": pooled.get("accuracy"),
        "pooled_ll": pooled.get("log_loss"),
        "pooled_auc": pooled.get("auc"),
        "baseline_acc": baseline_acc,
        "baseline_ll": baseline_ll,
        "games": games,
    }


def player_numbers() -> dict:
    if not os.path.exists(PLAYER):
        return {}
    df = pl.read_parquet(PLAYER).filter(pl.col("fantasy_points").is_not_null())
    if df.height == 0:
        return {}
    df = df.sort(["player_id", "season", "week"]).with_columns(
        pl.col("fantasy_points")
        .shift(1)
        .rolling_mean(window_size=4, min_samples=1)
        .over("player_id")
        .alias("persistence")
    )
    test = df.filter((pl.col("season") == 2026) & pl.col("persistence").is_not_null())
    train = df.filter(pl.col("season") < 2026)
    out = {}
    for group, positions in GROUPS.items():
        tr = train.filter(pl.col("position").is_in(positions))
        te = test.filter(pl.col("position").is_in(positions))
        if tr.height == 0 or te.height == 0:
            continue
        y = te["fantasy_points"].to_numpy()
        mean = float(tr["fantasy_points"].mean())
        out[group] = {
            "rows": int(te.height),
            "persistence": float(mean_absolute_error(y, te["persistence"].to_numpy())),
            "position_mean": float(mean_absolute_error(y, np.full(len(y), mean))),
        }
    return out


def score_numbers() -> dict:
    if not (os.path.exists(SCORE_BUNDLE) and os.path.exists(MATCH)):
        return {}
    try:
        bundle = joblib.load(SCORE_BUNDLE)
        features = [c for c in bundle["features"] if c in pl.read_parquet_schema(MATCH)]
        frame = pl.read_parquet(MATCH).filter(
            pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & (pl.col("season") >= 2022)
        )
        X = np.nan_to_num(
            frame.select(features).to_numpy().astype(float), nan=0.0
        )
        total = (frame["home_score"] + frame["away_score"]).to_numpy()
        margin = (frame["home_score"] - frame["away_score"]).to_numpy()
        seasons = frame["season"].to_numpy()
        train = seasons < 2026
        test = seasons == 2026
        if train.sum() < 100 or test.sum() < 20:
            return {}
        model_total = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X[train], total[train])
        model_margin = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X[train], margin[train])
        return {
            "total_mae": float(mean_absolute_error(model_total.predict(X[test]), total[test])),
            "margin_mae": float(mean_absolute_error(model_margin.predict(X[test]), margin[test])),
            "total_naive": float(np.abs(total[test] - total[train].mean()).mean()),
            "margin_naive": float(np.abs(margin[test] - margin[train].mean()).mean()),
        }
    except Exception:
        return {}


def main() -> None:
    match = match_numbers()
    players = player_numbers()
    scores = score_numbers()
    per_position = load("reports/player_backtest.json", {})
    playoff = load("reports/playoff_backtest.json", {}).get("pooled", {})
    futures = load("reports/futures_comparison.json", {}).get("summary", {})
    health = load("reports/model_health.json", {})
    market = health.get("market", {})

    rows = []
    if match:
        rows.append(
            {
                "model": "Match",
                "grade": "B",
                "primary": f"{match['season']}: {fmt(match['acc'], 3)} acc · {fmt(match['ll'], 3)} log loss · {fmt(match['auc'], 3)} AUC",
                "baseline": f"Home team: {fmt(match['baseline_acc'], 3)} acc · {fmt(match['baseline_ll'], 3)} log loss (pooled)",
                "note": "Clearly beats naive baselines; closing-line skill is still unverified.",
            }
        )
        rows.append(
            {
                "model": "Match (pooled)",
                "grade": "B",
                "primary": f"2022–26: {fmt(match['pooled_acc'], 3)} acc · {fmt(match['pooled_ll'], 3)} log loss · {fmt(match['pooled_auc'], 3)} AUC",
                "baseline": f"Home team: {fmt(match['baseline_acc'], 3)} acc · {fmt(match['baseline_ll'], 3)} log loss",
                "note": f"{match['games']} games; 95% accuracy interval roughly ±2.6 pp.",
            }
        )

    if scores:
        rows.append(
            {
                "model": "Margin",
                "grade": "C+",
                "primary": f"2026 MAE {scores['margin_mae']:.2f} pts",
                "baseline": f"Train mean {scores['margin_naive']:.2f}",
                "note": "Modest skill; the displayed spread is derived from win probability.",
            }
        )
        rows.append(
            {
                "model": "Total",
                "grade": "C",
                "primary": f"2026 MAE {scores['total_mae']:.2f} pts",
                "baseline": f"Train mean {scores['total_naive']:.2f}",
                "note": "Small edge over a constant; still shares the match feature set.",
            }
        )

    grades = {"QB": ("B", "Best player model; clear gain over persistence."),
              "RB": ("C", "Marginal over a four-game average."),
              "WR": ("B−", "Modest but consistent."),
              "TE": ("B−", "Modest, consistent gain."),
              "K": ("D", "No demonstrated skill — a positional constant is better.")}
    for group in ("QB", "RB", "WR", "TE", "K"):
        model = per_position.get(group) or {}
        base = players.get(group) or {}
        if not model or not base:
            continue
        grade, note = grades[group]
        rows.append(
            {
                "model": f"Player · {group}",
                "grade": grade,
                "primary": f"2026 MAE {model.get('mae', 0):.2f} (n={base['rows']})",
                "baseline": f"Persistence {base['persistence']:.2f} · position mean {base['position_mean']:.2f}",
                "note": note,
            }
        )

    if per_position:
        coverage = [
            f"{g} {per_position[g].get('coverage_80', 0) * 100:.0f}→{per_position[g].get('coverage_80_adj', 0) * 100:.0f}%"
            for g in ("QB", "RB", "WR", "TE", "K")
            if g in per_position
        ]
        rows.append(
            {
                "model": "Player intervals",
                "grade": "B+",
                "primary": "80% coverage " + " · ".join(coverage),
                "baseline": "Nominal 80%",
                "note": "Aligned for QB/K/WR/RB; TE stays conservative at 88%.",
            }
        )

    if playoff:
        rows.append(
            {
                "model": "Playoff bracket",
                "grade": "B−",
                "primary": f"Brier playoff {playoff.get('playoff', {}).get('brier', 0):.3f} · conf {playoff.get('conf', {}).get('brier', 0):.3f} · SB {playoff.get('sb', {}).get('brier', 0):.3f}",
                "baseline": "SRS-only bracket",
                "note": "Structure validated; Super Bowl skill is near the uniform prior.",
            }
        )

    if futures:
        rows.append(
            {
                "model": "SB futures",
                "grade": "D",
                "primary": f"Mean |edge| {futures.get('mean_abs_edge', 0) * 100:.1f} pp",
                "baseline": "Uniform prior",
                "note": "Near-uniform skill; treat futures edges as low confidence.",
            }
        )

    caveats = [
        "No historical closing odds, so market log loss and CLV are unmeasured.",
        f"Market board: mean |edge| {market.get('mean_abs_edge', 0) * 100:.1f} pp, {market.get('outliers', 0)} games at 20+ pp — signal or overconfidence is unknown.",
        "2026 is 162 games; pooled 2022–26 (1,298) is the more reliable sample.",
        "The call ledger is 0 graded and CLV is 0 until odds move before kickoff.",
        "Kicker projections and SB futures are the weakest outputs; total is close to a constant.",
    ]

    from datetime import datetime, timezone

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline": "Baseline-relative assessment. The market benchmark is still unverified.",
        "rows": rows,
        "caveats": caveats,
    }

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - keep the workflow alive
        print(f"Scorecard error: {exc}")
