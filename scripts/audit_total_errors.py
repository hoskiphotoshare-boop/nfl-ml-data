#!/usr/bin/env python3
"""Walk-forward total-model residual audit.

Refits the shipped total-regressor recipe on prior seasons and predicts each
test season, so the error distribution is out-of-sample rather than the
shipped model's in-sample fit. Diagnostic only; no retrain is promoted.

Output: reports/total_errors.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

MATCH = "features/match_features.parquet"
MODEL = "models/score_models.joblib"
OUT = "reports/total_errors.json"


def main() -> None:
    if not (os.path.exists(MATCH) and os.path.exists(MODEL)):
        print("Missing match features or score models; skipping total audit.")
        return

    import joblib

    bundle = joblib.load(MODEL)
    features = [c for c in bundle["features"] if c in pl.read_parquet_schema(MATCH)]
    frame = pl.read_parquet(MATCH).filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & pl.col("season").is_not_null()
    )
    if frame.height == 0 or not features:
        print("No scored games or features; skipping total audit.")
        return

    x = np.nan_to_num(
        frame.select(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in features]
        ).to_numpy(),
        nan=0.0,
    )
    actual = (frame["home_score"] + frame["away_score"]).to_numpy().astype(float)
    seasons = frame["season"].to_numpy()

    residuals = []
    per_season = {}
    for test_season in range(2023, int(seasons.max()) + 1):
        train_mask = seasons < test_season
        test_mask = seasons == test_season
        if train_mask.sum() < 500 or test_mask.sum() < 40:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
        )
        model.fit(x[train_mask], actual[train_mask])
        predicted = model.predict(x[test_mask])
        residual = actual[test_mask] - predicted
        residuals.append(residual)
        error = np.abs(residual)
        per_season[str(test_season)] = {
            "n": int(error.size),
            "mae": round(float(error.mean()), 3),
            "rmse": round(float(np.sqrt((residual**2).mean())), 3),
            "bias": round(float(residual.mean()), 3),
            "abs_p50": round(float(np.quantile(error, 0.50)), 2),
            "abs_p75": round(float(np.quantile(error, 0.75)), 2),
            "abs_p90": round(float(np.quantile(error, 0.90)), 2),
            "abs_p95": round(float(np.quantile(error, 0.95)), 2),
            "share_over_14": round(float((error > 14).mean()), 4),
            "share_over_20": round(float((error > 20).mean()), 4),
            "share_over_25": round(float((error > 25).mean()), 4),
        }

    if not residuals:
        print("Not enough seasons for a walk-forward total audit.")
        return

    all_residual = np.concatenate(residuals)
    error = np.abs(all_residual)
    overall = {
        "n": int(error.size),
        "mae": round(float(error.mean()), 3),
        "rmse": round(float(np.sqrt((all_residual**2).mean())), 3),
        "bias": round(float(all_residual.mean()), 3),
        "abs_p50": round(float(np.quantile(error, 0.50)), 2),
        "abs_p75": round(float(np.quantile(error, 0.75)), 2),
        "abs_p90": round(float(np.quantile(error, 0.90)), 2),
        "abs_p95": round(float(np.quantile(error, 0.95)), 2),
        "share_over_14": round(float((error > 14).mean()), 4),
        "share_over_20": round(float((error > 20).mean()), 4),
        "share_over_25": round(float((error > 25).mean()), 4),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "walk-forward HGB: train on prior seasons, predict test season",
        "test_seasons": sorted(int(s) for s in per_season),
        "all": overall,
        "per_season": per_season,
        "note": "Out-of-sample residuals. One 20+ point game is not a retrain signal; watch the share over 20 and p95 over a season.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"Total walk-forward: MAE {overall['mae']} · RMSE {overall['rmse']} · "
        f"|error|>20 {overall['share_over_20'] * 100:.1f}% · p95 {overall['abs_p95']}"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Total audit error: {exc}")
