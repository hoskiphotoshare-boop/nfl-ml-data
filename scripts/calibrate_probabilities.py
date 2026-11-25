#!/usr/bin/env python3
"""Fit an out-of-sample probability calibration for the match model.

Walk-forward: train on seasons before each test season, predict that season,
then fit Platt scaling on the pooled out-of-fold predictions. Writes
``models/platt.json``; ``predict_upcoming.py`` and the playoff simulator apply
it when it improves log loss.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import joblib

FEATURES = "features/match_features.parquet"
BUNDLE = "models/match_model.joblib"
OUTPUT = "models/platt.json"
TEST_SEASONS = (2022, 2023, 2024, 2025, 2026)


def logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))


def main() -> None:
    if not (os.path.exists(FEATURES) and os.path.exists(BUNDLE)):
        print("Missing match features or model; skipping calibration.")
        return
    bundle = joblib.load(BUNDLE)
    features = bundle["features"]
    frame = pl.read_parquet(FEATURES)
    X = np.nan_to_num(frame.select(features).to_numpy(), nan=0.0)
    y = frame["home_win"].to_numpy()
    seasons = frame["season"].to_numpy()

    predictions = []
    outcomes = []
    for season in TEST_SEASONS:
        train = seasons < season
        test = seasons == season
        if train.sum() < 100 or test.sum() < 20:
            continue
        model = LogisticRegression(max_iter=3000)
        model.fit(X[train], y[train])
        predictions.append(model.predict_proba(X[test])[:, 1])
        outcomes.append(y[test])

    if not predictions:
        print("No out-of-fold predictions; skipping calibration.")
        return

    prob = np.concatenate(predictions)
    outcome = np.concatenate(outcomes)
    platt = LogisticRegression().fit(logit(prob).reshape(-1, 1), outcome)
    a = float(platt.coef_[0][0])
    b = float(platt.intercept_[0])
    calibrated = 1.0 / (1.0 + np.exp(-(a * logit(prob) + b)))

    raw_loss = log_loss(outcome, prob)
    cal_loss = log_loss(outcome, calibrated)
    raw_brier = brier_score_loss(outcome, prob)
    cal_brier = brier_score_loss(outcome, calibrated)
    enabled = cal_loss + 0.001 < raw_loss

    payload = {
        "a": round(a, 6),
        "b": round(b, 6),
        "n": int(len(outcome)),
        "enabled": bool(enabled),
        "raw_log_loss": round(float(raw_loss), 6),
        "calibrated_log_loss": round(float(cal_loss), 6),
        "raw_brier": round(float(raw_brier), 6),
        "calibrated_brier": round(float(cal_brier), 6),
        "seasons": [s for s in TEST_SEASONS],
    }
    os.makedirs("models", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Platt a={a:.3f} b={b:.3f} n={len(outcome)} "
        f"logloss {raw_loss:.4f}->{cal_loss:.4f} "
        f"brier {raw_brier:.4f}->{cal_brier:.4f} enabled={enabled}"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Calibration error: {exc}")
