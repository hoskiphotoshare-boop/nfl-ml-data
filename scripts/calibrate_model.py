#!/usr/bin/env python3
"""Calibrate model win probabilities against market-implied probabilities.

Fits a simple linear map on the logit scale:

    logit(market) = a * logit(model) + b

and writes ``models/calibration.json``.  Also adds
``calibrated_home_win_prob`` to ``predictions/upcoming_games.json`` when
available.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    path = "reports/market_comparison.json"
    if not os.path.exists(path):
        print("No market comparison report; skipping calibration.")
        return

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    games = [
        g
        for g in report.get("games", [])
        if g.get("model_home_win_prob") is not None
        and g.get("market_home_win_prob") is not None
    ]
    if len(games) < 5:
        print(f"Only {len(games)} comparable games; skipping calibration.")
        return

    X = np.array([[logit(g["model_home_win_prob"])] for g in games])
    y = np.array([logit(g["market_home_win_prob"]) for g in games])

    model = LinearRegression()
    model.fit(X, y)
    a = float(model.coef_[0])
    b = float(model.intercept_)

    before = [g["model_home_win_prob"] for g in games]
    market = [g["market_home_win_prob"] for g in games]
    after = [sigmoid(a * logit(p) + b) for p in before]

    rmse_before = float(np.sqrt(np.mean((np.array(before) - np.array(market)) ** 2)))
    rmse_after = float(np.sqrt(np.mean((np.array(after) - np.array(market)) ** 2)))

    # Out-of-fold estimate so we only enable calibration when it generalises.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(games))
    for train_idx, test_idx in kf.split(X):
        fold = LinearRegression().fit(X[train_idx], y[train_idx])
        oof[test_idx] = fold.predict(X[test_idx])
    oof_prob = np.array([sigmoid(v) for v in oof])
    rmse_cv = float(np.sqrt(np.mean((oof_prob - np.array(market)) ** 2)))
    enabled = rmse_cv + 0.001 < rmse_before

    payload = {
        "a": round(a, 6),
        "b": round(b, 6),
        "n": len(games),
        "enabled": bool(enabled),
        "rmse_before": round(rmse_before, 6),
        "rmse_cv": round(rmse_cv, 6),
        "rmse_after": round(rmse_after, 6),
    }

    os.makedirs("models", exist_ok=True)
    with open("models/calibration.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    upcoming_path = "predictions/upcoming_games.json"
    if os.path.exists(upcoming_path):
        with open(upcoming_path, encoding="utf-8") as f:
            upcoming = json.load(f)
        for game in upcoming:
            if enabled:
                p = game.get("home_win_prob")
                if p is not None:
                    game["calibrated_home_win_prob"] = round(
                        sigmoid(a * logit(p) + b), 3
                    )
            else:
                game.pop("calibrated_home_win_prob", None)
        with open(upcoming_path, "w", encoding="utf-8") as f:
            json.dump(upcoming, f, indent=2)

    print(
        f"Calibration: a={a:.3f} b={b:.3f} n={len(games)} "
        f"RMSE before={rmse_before:.4f} cv={rmse_cv:.4f} after={rmse_after:.4f} "
        f"enabled={enabled}"
    )
    print("Wrote models/calibration.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Calibration error: {exc}")
    sys.exit(0)