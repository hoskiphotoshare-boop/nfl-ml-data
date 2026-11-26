#!/usr/bin/env python3
"""Permutation audit for the match features that drove CHI-DET.

Trains the shipped logistic recipe on 2022-2025, predicts 2026, then measures
how much 2026 log loss degrades when one feature column is shuffled. This is a
diagnostic; the shipped model is unchanged.

Output: reports/feature_audit.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

MATCH = "features/match_features.parquet"
MODEL = "models/match_model.joblib"
OUT = "reports/feature_audit.json"

FOCUS = [
    "home_qb_starts",
    "away_qb_starts",
    "home_prior_games",
    "away_prior_games",
    "blend_diff_diff",
    "home_blend_diff",
    "away_blend_diff",
    "home_passing_epa",
    "away_passing_epa",
    "home_win_pct",
    "away_win_pct",
    "home_inj_impact",
    "away_inj_impact",
]


def main() -> None:
    if not (os.path.exists(MATCH) and os.path.exists(MODEL)):
        print("Missing match features or model; skipping feature audit.")
        return

    bundle = joblib.load(MODEL)
    features = [c for c in bundle["features"] if c in pl.read_parquet_schema(MATCH)]
    frame = pl.read_parquet(MATCH).filter(pl.col("home_win").is_not_null())
    train = frame.filter(pl.col("season") < 2026)
    test = frame.filter(pl.col("season") == 2026)
    if train.height < 500 or test.height < 100 or not features:
        print("Not enough data for the 2026 feature audit.")
        return

    def matrix(data):
        return data.select(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in features]
        ).to_numpy()

    x_train, y_train = matrix(train), train["home_win"].cast(pl.Int8).to_numpy()
    x_test, y_test = matrix(test), test["home_win"].cast(pl.Int8).to_numpy()
    model = LogisticRegression(max_iter=4000, C=0.03)
    model.fit(x_train, y_train)
    base_prob = model.predict_proba(x_test)[:, 1]
    base = {
        "log_loss": round(float(log_loss(y_test, base_prob)), 4),
        "accuracy": round(float(accuracy_score(y_test, base_prob >= 0.5)), 4),
        "auc": round(float(roc_auc_score(y_test, base_prob)), 4),
        "n": int(len(y_test)),
    }

    rng = np.random.default_rng(42)
    importances = {}
    for name in FOCUS:
        if name not in features:
            continue
        index = features.index(name)
        deltas = []
        for _ in range(12):
            shuffled = x_test.copy()
            rng.shuffle(shuffled[:, index])
            prob = model.predict_proba(shuffled)[:, 1]
            deltas.append(float(log_loss(y_test, prob)) - base["log_loss"])
        importances[name] = {
            "mean_delta_log_loss": round(float(np.mean(deltas)), 5),
            "std": round(float(np.std(deltas)), 5),
            "coef": round(float(model.coef_[0][index]), 4),
        }

    ranked = sorted(
        importances.items(),
        key=lambda item: item[1]["mean_delta_log_loss"],
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "logistic C=0.03 train 2022-2025, permutation importance on 2026",
        "baseline_2026": base,
        "features": dict(ranked),
        "note": "Positive mean delta = shuffling the feature hurts log loss, i.e. the model used it. Diagnostic only.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Baseline 2026: LL {base['log_loss']} · acc {base['accuracy']} · AUC {base['auc']}")
    for name, row in ranked[:6]:
        print(f"  {name:20s} ΔLL {row['mean_delta_log_loss']:+.5f} (coef {row['coef']:+.4f})")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Feature audit error: {exc}")
