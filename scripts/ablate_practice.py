#!/usr/bin/env python3
"""Research: does practice participation add match-model signal?

Builds point-in-time team-week practice features from the injury report
(DNP / limited / out / doubtful / questionable, with QB/OL/defence splits),
merges them into the match feature table and runs the same walk-forward
logistic setup used by the shipped model. The shipped model is not changed:
this is a gate report, not a promotion.

Output: reports/practice_ablation.json
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MATCH = "features/match_features.parquet"
INJURIES = "data/injuries_latest.parquet"
MODEL = "models/match_model.joblib"
OUT = "reports/practice_ablation.json"
TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
QB = "QB"
OL = {"T", "G", "C", "OT", "OG", "OL"}
DEF = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB"}


def practice_features() -> pl.DataFrame:
    injuries = pl.read_parquet(INJURIES).filter(pl.col("season") >= 2022)
    injuries = injuries.with_columns(
        pl.col("team").replace(TEAM_ALIASES).alias("team")
    )
    injuries = injuries.sort("date_modified").group_by(
        ["season", "week", "gsis_id"], maintain_order=True
    ).agg(pl.all().last())

    dnp = pl.col("practice_status").fill_null("").str.contains("Did Not Participate")
    limited = pl.col("practice_status").fill_null("").str.contains("Limited Participation")
    report = pl.col("report_status").fill_null("")
    position = pl.col("position").fill_null("")

    return injuries.group_by(["season", "week", "team"]).agg(
        pl.len().alias("report_rows"),
        dnp.sum().alias("dnp"),
        limited.sum().alias("limited"),
        (report == "Out").sum().alias("out"),
        (report == "Doubtful").sum().alias("doubtful"),
        (report == "Questionable").sum().alias("questionable"),
        ((position == QB) & dnp).sum().alias("qb_dnp"),
        (position.is_in(list(OL)) & dnp).sum().alias("ol_dnp"),
        (position.is_in(list(DEF)) & dnp).sum().alias("def_dnp"),
    )


def add_team_features(df: pl.DataFrame, practice: pl.DataFrame, side: str) -> pl.DataFrame:
    prefix = f"{side}_"
    team_col = f"{side}_team"
    renamed = practice.rename(
        {
            "team": team_col,
            **{col: f"{prefix}practice_{col}" for col in practice.columns if col not in ("season", "week", "team")},
        }
    )
    out = df.join(renamed, on=["season", "week", team_col], how="left")
    count = f"{prefix}practice_report_rows"
    out = out.with_columns(
        pl.col(count).is_null().cast(pl.Int8).alias(f"{prefix}practice_missing")
    )
    for col in renamed.columns:
        if col in ("season", "week", team_col):
            continue
        out = out.with_columns(pl.col(col).fill_null(0))
    return out


def metrics(y, prob) -> dict:
    return {
        "n": int(len(y)),
        "log_loss": round(float(log_loss(y, prob)), 4),
        "accuracy": round(float(accuracy_score(y, prob >= 0.5)), 4),
        "auc": round(float(roc_auc_score(y, prob)), 4),
    }


def main() -> None:
    if not (os.path.exists(MATCH) and os.path.exists(INJURIES) and os.path.exists(MODEL)):
        print("Missing match features, injuries or model; skipping practice ablation.")
        return

    df = pl.read_parquet(MATCH)
    practice = practice_features()
    df = add_team_features(df, practice, "home")
    df = add_team_features(df, practice, "away")
    df = df.with_columns(
        pl.col("home_practice_report_rows").fill_null(0).clip(1).alias(
            "_home_report_rows"
        ),
        pl.col("away_practice_report_rows").fill_null(0).clip(1).alias(
            "_away_report_rows"
        ),
    ).with_columns(
        (pl.col("home_practice_dnp") / pl.col("_home_report_rows")).alias(
            "home_practice_dnp_rate"
        ),
        (pl.col("away_practice_dnp") / pl.col("_away_report_rows")).alias(
            "away_practice_dnp_rate"
        ),
        (pl.col("home_practice_out") / pl.col("_home_report_rows")).alias(
            "home_practice_out_rate"
        ),
        (pl.col("away_practice_out") / pl.col("_away_report_rows")).alias(
            "away_practice_out_rate"
        ),
    ).with_columns(
        (pl.col("home_practice_dnp_rate") - pl.col("away_practice_dnp_rate")).alias(
            "practice_dnp_diff"
        ),
        (pl.col("home_practice_out_rate") - pl.col("away_practice_out_rate")).alias(
            "practice_out_diff"
        ),
    )

    bundle = joblib.load(MODEL)
    base_features = [c for c in bundle["features"] if c in df.columns]
    extra = [
        "home_practice_dnp_rate",
        "away_practice_dnp_rate",
        "practice_dnp_diff",
        "home_practice_out_rate",
        "away_practice_out_rate",
        "practice_out_diff",
        "home_practice_qb_dnp",
        "away_practice_qb_dnp",
        "home_practice_ol_dnp",
        "away_practice_ol_dnp",
        "home_practice_def_dnp",
        "away_practice_def_dnp",
        "home_practice_missing",
        "away_practice_missing",
    ]
    extra = [c for c in extra if c in df.columns]

    def matrix(frame, cols):
        return frame.select(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in cols]
        ).to_numpy()

    seasons = sorted(
        int(x)
        for x in df.filter(
            pl.col("home_win").is_not_null()
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )["season"].unique().to_list()
        if int(x) >= 2022
    )

    results = {"baseline": {}, "with_practice": {}}
    pooled = {"baseline": [], "with_practice": []}
    paired = {"y": [], "baseline": [], "practice": []}
    for season in seasons[1:]:
        train = df.filter((pl.col("season") < season) & pl.col("home_win").is_not_null())
        test = df.filter((pl.col("season") == season) & pl.col("home_win").is_not_null())
        if train.height < 150 or test.height < 20:
            continue
        y_train = train["home_win"].cast(pl.Int8).to_numpy()
        y_test = test["home_win"].cast(pl.Int8).to_numpy()
        for label, columns in (
            ("baseline", base_features),
            ("with_practice", base_features + extra),
        ):
            if not columns:
                continue
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=4000, C=0.03)
            )
            model.fit(matrix(train, columns), y_train)
            prob = model.predict_proba(matrix(test, columns))[:, 1]
            result = metrics(y_test, prob)
            result["season"] = season
            results[label][season] = result
            pooled[label].append(result)
            if label == "baseline":
                paired["y"].extend(y_test.tolist())
                paired["baseline"].extend(prob.tolist())
            else:
                paired["practice"].extend(prob.tolist())

    if not pooled["baseline"] or not pooled["with_practice"]:
        print("Not enough seasons for a walk-forward practice test.")
        return

    def weighted(rows):
        n = sum(r["n"] for r in rows)
        return {
            "n": n,
            "log_loss": round(sum(r["log_loss"] * r["n"] for r in rows) / n, 4),
            "accuracy": round(sum(r["accuracy"] * r["n"] for r in rows) / n, 4),
            "auc": round(sum(r["auc"] * r["n"] for r in rows) / n, 4),
        }

    base = weighted(pooled["baseline"])
    with_practice = weighted(pooled["with_practice"])
    delta = {
        "log_loss": round(base["log_loss"] - with_practice["log_loss"], 4),
        "accuracy": round(with_practice["accuracy"] - base["accuracy"], 4),
        "auc": round(with_practice["auc"] - base["auc"], 4),
    }
    accepted = delta["log_loss"] >= 0.005 and delta["accuracy"] >= -0.002
    covered = df.filter(pl.col("season") >= 2022)
    coverage = {
        "team_weeks": int(practice.height),
        "games": int(covered.height),
        "home_missing_rate": round(
            float(covered["home_practice_missing"].mean() or 0.0), 4
        ),
        "away_missing_rate": round(
            float(covered["away_practice_missing"].mean() or 0.0), 4
        ),
        "seasons": [int(x) for x in sorted(practice["season"].unique().to_list())],
        "by_season": {
            str(int(season)): round(
                float(
                    covered.filter(pl.col("season") == season)[
                        "home_practice_missing"
                    ].mean()
                    or 0.0
                ),
                4,
            )
            for season in sorted(covered["season"].unique().to_list())
        },
    }

    # Paired bootstrap over test games: is the log-loss gain distinguishable
    # from zero? Positive = practice model better.
    y_all = np.asarray(paired["y"], dtype=float)
    p_base = np.clip(np.asarray(paired["baseline"], dtype=float), 1e-6, 1 - 1e-6)
    p_prac = np.clip(np.asarray(paired["practice"], dtype=float), 1e-6, 1 - 1e-6)
    loss_base = -(y_all * np.log(p_base) + (1 - y_all) * np.log(1 - p_base))
    loss_prac = -(y_all * np.log(p_prac) + (1 - y_all) * np.log(1 - p_prac))
    per_game_delta = loss_base - loss_prac
    rng = np.random.default_rng(42)
    boot = [
        float(np.mean(per_game_delta[rng.integers(0, len(per_game_delta), len(per_game_delta))]))
        for _ in range(2000)
    ]
    delta["log_loss_ci95"] = [
        round(float(np.quantile(boot, 0.025)), 4),
        round(float(np.quantile(boot, 0.975)), 4),
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": "walk-forward match-model ablation: injury practice participation",
        "features": extra,
        "coverage": coverage,
        "baseline": base,
        "with_practice": with_practice,
        "delta": delta,
        "accepted": bool(accepted),
        "gate": "accept only if log loss improves by >= 0.005 and accuracy does not drop by > 0.002",
        "per_season": results,
        "point_in_time": "Uses the final weekly injury report; a rigorous version would filter date_modified < kickoff per game.",
        "note": "Research only. The shipped match model is unchanged; this report gates promotion.",
    }
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"Practice ablation: baseline LL {base['log_loss']} -> {with_practice['log_loss']} "
        f"({delta['log_loss']:+.4f}), acc {delta['accuracy']:+.4f}, accepted={accepted}"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Practice ablation error: {exc}")
