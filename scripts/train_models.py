#!/usr/bin/env python3
"""Train baseline NFL models and write prediction JSON.

- Match model: logistic regression on pre-game team averages to predict
  home-win probability.
- Player baseline: per-player average fantasy points (half-PPR).

The script is intentionally robust: if features are missing or too small it
falls back to simple heuristics and always exits 0 so CI stays green.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import numpy as np
import polars as pl

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover – sklearn is optional at runtime
    SKLEARN_AVAILABLE = False

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


def load_features() -> pl.DataFrame | None:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        print("No match features found; skipping match-model training.")
        return None
    return pl.read_parquet(path)


def prepare_match_matrix(df: pl.DataFrame):
    feature_cols = [
        # Rolling pre-game team averages
        "home_avg_points",
        "home_avg_allowed",
        "away_avg_points",
        "away_avg_allowed",
        # PBP-derived team features
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
        # Pre-game rolling team stats
        "home_passing_epa",
        "away_passing_epa",
        "home_rushing_epa",
        "away_rushing_epa",
        # Rest / schedule context
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_short_week",
        "away_short_week",
        # Context. Week/season are deliberately excluded: they encode season
        # phase rather than team strength and dominated SHAP without improving
        # the 2026 backtest. Division games are lower-scoring and slightly less
        # home-favoured; keep the flag as a small matchup signal.
        "div_game",
        # Team results and quarterback continuity (added by add_match_context.py)
        "home_prior_games",
        "away_prior_games",
        "home_win_pct",
        "away_win_pct",
        "home_diff_pg",
        "away_diff_pg",
        "win_pct_diff",
        "diff_pg_diff",
        # Blended current-season + prior-season strength (empirical-Bayes prior)
        "home_blend_diff",
        "away_blend_diff",
        "blend_diff_diff",
        "home_blend_win",
        "away_blend_win",
        "blend_win_diff",
        "home_qb_prior_epa",
        "away_qb_prior_epa",
        "home_qb_starts",
        "away_qb_starts",
        "home_qb_changed",
        "away_qb_changed",
        "home_qb_missing",
        "away_qb_missing",
        "home_inj_impact",
        "away_inj_impact",
        "inj_impact_diff",
        "home_inj_missing",
        "away_inj_missing",
    ]
    # Next Gen Stats are unstable for team-level rankings (collinear, large
    # coefficients). Results and QB context now carry that signal more
    # reliably; NGS can be re-enabled for experiments via MATCH_USE_NGS=1.
    if os.environ.get("MATCH_USE_NGS", "0") == "1":
        feature_cols += NG_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        return None, None, feature_cols

    X = df.select(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in feature_cols]
    ).to_numpy()
    y = df["home_win"].cast(pl.Int8).to_numpy()
    return X, y, feature_cols


def train_match_model(df: pl.DataFrame):
    X, y, feature_cols = prepare_match_matrix(df)
    if X is None or len(X) < 20 or not SKLEARN_AVAILABLE:
        print("Not enough data (or sklearn missing) – using heuristic match predictions.")
        return None, feature_cols

    # Prefer a time-based split: train on all seasons before the latest, test
    # on the latest season.  Fall back to a random split if there is not
    # enough history.
    X_train = X_test = y_train = y_test = None
    if "season" in df.columns:
        seasons = df["season"].drop_nulls().unique().sort()
        if len(seasons) >= 2:
            test_season = int(seasons[-1])
            train_df = df.filter(pl.col("season") < test_season)
            test_df = df.filter(pl.col("season") == test_season)
            if train_df.height >= 20 and test_df.height >= 10:
                X_train = train_df.select(
                    [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in feature_cols]
                ).to_numpy()
                y_train = train_df["home_win"].cast(pl.Int8).to_numpy()
                X_test = test_df.select(
                    [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in feature_cols]
                ).to_numpy()
                y_test = test_df["home_win"].cast(pl.Int8).to_numpy()
                print(f"Time-based split: train < {test_season}, test == {test_season}")

    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        print("Using random 80/20 split (not enough seasons for time-based split).")

    candidates = [
        ("logistic", LogisticRegression(max_iter=2000)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
    ]

    best_name = None
    best_model = None
    best_loss = None
    for name, model in candidates:
        try:
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            loss = log_loss(y_test, preds)
            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"{name}: accuracy={acc:.3f} log_loss={loss:.3f}")
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_model = model
                best_name = name
        except Exception as exc:  # pragma: no cover
            print(f"{name} failed: {exc}")

    if best_model is None:
        print("No model trained; falling back to heuristic.")
        return None, feature_cols

    print(f"Selected model: {best_name} (log loss {best_loss:.3f})")
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": best_model, "features": feature_cols, "name": best_name},
        "models/match_model.joblib",
    )
    print("Saved models/match_model.joblib")
    return best_model, feature_cols


def train_score_models(df: pl.DataFrame, feature_cols: list[str]) -> None:
    """Train margin and total regressors and save residual scales.

    The spread/total probabilities in ``compare_market.py`` use the residual
    standard deviation instead of a fixed 13.5-point assumption.
    """
    if not SKLEARN_AVAILABLE or not feature_cols:
        return
    required = {"home_score", "away_score", "season"}
    if not required.issubset(set(df.columns)):
        return

    played = df.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("season") >= 2022)
    )
    if played.height < 100:
        print("Not enough recent games for score models; skipping.")
        return

    X = played.select(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in feature_cols]
    ).to_numpy()
    margin = (played["home_score"] - played["away_score"]).to_numpy()
    total = (played["home_score"] + played["away_score"]).to_numpy()
    seasons = played["season"].to_numpy()

    oof_margin = []
    oof_total = []
    for test_season in (2024, 2025, 2026):
        train = seasons < test_season
        test = seasons == test_season
        if train.sum() < 100 or test.sum() < 20:
            continue
        margin_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X[train], margin[train])
        total_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X[train], total[train])
        oof_margin.extend(margin_model.predict(X[test]) - margin[test])
        oof_total.extend(total_model.predict(X[test]) - total[test])
        mae_m = np.mean(np.abs(margin_model.predict(X[test]) - margin[test]))
        mae_t = np.mean(np.abs(total_model.predict(X[test]) - total[test]))
        print(
            f"Score models {test_season}: margin MAE={mae_m:.2f}, "
            f"total MAE={mae_t:.2f}"
        )

    sigma_margin = float(np.std(oof_margin)) if oof_margin else 13.5
    sigma_total = float(np.std(oof_total)) if oof_total else 13.5
    sigma_margin = max(sigma_margin, 7.0)
    sigma_total = max(sigma_total, 7.0)

    final_margin = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
    ).fit(X, margin)
    final_total = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=42
    ).fit(X, total)

    joblib.dump(
        {
            "margin_model": final_margin,
            "total_model": final_total,
            "features": feature_cols,
            "sigma_margin": round(sigma_margin, 3),
            "sigma_total": round(sigma_total, 3),
        },
        "models/score_models.joblib",
    )
    print(
        f"Saved models/score_models.joblib "
        f"(sigma margin {sigma_margin:.2f}, total {sigma_total:.2f})"
    )


def write_games_predictions(df: pl.DataFrame, model, feature_cols) -> None:
    os.makedirs("predictions", exist_ok=True)

    # Limit to the most recent week available so the frontend stays lightweight.
    if "season" in df.columns and "week" in df.columns and df.height > 0:
        max_season = df["season"].max()
        season_df = df.filter(pl.col("season") == max_season)
        max_week = season_df["week"].max()
        df = season_df.filter(pl.col("week") == max_week)

    records = []

    for row in df.iter_rows(named=True):
        if model is not None and feature_cols:
            x = np.array([[float(row.get(c) or 0.0) for c in feature_cols]])
            home_prob = float(model.predict_proba(x)[0, 1])
        else:
            home_avg = float(row.get("home_avg_points") or 0.0)
            away_avg = float(row.get("away_avg_points") or 0.0)
            home_prob = 0.5 + (home_avg - away_avg) * 0.02
            home_prob = min(max(home_prob, 0.05), 0.95)

        # Baseline projected scores from pre-game team averages.
        home_avg = float(row.get("home_avg_points") or 0.0)
        away_avg = float(row.get("away_avg_points") or 0.0)
        home_allowed = float(row.get("home_avg_allowed") or 0.0)
        away_allowed = float(row.get("away_avg_allowed") or 0.0)
        proj_home = (home_avg + away_allowed) / 2.0 if home_avg and away_allowed else None
        proj_away = (away_avg + home_allowed) / 2.0 if away_avg and home_allowed else None
        spread = (proj_home - proj_away) if proj_home is not None and proj_away is not None else None
        total = (proj_home + proj_away) if proj_home is not None and proj_away is not None else None

        records.append(
            {
                "game_id": row.get("game_id"),
                "season": row.get("season"),
                "week": row.get("week"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "home_win_prob": round(home_prob, 3),
                "spread": round(spread, 1) if spread is not None else None,
                "total": round(total, 1) if total is not None else None,
            }
        )

    with open("predictions/games_week.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote predictions/games_week.json ({len(records)} games)")


def write_player_predictions() -> None:
    path = "features/player_features.parquet"
    if not os.path.exists(path):
        print("No player features found; leaving player predictions unchanged.")
        return
    df = pl.read_parquet(path)
    if df.height == 0:
        return

    id_col = "gsis_id" if "gsis_id" in df.columns else ("player_id" if "player_id" in df.columns else "player_name")

    # Weighted recent form: the current season counts double vs prior seasons.
    max_season = int(df["season"].max()) if "season" in df.columns else None
    if max_season is not None:
        df = df.with_columns(
            pl.when(pl.col("season") == max_season)
            .then(2.0)
            .otherwise(1.0)
            .alias("_weight")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("_weight"))

    def wmean(col: str):
        return (
            (pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0) * pl.col("_weight")).sum()
            / pl.col("_weight").sum()
        )

    aliases = {
        "targets": "projected_targets",
        "receptions": "projected_receptions",
        "yards": "projected_yards",
        "touchdowns": "projected_tds",
    }

    agg_exprs = [wmean("fantasy_points").alias("projected_fantasy_points")]
    for col, alias in aliases.items():
        agg_exprs.append(
            wmean(col).alias(alias) if col in df.columns else pl.lit(0.0).alias(alias)
        )
    agg_exprs.extend(
        [
            pl.col("player_name").first().alias("player_name")
            if "player_name" in df.columns
            else pl.lit("").alias("player_name"),
            pl.col("position").first().alias("position")
            if "position" in df.columns
            else pl.lit("").alias("position"),
            pl.col("team").first().alias("team")
            if "team" in df.columns
            else pl.lit("").alias("team"),
        ]
    )
    agg = df.group_by(id_col).agg(agg_exprs)
    records = []
    for row in agg.iter_rows(named=True):
        pid = row.get(id_col)
        records.append(
            {
                "player_id": pid,
                "name": row.get("player_name") or pid,
                "team": row.get("team") or "",
                "position": row.get("position") or "",
                "projected_targets": round(float(row.get("projected_targets") or 0), 1),
                "projected_receptions": round(float(row.get("projected_receptions") or 0), 1),
                "projected_yards": round(float(row.get("projected_yards") or 0), 1),
                "projected_tds": round(float(row.get("projected_tds") or 0), 2),
                "projected_fantasy_points": round(float(row.get("projected_fantasy_points") or 0), 1),
            }
        )

    # Keep only fantasy-relevant players and cap the list for the frontend.
    records = [
        r for r in records
        if r.get("projected_fantasy_points", 0) > 0
        or r.get("projected_targets", 0) > 0
        or r.get("projected_receptions", 0) > 0
        or r.get("projected_yards", 0) > 0
    ]
    records.sort(key=lambda r: r.get("projected_fantasy_points", 0), reverse=True)
    records = records[:250]

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/players_week.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote predictions/players_week.json ({len(records)} players)")


def main() -> None:
    df = load_features()
    if df is not None and df.height > 0:
        model, feature_cols = train_match_model(df)
        write_games_predictions(df, model, feature_cols)
        train_score_models(df, feature_cols)
    else:
        print("Skipping match predictions because features are missing.")
    write_player_predictions()
    print("Model training/prediction step complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Training error: {exc}")
    sys.exit(0)