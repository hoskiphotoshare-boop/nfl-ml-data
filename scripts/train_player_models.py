#!/usr/bin/env python3
"""Train per-position player fantasy models and write player projections.

Builds pre-game rolling player features from ``features/player_features.parquet``,
trains a HistGradientBoosting regressor per position group, evaluates on the
latest season, and writes ``predictions/players_week.json``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import joblib
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

WINDOW = 4
POSITION_GROUPS = {
    "QB": ["QB"],
    "RB": ["RB", "FB", "HB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "K": ["K"],
}
ROLL_COLS = [
    "fantasy_points",
    "targets",
    "receptions",
    "yards",
    "touchdowns",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "attempts",
    "completions",
    "fg_made",
    "fg_att",
    "pat_made",
    "pat_att",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]


def load_player_features() -> pl.DataFrame:
    path = "features/player_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def add_position_group(df: pl.DataFrame) -> pl.DataFrame:
    expr = pl.lit("OTHER")
    for group, positions in POSITION_GROUPS.items():
        expr = (
            pl.when(pl.col("position").is_in(positions))
            .then(pl.lit(group))
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("pos_group"))


def enrich_defense_rating(df: pl.DataFrame) -> pl.DataFrame:
    if "pos_group" not in df.columns:
        return df

    agg = (
        df.filter(pl.col("fantasy_points").is_not_null())
        .group_by(["season", "week", "opponent_team", "pos_group"])
        .agg(pl.col("fantasy_points").mean().alias("pts_allowed"))
        .sort(["opponent_team", "pos_group", "season", "week"])
    )
    agg = agg.with_columns(
        pl.col("pts_allowed")
        .shift(1)
        .rolling_mean(window_size=4, min_samples=1)
        .over(["opponent_team", "pos_group"])
        .alias("opp_def_rating")
    )
    means = agg.group_by("pos_group").agg(pl.col("pts_allowed").mean().alias("_mean"))
    agg = (
        agg.join(means, on="pos_group", how="left")
        .with_columns(pl.col("opp_def_rating").fill_null(pl.col("_mean")))
        .drop("_mean")
    )
    return df.join(
        agg.select("season", "week", "opponent_team", "pos_group", "opp_def_rating"),
        on=["season", "week", "opponent_team", "pos_group"],
        how="left",
    )


def enrich_context(df: pl.DataFrame) -> pl.DataFrame:
    path = None
    for candidate in ["data/schedules.parquet", "data/schedules.csv"]:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        return df

    sched = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
    date_col = "gameday" if "gameday" in sched.columns else ("date" if "date" in sched.columns else None)
    if date_col is None or "game_id" not in sched.columns:
        return df

    sched = sched.with_columns(
        pl.col(date_col).cast(pl.Utf8).str.to_date(strict=False).alias("_game_date")
    ).select("game_id", "home_team", "away_team", "_game_date")

    out = df.join(sched, on="game_id", how="left")
    out = out.with_columns(
        (pl.col("team") == pl.col("home_team")).cast(pl.Int8).alias("is_home")
    )

    home = sched.select(
        "game_id", pl.col("home_team").alias("team"), pl.col("_game_date").alias("game_date")
    )
    away = sched.select(
        "game_id", pl.col("away_team").alias("team"), pl.col("_game_date").alias("game_date")
    )
    rest = (
        pl.concat([home, away])
        .sort(["team", "game_date", "game_id"])
        .with_columns(
            pl.col("game_date").diff().dt.total_days().over("team").alias("rest_days")
        )
        .with_columns(pl.col("rest_days").fill_null(7).clip(0, 21))
        .select("game_id", "team", "rest_days")
    )
    return out.join(rest, on=["game_id", "team"], how="left")


def build_rolling(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    id_col = "player_id" if "player_id" in df.columns else ("gsis_id" if "gsis_id" in df.columns else "player_name")
    df = df.sort([id_col, "season", "week"])

    exprs = []
    for col in ROLL_COLS:
        if col in df.columns:
            exprs.append(
                pl.col(col)
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .shift(1)
                .rolling_mean(window_size=WINDOW, min_samples=1)
                .over(id_col)
                .alias(f"roll_{col}")
            )
    exprs.append(
        pl.int_range(1, pl.len() + 1).over(id_col).alias("game_num")
    )
    out = df.with_columns(exprs)
    out = out.with_columns((pl.col("game_num") - 1).alias("games_played"))
    return out, id_col


def feature_cols(df: pl.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("roll_")]
    # Weather columns are kept in the feature table for display, but the
    # current schedules have too many null forecasts to improve the models.
    for extra in ["is_home", "rest_days", "opp_def_rating", "div_game"]:
        if extra in df.columns:
            cols.append(extra)
    return cols


def train_group(df: pl.DataFrame, group: str, positions: list[str]):
    data = df.filter(
        pl.col("position").is_in(positions)
        & (pl.col("games_played") >= 1)
        & pl.col("roll_fantasy_points").is_not_null()
    )
    if data.height < 50:
        print(f"{group}: only {data.height} rows; using rolling average fallback.")
        return None, {}, {}

    seasons = sorted(int(s) for s in data["season"].unique().to_list())
    if len(seasons) < 2:
        print(f"{group}: need at least two seasons; using fallback.")
        return None, {}, {}

    test_season = seasons[-1]
    train = data.filter(pl.col("season") < test_season)
    test = data.filter(pl.col("season") == test_season)
    if train.height < 50 or test.height < 10:
        print(f"{group}: split too small; using fallback.")
        return None, {}, {}

    cols = feature_cols(data)
    X_train = train.select(cols).to_numpy()
    y_train = train["fantasy_points"].to_numpy()
    X_test = test.select(cols).to_numpy()
    y_test = test["fantasy_points"].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    residuals = y_test - pred
    quantiles = {}
    q10_pred = q90_pred = None
    for label, quantile in (("q10", 0.1), ("q90", 0.9)):
        q_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        q_model.fit(X_train, y_train)
        quantiles[label] = q_model
        forecast = q_model.predict(X_test)
        if label == "q10":
            q10_pred = forecast
        else:
            q90_pred = forecast
    coverage = float(np.mean((y_test >= q10_pred) & (y_test <= q90_pred)))
    metrics = {
        "rows": int(len(y_test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "q10": round(float(np.quantile(residuals, 0.1)), 3),
        "q90": round(float(np.quantile(residuals, 0.9)), 3),
        "coverage_80": round(coverage, 3),
    }
    print(
        f"{group}: n={metrics['rows']} MAE={metrics['mae']} RMSE={metrics['rmse']} "
        f"80% coverage={metrics['coverage_80']}"
    )
    return model, metrics, quantiles


def predict_latest(df: pl.DataFrame, models: dict, id_col: str, quantiles: dict | None = None) -> list[dict]:
    upcoming_ctx: dict[str, dict] = {}
    try:
        schedules = pl.read_parquet("data/schedules.parquet")
        if {"season", "home_team", "away_team", "div_game"}.issubset(set(schedules.columns)):
            current = schedules["season"].max()
            team_dates: dict[str, list[datetime]] = {}
            for scheduled in schedules.filter(pl.col("season") == current).iter_rows(
                named=True
            ):
                day = scheduled.get("gameday")
                if not day:
                    continue
                try:
                    game_day = datetime.fromisoformat(str(day)[:10])
                except ValueError:
                    continue
                for team in (scheduled["home_team"], scheduled["away_team"]):
                    team_dates.setdefault(team, []).append(game_day)
            for team in team_dates:
                team_dates[team] = sorted(set(team_dates[team]))

            unplayed = schedules.filter(
                (pl.col("season") == current)
                & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
            ).sort(["week", "gameday"] if "gameday" in schedules.columns else ["week"])
            for game in unplayed.iter_rows(named=True):
                day = game.get("gameday")
                try:
                    game_day = datetime.fromisoformat(str(day)[:10]) if day else None
                except ValueError:
                    game_day = None
                for team, opponent, is_home in (
                    (game["home_team"], game["away_team"], 1),
                    (game["away_team"], game["home_team"], 0),
                ):
                    # Keep the next game only; later weeks should not overwrite it.
                    if team in upcoming_ctx:
                        continue
                    rest = 7
                    if game_day is not None:
                        prior = [
                            prior_day
                            for prior_day in team_dates.get(team, [])
                            if prior_day < game_day
                        ]
                        if prior:
                            rest = max(0, min(21, (game_day - prior[-1]).days))
                    upcoming_ctx[team] = {
                        "opponent": opponent,
                        "is_home": is_home,
                        "rest_days": rest,
                        "div_game": int(game.get("div_game") or 0),
                        "gameday": str(day) if day else None,
                    }
    except Exception:
        upcoming_ctx = {}

    # Latest opponent-position rating, used for the upcoming matchup rather
    # than the player's previous opponent.
    rating_map: dict[tuple[str, str], float] = {}
    if {"opponent_team", "pos_group", "opp_def_rating"}.issubset(set(df.columns)):
        ratings = (
            df.filter(pl.col("opp_def_rating").is_not_null())
            .sort(["season", "week"])
            .group_by(["opponent_team", "pos_group"])
            .agg(pl.col("opp_def_rating").last())
        )
        rating_map = {
            (row["opponent_team"], row["pos_group"]): float(row["opp_def_rating"])
            for row in ratings.iter_rows(named=True)
        }

    intervals = {}
    path = "models/player_intervals.json"
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                intervals = json.load(f)
        except Exception:
            intervals = {}
    # Latest row per player.
    latest = (
        df.sort(["season", "week"])
        .group_by(id_col)
        .agg(pl.all().last())
    )

    recent_cols = [
        "fantasy_points",
        "targets",
        "receptions",
        "yards",
        "touchdowns",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "completions",
        "attempts",
        "rushing_yards",
        "rushing_tds",
        "fg_made",
        "fg_att",
        "pat_made",
    ]
    recent = (
        df.sort(["season", "week"])
        .group_by(id_col)
        .agg(
            [
                pl.col(col).tail(6).alias(f"recent_{col}")
                for col in recent_cols
                if col in df.columns
            ]
        )
    )
    recent_map = {row[id_col]: row for row in recent.iter_rows(named=True)}

    records = []
    feature_vectors: dict[str, dict] = {}
    for row in latest.iter_rows(named=True):
        pos = row.get("position") or ""
        group = next((g for g, ps in POSITION_GROUPS.items() if pos in ps), None)
        if group is None:
            continue

        model = models.get(group)
        stats = row
        ctx = upcoming_ctx.get(stats.get("team"), {})
        values = None
        if model is not None:
            cols = feature_cols(df)
            values = []
            for c in cols:
                if c == "is_home":
                    values.append(float(ctx.get("is_home", 0.5)))
                elif c == "rest_days":
                    values.append(float(ctx.get("rest_days", 7.0)))
                elif c == "opp_def_rating":
                    values.append(
                        float(
                            rating_map.get(
                                (ctx.get("opponent"), stats.get("pos_group")), 0.0
                            )
                            or 0.0
                        )
                    )
                elif c == "div_game":
                    values.append(float(ctx.get("div_game", 0.0)))
                else:
                    values.append(float(stats.get(c) or 0.0))
            fantasy = float(model.predict([values])[0])
            feature_vectors[str(stats.get(id_col))] = {
                c: float(v) for c, v in zip(cols, values)
            }
        else:
            fantasy = float(stats.get("roll_fantasy_points") or 0.0)

        interval = intervals.get(group, {})
        residual_floor = max(0.0, fantasy + float(interval.get("q10", 0.0)))
        residual_ceiling = max(0.0, fantasy + float(interval.get("q90", 0.0)))
        floor = residual_floor
        ceiling = residual_ceiling
        if values is not None and quantiles and group in quantiles:
            try:
                q10 = float(quantiles[group]["q10"].predict([values])[0])
                q90 = float(quantiles[group]["q90"].predict([values])[0])
                # Envelope: conditional models can regress too far toward the
                # position mean for elite players, so keep at least the
                # unconditional residual bound.
                floor = max(0.0, max(q10, residual_floor))
                ceiling = max(0.0, max(q90, residual_ceiling))
                floor = min(floor, fantasy)
                ceiling = max(ceiling, fantasy)
            except Exception:
                pass

        recent_row = recent_map.get(stats.get(id_col)) or {}
        recent_stats = {}
        for col in recent_cols:
            values = recent_row.get(f"recent_{col}") or []
            recent_stats[col] = [
                round(float(value), 1) if value is not None else 0.0 for value in values
            ]

        records.append(
            {
                "player_id": stats.get(id_col),
                "name": stats.get("player_name") or stats.get("player_display_name") or stats.get(id_col),
                "team": stats.get("team") or "",
                "position": pos,
                "projected_targets": round(float(stats.get("roll_targets") or 0.0), 1),
                "projected_receptions": round(float(stats.get("roll_receptions") or 0.0), 1),
                "projected_yards": round(float(stats.get("roll_yards") or 0.0), 1),
                "projected_tds": round(float(stats.get("roll_touchdowns") or 0.0), 2),
                "projected_fantasy_points": round(fantasy, 1),
                "floor_fantasy_points": round(floor, 1),
                "ceiling_fantasy_points": round(ceiling, 1),
                "projected_pass_yards": round(float(stats.get("roll_passing_yards") or 0.0), 1)
                if pos == "QB"
                else None,
                "projected_pass_tds": round(float(stats.get("roll_passing_tds") or 0.0), 2)
                if pos == "QB"
                else None,
                "projected_interceptions": round(
                    float(stats.get("roll_passing_interceptions") or 0.0), 2
                )
                if pos == "QB"
                else None,
                "projected_rush_yards": round(float(stats.get("roll_rushing_yards") or 0.0), 1)
                if pos == "QB"
                else None,
                "projected_rush_tds": round(float(stats.get("roll_rushing_tds") or 0.0), 2)
                if pos == "QB"
                else None,
                "projected_fg": round(float(stats.get("roll_fg_made") or 0.0), 1)
                if pos == "K"
                else None,
                "projected_fg_att": round(float(stats.get("roll_fg_att") or 0.0), 1)
                if pos == "K"
                else None,
                "projected_xp": round(float(stats.get("roll_pat_made") or 0.0), 1)
                if pos == "K"
                else None,
                "recent_fantasy_points": recent_stats.get("fantasy_points", []),
                "recent_stats": recent_stats,
            }
        )

    records.sort(key=lambda r: r["projected_fantasy_points"], reverse=True)
    os.makedirs("models", exist_ok=True)
    with open("models/player_feature_vectors.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "features": feature_cols(df),
                "players": feature_vectors,
            },
            f,
            indent=2,
        )
    return records


def apply_injury_adjustments(records: list[dict]) -> list[dict]:
    path = "data/injuries_latest.parquet"
    if not os.path.exists(path):
        return records
    try:
        inj = pl.read_parquet(path)
        required = {"season", "week"}
        if not required.issubset(set(inj.columns)):
            return records
        latest_season = inj["season"].max()
        latest_week = inj.filter(pl.col("season") == latest_season)["week"].max()
        inj = inj.filter(
            (pl.col("season") == latest_season) & (pl.col("week") == latest_week)
        )
        status_col = next((c for c in inj.columns if "status" in c.lower()), None)
        id_col = next(
            (c for c in ["gsis_id", "player_id", "full_name", "player_name"] if c in inj.columns),
            None,
        )
        if status_col is None or id_col is None:
            return records

        status_map = {}
        for row in inj.iter_rows(named=True):
            pid = row.get(id_col)
            if pid is None:
                continue
            status_map[pid] = str(row.get(status_col) or "").lower()

        multipliers = {
            "out": 0.0,
            "doubtful": 0.25,
            "questionable": 0.75,
            "probable": 1.0,
        }
        stat_keys = [
            "projected_targets",
            "projected_receptions",
            "projected_yards",
            "projected_tds",
            "projected_fantasy_points",
            "projected_pass_yards",
            "projected_pass_tds",
        ]
        for record in records:
            status = status_map.get(record.get("player_id"))
            if not status:
                continue
            mult = multipliers.get(status, 1.0)
            record["injury_status"] = status
            for key in stat_keys:
                if record.get(key) is not None:
                    record[key] = round(record[key] * mult, 2)
        records.sort(key=lambda r: r.get("projected_fantasy_points", 0), reverse=True)
    except Exception as exc:  # pragma: no cover
        print(f"Injury adjustment skipped: {exc}")
    return records


def apply_depth_chart_adjustments(records: list[dict]) -> list[dict]:
    path = "data/depth_charts_latest.parquet"
    if not os.path.exists(path):
        return records
    try:
        dc = pl.read_parquet(path)
        if "gsis_id" not in dc.columns or "pos_rank" not in dc.columns or "dt" not in dc.columns:
            return records

        latest = dc.filter(pl.col("dt") == dc["dt"].max())
        rank_map = {}
        for row in latest.iter_rows(named=True):
            pid = row.get("gsis_id")
            rank = row.get("pos_rank")
            if pid is None or rank is None:
                continue
            rank_map[pid] = int(rank)

        multipliers = {1: 1.0, 2: 0.7, 3: 0.4}
        stat_keys = [
            "projected_targets",
            "projected_receptions",
            "projected_yards",
            "projected_tds",
            "projected_fantasy_points",
            "projected_pass_yards",
            "projected_pass_tds",
        ]
        for record in records:
            rank = rank_map.get(record.get("player_id"))
            if rank is None:
                continue
            mult = multipliers.get(rank, 0.2)
            record["depth_rank"] = rank
            for key in stat_keys:
                if record.get(key) is not None:
                    record[key] = round(record[key] * mult, 2)
        records.sort(key=lambda r: r.get("projected_fantasy_points", 0), reverse=True)
    except Exception as exc:  # pragma: no cover
        print(f"Depth-chart adjustment skipped: {exc}")
    return records


def apply_roster_status(records: list[dict]) -> list[dict]:
    path = None
    for candidate in ["data/rosters_weekly_history.parquet", "data/rosters_weekly_latest.parquet"]:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        return records
    try:
        rosters = pl.read_parquet(path)
        required = {"season", "week", "gsis_id", "status"}
        if not required.issubset(set(rosters.columns)):
            return records
        latest = (
            rosters.sort(["season", "week"])
            .group_by("gsis_id", maintain_order=True)
            .agg(pl.all().last())
        )
        status = {row["gsis_id"]: row["status"] for row in latest.iter_rows(named=True)}
        available = {"ACT"}
        kept = []
        for record in records:
            state = status.get(record.get("player_id"))
            record["roster_status"] = state
            if state is not None and state not in available:
                continue
            kept.append(record)
        return kept
    except Exception:
        return records


def main() -> None:
    df = load_player_features()
    if df.height == 0:
        print("No player features found; skipping player models.")
        return

    df = enrich_context(df)
    df = add_position_group(df)
    df = enrich_defense_rating(df)
    rolling, id_col = build_rolling(df)
    models = {}
    metrics = {}
    quantiles = {}

    for group, positions in POSITION_GROUPS.items():
        model, group_metrics, group_quantiles = train_group(rolling, group, positions)
        models[group] = model
        metrics[group] = group_metrics
        quantiles[group] = group_quantiles

    os.makedirs("models", exist_ok=True)
    joblib.dump(models, "models/player_models.joblib")
    joblib.dump(quantiles, "models/player_quantiles.joblib")
    with open("models/player_features.json", "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols(rolling)}, f, indent=2)

    os.makedirs("reports", exist_ok=True)
    with open("reports/player_backtest.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    intervals = {
        group: {"q10": m.get("q10", 0.0), "q90": m.get("q90", 0.0)}
        for group, m in metrics.items()
        if m
    }
    with open("models/player_intervals.json", "w", encoding="utf-8") as f:
        json.dump(intervals, f, indent=2)

    records = predict_latest(rolling, models, id_col, quantiles)
    records = apply_depth_chart_adjustments(records)
    records = apply_injury_adjustments(records)
    records = apply_roster_status(records)
    records = [r for r in records if r["projected_fantasy_points"] > 0][:250]

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/players_week.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote models/player_models.joblib and {len(records)} player projections.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Player model error: {exc}")
    sys.exit(0)