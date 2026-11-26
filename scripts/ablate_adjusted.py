#!/usr/bin/env python3
"""Research ablation: opponent-adjusted (strength-of-schedule) team ratings.

Raw rolling EPA mixes team quality with the quality of the opponents faced.
This script fits a weekly ridge model ``off_epa = off_i - def_j + home`` on
games played *before* the target week, then tests whether the adjusted ratings
improve the match model beyond the shipped feature set.

Writes ``reports/adjusted_ablation.json``. Nothing is promoted unless a variant
wins walk-forward.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

import ablate_matchups as base

MATCH = "features/match_features.parquet"
PBP = "data/pbp_latest.parquet"
OUT = "reports/adjusted_ablation.json"
ALPHA = 12.0


def team_game_table() -> pl.DataFrame:
    pbp = pl.read_parquet(PBP)
    pbp = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
    )
    off = (
        pbp.group_by(["game_id", "posteam"])
        .agg(pl.col("epa").mean().alias("off_epa"))
        .rename({"posteam": "team"})
    )
    deff = (
        pbp.group_by(["game_id", "defteam"])
        .agg(pl.col("epa").mean().alias("def_epa"))
        .rename({"defteam": "team"})
    )
    tg = off.join(deff, on=["game_id", "team"], how="left")

    sched = (
        pl.read_parquet(MATCH)
        .select("game_id", "season", "week", "home_team", "away_team")
        .unique(subset=["game_id"])
    )
    tg = tg.join(sched, on="game_id", how="inner")

    home = tg.filter(pl.col("team") == pl.col("home_team")).select(
        "game_id", "season", "week",
        pl.col("team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(1).alias("is_home"),
        "off_epa",
    )
    away = tg.filter(pl.col("team") == pl.col("away_team")).select(
        "game_id", "season", "week",
        pl.col("team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(0).alias("is_home"),
        "off_epa",
    )
    return pl.concat([home, away]).sort(["season", "week", "game_id", "team"])


def fit_ratings(train: pl.DataFrame, teams: list[str], alpha: float) -> tuple[dict, dict]:
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    rows = train.filter(
        pl.col("team").is_in(teams) & pl.col("opponent").is_in(teams)
    )
    if rows.height < 40:
        return {}, {}
    X = np.zeros((rows.height, 2 * n + 1))
    y = rows["off_epa"].to_numpy()
    is_home = rows["is_home"].to_numpy()
    for i, (team, opp) in enumerate(zip(rows["team"], rows["opponent"])):
        X[i, index[team]] += 1.0
        X[i, n + index[opp]] -= 1.0
        X[i, -1] = float(is_home[i])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    coef = model.coef_
    off = {team: float(coef[index[team]]) for team in teams}
    # Column j enters as -def_j, so negate to get "EPA allowed" (lower is better).
    deff = {team: float(-coef[n + index[team]]) for team in teams}
    off_mean = float(np.mean(list(off.values())))
    def_mean = float(np.mean(list(deff.values())))
    return (
        {k: v - off_mean for k, v in off.items()},
        {k: v - def_mean for k, v in deff.items()},
    )


def add_adjusted(match: pl.DataFrame, prior_weight: float = 1.0) -> pl.DataFrame:
    tg = team_game_table()
    teams = sorted(set(match["home_team"].unique().to_list()) | set(match["away_team"].unique().to_list()))
    seasons = sorted(int(s) for s in match["season"].unique().to_list())
    weeks = sorted(int(w) for w in match["week"].unique().to_list())

    records = []
    for season in seasons:
        for week in weeks:
            current = tg.filter((pl.col("season") == season) & (pl.col("week") < week))
            prior = tg.filter(pl.col("season") == season - 1)
            parts = []
            if current.height:
                parts.append(current)
            if prior_weight > 0 and prior.height:
                parts.append(prior)
            if not parts:
                continue
            train = pl.concat(parts)
            off, deff = fit_ratings(train, teams, ALPHA)
            if not off:
                continue
            records.append(
                {
                    "season": season,
                    "week": week,
                    "off": off,
                    "def": deff,
                }
            )

    rows = []
    for rec in records:
        off, deff = rec["off"], rec["def"]
        for row in match.filter(
            (pl.col("season") == rec["season"]) & (pl.col("week") == rec["week"])
        ).iter_rows(named=True):
            home, away = row["home_team"], row["away_team"]
            if home not in off or away not in off:
                continue
            rows.append(
                {
                    "game_id": row["game_id"],
                    "home_adj_off": off[home],
                    "away_adj_off": off[away],
                    "home_adj_def": deff[home],
                    "away_adj_def": deff[away],
                }
            )
    if not rows:
        return match
    adj = pl.DataFrame(rows).unique(subset=["game_id"])
    out = match.join(adj, on="game_id", how="left").with_columns(
        (pl.col("home_adj_off") - pl.col("away_adj_off")).alias("adj_off_diff"),
        (pl.col("home_adj_def") - pl.col("away_adj_def")).alias("adj_def_diff"),
        (pl.col("home_adj_off") - pl.col("away_adj_def")).alias("home_adj_net"),
        (pl.col("away_adj_off") - pl.col("home_adj_def")).alias("away_adj_net"),
    )
    return out.with_columns(
        (pl.col("home_adj_net") - pl.col("away_adj_net")).alias("adj_net_diff")
    )


def main() -> None:
    match = pl.read_parquet(MATCH).filter(pl.col("home_win").is_not_null())
    adj_cols = [
        "home_adj_off", "away_adj_off", "home_adj_def", "away_adj_def",
        "adj_off_diff", "adj_def_diff", "adj_net_diff",
    ]

    results = {}
    variants = {
        "prod": base.PROD,
        "prod+adj": base.PROD + adj_cols,
        "prod+adj_net": base.PROD + ["home_adj_net", "away_adj_net", "adj_net_diff"],
    }
    for prior_weight in (0.0, 1.0):
        tag = "cur" if prior_weight == 0.0 else "curprior"
        frame = add_adjusted(match, prior_weight=prior_weight)
        if "home_adj_off" not in frame.columns:
            continue
        for name, feats in variants.items():
            present = [c for c in feats if c in frame.columns]
            clean = frame.with_columns(
                [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in present]
            )
            report = base.backtest(clean, present, test_from=2023)
            results[f"{name}_{tag}"] = report
            p = report["pooled"]
            print(
                f"{name+'_'+tag:22s} 2023+ ll={p['log_loss']:.4f} "
                f"acc={p['accuracy']:.4f} auc={p['auc']:.4f}"
            )

    baseline = results.get("prod_cur", {}).get("pooled", {}).get("log_loss")
    best = min(results, key=lambda k: results[k]["pooled"]["log_loss"])
    print(f"baseline={baseline:.4f} best={best} ll={results[best]['pooled']['log_loss']:.4f}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"best": best, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
