#!/usr/bin/env python3
"""Research ablation: NGS and PFR advanced stats for player models.

Both datasets are already downloaded but never merged into the player feature
table. This script joins them by player/game, mean-imputes missing values within
the position group (never zero-fills, which produced the earlier fake NGS gain),
and measures walk-forward MAE per position.

Writes ``reports/player_advanced_ablation.json``. Nothing is promoted unless a
variant wins the gate.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import train_player_models as T

OUT = "reports/player_advanced_ablation.json"
TEST_SEASONS = (2023, 2024, 2025, 2026)
WINDOW = T.WINDOW

NGS_REC = [
    "avg_separation", "avg_cushion", "avg_yac_above_expectation",
    "percent_share_of_intended_air_yards", "catch_percentage",
]
NGS_RUSH = ["rush_yards_over_expected_per_att", "rush_pct_over_expected", "efficiency"]
NGS_PASS = [
    "avg_time_to_throw", "completion_percentage_above_expectation",
    "aggressiveness", "passer_rating",
]
PFR_REC = ["receiving_drop_pct", "receiving_rat", "receiving_broken_tackles"]
PFR_PASS = ["passing_drop_pct", "passing_bad_throw_pct", "times_pressured_pct"]
PFR_RUSH = [
    "rushing_yards_before_contact_avg",
    "rushing_yards_after_contact_avg",
    "rushing_broken_tackles",
]


def pfr_id_map() -> dict[str, str]:
    players = pl.read_parquet("data/players_latest.parquet").select(["gsis_id", "pfr_id"])
    mapping = {
        str(row["pfr_id"]): str(row["gsis_id"])
        for row in players.iter_rows(named=True)
        if row["gsis_id"] and row["pfr_id"]
    }
    rosters = pl.read_parquet("data/rosters_weekly_latest.parquet")
    for row in rosters.select(["gsis_id", "pfr_id"]).iter_rows(named=True):
        if row["gsis_id"] and row["pfr_id"]:
            mapping.setdefault(str(row["pfr_id"]), str(row["gsis_id"]))
    return mapping


def merge_ngs(df: pl.DataFrame) -> pl.DataFrame:
    for file, cols in (
        ("data/nextgen_receiving_latest.parquet", NGS_REC),
        ("data/nextgen_rushing_latest.parquet", NGS_RUSH),
        ("data/nextgen_passing_latest.parquet", NGS_PASS),
    ):
        if not os.path.exists(file):
            continue
        ngs = pl.read_parquet(file)
        present = [c for c in cols if c in ngs.columns]
        if not present:
            continue
        ngs = (
            ngs.filter(pl.col("week") > 0)
            .select(
                ["player_gsis_id", "season", "week"]
                + [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in present]
            )
            .rename({"player_gsis_id": "player_id"})
            .group_by(["player_id", "season", "week"])
            .agg([pl.col(c).mean() for c in present])
        )
        df = df.join(ngs, on=["player_id", "season", "week"], how="left")
    return df


def merge_pfr(df: pl.DataFrame) -> pl.DataFrame:
    mapping = pfr_id_map()
    for file, cols in (
        ("data/pfr_rec_latest.parquet", PFR_REC),
        ("data/pfr_pass_latest.parquet", PFR_PASS),
        ("data/pfr_rush_latest.parquet", PFR_RUSH),
    ):
        if not os.path.exists(file):
            continue
        pfr = pl.read_parquet(file)
        present = [c for c in cols if c in pfr.columns]
        if not present:
            continue
        pfr = pfr.select(["game_id", "pfr_player_id"] + present)
        pfr = pfr.with_columns(
            pl.col("pfr_player_id").replace_strict(mapping, default=None).alias("player_id")
        ).filter(pl.col("player_id").is_not_null())
        pfr = pfr.group_by(["player_id", "game_id"]).agg(
            [pl.col(c).cast(pl.Float64, strict=False).mean() for c in present]
        )
        df = df.join(pfr, on=["player_id", "game_id"], how="left")
    return df


def add_rolling(df: pl.DataFrame, id_col: str, cols: list[str]) -> pl.DataFrame:
    # Mean-impute within the position group so a missing advanced stat is neutral,
    # not an extreme zero (the mistake that inflated the earlier NGS test).
    df = df.with_columns(
        [
            pl.col(c)
            .cast(pl.Float64, strict=False)
            .fill_null(pl.col(c).cast(pl.Float64, strict=False).mean().over("pos_group"))
            .alias(c)
            for c in cols
        ]
    )
    for c in cols:
        df = df.with_columns(
            pl.col(c)
            .shift(1)
            .rolling_mean(window_size=WINDOW, min_samples=1)
            .over(id_col)
            .alias(f"roll_{c}")
        )
    return df


def add_cumulative(df: pl.DataFrame, id_col: str, cols: list[str]) -> pl.DataFrame:
    """Season-to-date pre-game averages: stable versions of noisy efficiency stats."""
    for c in cols:
        prior = (
            pl.col(c).cum_sum().over([id_col, "season"]) - pl.col(c)
        ) / (pl.col(c).cum_count().over([id_col, "season"]) - 1).clip(1)
        df = df.with_columns(
            pl.when(pl.col(c).cum_count().over([id_col, "season"]) > 1)
            .then(prior)
            .otherwise(None)
            .alias(f"cum_{c}")
        )
    cum_cols = [f"cum_{c}" for c in cols]
    return df.with_columns(
        [
            pl.col(c).fill_null(pl.col(c).mean().over("pos_group")).alias(c)
            for c in cum_cols
        ]
    )


def evaluate(df: pl.DataFrame, features: list[str]) -> dict:
    out = {}
    weighted_abs = 0.0
    weighted_n = 0
    for group, positions in T.POSITION_GROUPS.items():
        abs_sum = 0.0
        rows = 0
        per_season = []
        for season in TEST_SEASONS:
            train = df.filter(
                (pl.col("season") < season)
                & pl.col("position").is_in(positions)
                & (pl.col("games_played") >= 1)
                & pl.col("roll_fantasy_points").is_not_null()
            )
            test = df.filter(
                (pl.col("season") == season)
                & pl.col("position").is_in(positions)
                & (pl.col("games_played") >= 1)
                & pl.col("roll_fantasy_points").is_not_null()
            )
            if train.height < 50 or test.height < 10:
                continue
            model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=3, random_state=42
            )
            train_matrix = train.select(features).to_numpy()
            test_matrix = test.select(features).to_numpy()
            # Drop columns that are constant within this training fold: HGB's
            # binning fails on single-valued features, and they add no signal.
            keep = np.nanstd(train_matrix, axis=0) > 1e-9
            model.fit(train_matrix[:, keep], train["fantasy_points"].to_numpy())
            pred = model.predict(test_matrix[:, keep])
            y = test["fantasy_points"].to_numpy()
            mae = mean_absolute_error(y, pred)
            per_season.append((season, round(float(mae), 3)))
            abs_sum += float(np.abs(y - pred).sum())
            rows += len(y)
        if rows:
            out[group] = {
                "mae": round(abs_sum / rows, 4),
                "per_season": per_season,
            }
            weighted_abs += abs_sum
            weighted_n += rows
    if weighted_n:
        out["_weighted"] = round(weighted_abs / weighted_n, 4)
    return out


def main() -> None:
    df = T.load_player_features()
    df = T.enrich_context(df)
    df = T.add_position_group(df)
    df = T.enrich_defense_rating(df)
    df = merge_ngs(df)
    df = merge_pfr(df)

    rolling, id_col = T.build_rolling(df)
    base = T.feature_cols(rolling)

    advanced = [c for c in NGS_REC + NGS_RUSH + NGS_PASS + PFR_REC + PFR_PASS + PFR_RUSH if c in rolling.columns]
    rolling = add_rolling(rolling, id_col, advanced)
    rolling = add_cumulative(rolling, id_col, advanced)

    def roll(cols):
        return [f"roll_{c}" for c in cols if f"roll_{c}" in rolling.columns]

    def cum(cols):
        return [f"cum_{c}" for c in cols if f"cum_{c}" in rolling.columns]

    variants = {
        "base": base,
        "base+ngs_rec": base + roll(NGS_REC),
        "base+ngs_rush": base + roll(NGS_RUSH),
        "base+ngs_pass": base + roll(NGS_PASS),
        "base+ngs_all": base + roll(NGS_REC + NGS_RUSH + NGS_PASS),
        "base+pfr": base + roll(PFR_REC + PFR_PASS + PFR_RUSH),
        "base+all": base + roll(NGS_REC + NGS_RUSH + NGS_PASS + PFR_REC + PFR_PASS + PFR_RUSH),
        "base+ngs_cum": base + cum(NGS_REC + NGS_RUSH + NGS_PASS),
        "base+pfr_cum": base + cum(PFR_REC + PFR_PASS + PFR_RUSH),
        "base+all_cum": base + cum(NGS_REC + NGS_RUSH + NGS_PASS + PFR_REC + PFR_PASS + PFR_RUSH),
    }

    results = {}
    for name, feats in variants.items():
        present = [c for c in feats if c in rolling.columns]
        results[name] = evaluate(rolling, present)
        print(name, {k: v["mae"] if isinstance(v, dict) else v for k, v in results[name].items()})

    base_score = results["base"]["_weighted"]
    best = min(results, key=lambda k: results[k]["_weighted"])
    print(f"baseline={base_score:.4f} best={best} weighted={results[best]['_weighted']:.4f}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
