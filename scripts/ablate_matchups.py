#!/usr/bin/env python3
"""Research ablation: matchup cross-features for the match model.

The shipped model compares each team's own rolling offense and defense in
parallel. It never asks how a specific offense matches up with the specific
defense it is about to face. This script builds those cross terms from
play-by-play data and measures whether they beat the current feature set in a
walk-forward backtest.

Writes ``reports/matchup_ablation.json``. Nothing here changes the shipped
model; promotion happens only if a variant wins.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MATCH = "features/match_features.parquet"
PBP = "data/pbp_latest.parquet"
OUT = "reports/matchup_ablation.json"

PROD = [
    # rolling EPA / success (pre-game, offense and defense)
    "home_off_epa", "home_def_epa", "home_success_rate",
    "home_pass_epa", "home_rush_epa",
    "away_off_epa", "away_def_epa", "away_success_rate",
    "away_pass_epa", "away_rush_epa",
    # rolling team stats
    "home_passing_epa", "away_passing_epa",
    "home_rushing_epa", "away_rushing_epa",
    # rest / context
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_short_week", "away_short_week", "div_game",
    # results
    "home_prior_games", "away_prior_games",
    "home_win_pct", "away_win_pct",
    "home_diff_pg", "away_diff_pg",
    "win_pct_diff", "diff_pg_diff",
    # quarterback continuity
    "home_qb_prior_epa", "away_qb_prior_epa",
    "home_qb_starts", "away_qb_starts",
    "home_qb_changed", "away_qb_changed",
    "home_qb_missing", "away_qb_missing",
    # injuries
    "home_inj_impact", "away_inj_impact", "inj_impact_diff",
    "home_inj_missing", "away_inj_missing",
]

CROSS = [
    "home_off_matchup",
    "away_off_matchup",
    "home_pass_matchup",
    "away_pass_matchup",
    "home_rush_matchup",
    "away_rush_matchup",
]

CROSS_DIFF = [
    "off_matchup_diff",
    "pass_matchup_diff",
    "rush_matchup_diff",
    "succ_matchup_diff",
]

DEF_RAW = [
    "home_def_pass_epa", "away_def_pass_epa",
    "home_def_rush_epa", "away_def_rush_epa",
    "home_def_success_allowed", "away_def_success_allowed",
]


def rolling_pre_game(team_game: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Pre-game averages: cumulative season history and a last-8 window."""
    team_game = team_game.sort(["season", "week", "game_id", "team"])
    team_game = team_game.with_columns(
        [pl.col(c).cum_sum().over("team").alias(f"cum_{c}") for c in cols]
        + [pl.col(cols[0]).cum_count().over("team").alias("game_num")]
    )
    for col in cols:
        team_game = team_game.with_columns(
            ((pl.col(f"cum_{col}") - pl.col(col)) / (pl.col("game_num") - 1).clip(1))
            .alias(f"pre_{col}")
        )
    # Last-8-game form: shifted so the current game is excluded.
    for col in cols:
        team_game = team_game.with_columns(
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=8, min_samples=1)
            .over("team")
            .alias(f"r8_{col}")
        )
    return team_game


def build_defense_features(match: pl.DataFrame) -> pl.DataFrame:
    if not os.path.exists(PBP):
        raise FileNotFoundError(PBP)
    pbp = pl.read_parquet(PBP)
    pbp = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
    )

    is_pass = pl.col("play_type") == "pass"
    is_rush = pl.col("play_type") == "run"

    offense = (
        pbp.group_by(["game_id", "posteam"])
        .agg(
            pl.col("epa").mean().alias("off_epa"),
            (pl.col("epa") > 0).mean().alias("off_success"),
            pl.col("epa").filter(is_pass).mean().alias("off_pass_epa"),
            pl.col("epa").filter(is_rush).mean().alias("off_rush_epa"),
            pl.len().alias("plays"),
        )
        .rename({"posteam": "team"})
    )
    defense = (
        pbp.group_by(["game_id", "defteam"])
        .agg(
            pl.col("epa").mean().alias("def_epa"),
            (pl.col("epa") > 0).mean().alias("def_success_allowed"),
            pl.col("epa").filter(is_pass).mean().alias("def_pass_epa"),
            pl.col("epa").filter(is_rush).mean().alias("def_rush_epa"),
        )
        .rename({"defteam": "team"})
    )

    team_game = offense.join(defense, on=["game_id", "team"], how="left")
    team_game = team_game.join(
        match.select("game_id", "season", "week").unique(), on="game_id", how="inner"
    )
    metric_cols = [
        "off_epa", "def_epa", "off_success", "def_success_allowed",
        "off_pass_epa", "def_pass_epa", "off_rush_epa", "def_rush_epa",
    ]
    team_game = rolling_pre_game(team_game, metric_cols)

    home = team_game.select(
        "game_id",
        pl.col("team").alias("home_team"),
        *[pl.col(f"pre_{c}").alias(f"home_{c}") for c in metric_cols],
        *[pl.col(f"r8_{c}").alias(f"home_r8_{c}") for c in metric_cols],
    )
    away = team_game.select(
        "game_id",
        pl.col("team").alias("away_team"),
        *[pl.col(f"pre_{c}").alias(f"away_{c}") for c in metric_cols],
        *[pl.col(f"r8_{c}").alias(f"away_r8_{c}") for c in metric_cols],
    )
    return match.join(home, on=["game_id", "home_team"], how="left").join(
        away, on=["game_id", "away_team"], how="left"
    )


def add_cross(df: pl.DataFrame) -> pl.DataFrame:
    z = pl.lit(0.0)
    return df.with_columns(
        # Offense vs the defense it actually faces.
        (pl.col("home_off_epa") - pl.col("away_def_epa")).alias("home_off_matchup"),
        (pl.col("away_off_epa") - pl.col("home_def_epa")).alias("away_off_matchup"),
        (pl.col("home_pass_epa") - pl.col("away_def_pass_epa")).alias("home_pass_matchup"),
        (pl.col("away_pass_epa") - pl.col("home_def_pass_epa")).alias("away_pass_matchup"),
        (pl.col("home_rush_epa") - pl.col("away_def_rush_epa")).alias("home_rush_matchup"),
        (pl.col("away_rush_epa") - pl.col("home_def_rush_epa")).alias("away_rush_matchup"),
        (pl.col("home_success_rate") - pl.col("away_def_success_allowed")).alias(
            "home_succ_matchup"
        ),
        (pl.col("away_success_rate") - pl.col("home_def_success_allowed")).alias(
            "away_succ_matchup"
        ),
    ).with_columns(
        (pl.col("home_off_matchup") - pl.col("away_off_matchup")).alias("off_matchup_diff"),
        (pl.col("home_pass_matchup") - pl.col("away_pass_matchup")).alias("pass_matchup_diff"),
        (pl.col("home_rush_matchup") - pl.col("away_rush_matchup")).alias("rush_matchup_diff"),
        (pl.col("home_succ_matchup") - pl.col("away_succ_matchup")).alias("succ_matchup_diff"),
    )


def add_prior_season(df: pl.DataFrame) -> pl.DataFrame:
    """Prior-season team quality: the signal current-season rolling stats lack in Weeks 1-4."""
    played = df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    home = played.select(
        pl.col("season"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Float64).alias("win"),
    )
    away = played.select(
        pl.col("season"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Float64).alias("win"),
    )
    team_season = (
        pl.concat([home, away])
        .group_by(["season", "team"])
        .agg(
            pl.col("pf").sum().alias("pf"),
            pl.col("pa").sum().alias("pa"),
            pl.col("win").mean().alias("win_pct"),
            pl.len().alias("games"),
        )
        .with_columns(
            ((pl.col("pf") - pl.col("pa")) / pl.col("games").clip(1)).alias("diff_pg")
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    prior = team_season.select(
        "season",
        "team",
        pl.col("diff_pg").alias("prev_diff_pg"),
        pl.col("win_pct").alias("prev_win_pct"),
    )
    home_prior = prior.select(
        "season",
        pl.col("team").alias("home_team"),
        pl.col("prev_diff_pg").alias("home_prev_diff_pg"),
        pl.col("prev_win_pct").alias("home_prev_win_pct"),
    )
    away_prior = prior.select(
        "season",
        pl.col("team").alias("away_team"),
        pl.col("prev_diff_pg").alias("away_prev_diff_pg"),
        pl.col("prev_win_pct").alias("away_prev_win_pct"),
    )
    out = df.join(home_prior, on=["season", "home_team"], how="left").join(
        away_prior, on=["season", "away_team"], how="left"
    )
    return out.with_columns(
        (pl.col("home_prev_diff_pg") - pl.col("away_prev_diff_pg")).alias(
            "prev_diff_pg_diff"
        )
    )


def add_blend(df: pl.DataFrame, k: float, tag: str) -> pl.DataFrame:
    """Shrink current-season results toward prior-season quality (empirical Bayes)."""
    hg = pl.col("home_prior_games").cast(pl.Float64)
    ag = pl.col("away_prior_games").cast(pl.Float64)
    return df.with_columns(
        ((hg * pl.col("home_diff_pg") + k * pl.col("home_prev_diff_pg")) / (hg + k)).alias(
            f"home_blend_diff_{tag}"
        ),
        ((ag * pl.col("away_diff_pg") + k * pl.col("away_prev_diff_pg")) / (ag + k)).alias(
            f"away_blend_diff_{tag}"
        ),
        ((hg * pl.col("home_win_pct") + k * pl.col("home_prev_win_pct")) / (hg + k)).alias(
            f"home_blend_win_{tag}"
        ),
        ((ag * pl.col("away_win_pct") + k * pl.col("away_prev_win_pct")) / (ag + k)).alias(
            f"away_blend_win_{tag}"
        ),
    ).with_columns(
        (pl.col(f"home_blend_diff_{tag}") - pl.col(f"away_blend_diff_{tag}")).alias(
            f"blend_diff_diff_{tag}"
        ),
        (pl.col(f"home_blend_win_{tag}") - pl.col(f"away_blend_win_{tag}")).alias(
            f"blend_win_diff_{tag}"
        ),
    )


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else None,
    }


def backtest(df: pl.DataFrame, features: list[str], test_from: int = 2022, scale: bool = False) -> dict:
    seeds = (0, 7, 21)
    pooled = {seed: ([], []) for seed in seeds}
    per_season = []
    seasons = sorted(int(s) for s in df["season"].unique().to_list() if s >= test_from)

    for season in seasons:
        train = df.filter(pl.col("season") < season)
        test = df.filter(pl.col("season") == season)
        if train.height < 100 or test.height < 20:
            continue
        X_train = train.select(features).to_numpy()
        y_train = train["home_win"].to_numpy()
        X_test = test.select(features).to_numpy()
        y_test = test["home_win"].to_numpy()
        season_probas = []
        for seed in seeds:
            lr = LogisticRegression(max_iter=4000, C=1.0, random_state=seed)
            model = make_pipeline(StandardScaler(), lr) if scale else lr
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pooled[seed][0].extend(y_test.tolist())
            pooled[seed][1].extend(proba.tolist())
            season_probas.append(proba)
        mean_proba = np.mean(season_probas, axis=0)
        per_season.append(
            {"season": season, "games": int(len(y_test)), **evaluate(y_test, mean_proba)}
        )

    summary = {}
    for seed, (y, p) in pooled.items():
        summary[str(seed)] = evaluate(np.array(y), np.array(p)) if y else {}
    if summary:
        keys = ("accuracy", "log_loss", "brier", "auc")
        pooled_mean = {
            k: float(np.mean([summary[str(s)][k] for s in seeds if summary[str(s)]]))
            for k in keys
        }
    else:
        pooled_mean = {}
    return {"features": features, "per_season": per_season, "pooled": pooled_mean}


def main() -> None:
    match = pl.read_parquet(MATCH)
    match = build_defense_features(match)
    match = add_cross(match)
    match = add_prior_season(match)
    match = match.filter(pl.col("home_win").is_not_null())

    experiments = {
        "prod": PROD,
        "prod+def_raw": PROD + [c for c in DEF_RAW if c in match.columns],
        "prod+cross": PROD + CROSS,
        "prod+cross+diff": PROD + CROSS + CROSS_DIFF,
        "prod+def_raw+cross+diff": PROD + [c for c in DEF_RAW if c in match.columns] + CROSS + CROSS_DIFF,
    }

    # Swap the cumulative EPA block for last-8 form, and add it alongside.
    last8_replace = {
        "home_off_epa": "home_r8_off_epa",
        "home_def_epa": "home_r8_def_epa",
        "home_success_rate": "home_r8_off_success",
        "home_pass_epa": "home_r8_off_pass_epa",
        "home_rush_epa": "home_r8_off_rush_epa",
        "away_off_epa": "away_r8_off_epa",
        "away_def_epa": "away_r8_def_epa",
        "away_success_rate": "away_r8_off_success",
        "away_pass_epa": "away_r8_off_pass_epa",
        "away_rush_epa": "away_r8_off_rush_epa",
    }
    last8_like = [last8_replace[c] for c in PROD if c in last8_replace]
    experiments["prod_last8"] = [
        last8_replace.get(c, c) for c in PROD
    ]
    experiments["prod+last8"] = PROD + last8_like

    ngs_all = [c for c in match.columns if c.startswith(("home_ng_", "away_ng_"))]
    ngs_core_names = [
        "ng_pass_avg_time_to_throw",
        "ng_pass_completion_percentage_above_expectation",
        "ng_rec_avg_separation",
        "ng_rec_avg_yac_above_expectation",
        "ng_rec_percent_share_of_intended_air_yards",
        "ng_rush_rush_yards_over_expected_per_att",
        "ng_rush_rush_pct_over_expected",
    ]
    ngs_core = [
        f"{side}_{name}" for name in ngs_core_names for side in ("home", "away")
    ]
    ngs_core = [c for c in ngs_core if c in match.columns]

    injury_cols = [
        "home_inj_impact", "away_inj_impact", "inj_impact_diff",
        "home_inj_missing", "away_inj_missing",
    ]
    experiments["prod_no_injury"] = [c for c in PROD if c not in injury_cols]
    experiments["prod+inj_groups"] = PROD + [
        c for c in ("inj_qb_diff", "inj_ol_diff", "inj_skill_diff", "inj_def_diff")
        if c in match.columns
    ]
    prior_cols = [
        "home_prev_diff_pg", "away_prev_diff_pg", "prev_diff_pg_diff",
        "home_prev_win_pct", "away_prev_win_pct",
    ]
    experiments["prod+prev_season"] = PROD + prior_cols
    experiments["prod_no_injury+prev_season"] = [
        c for c in PROD if c not in injury_cols
    ] + prior_cols

    for k in (2, 4):
        match = add_blend(match, k, f"k{k}")
        experiments[f"prod+blend{k}"] = PROD + [
            f"home_blend_diff_k{k}", f"away_blend_diff_k{k}", f"blend_diff_diff_k{k}",
            f"home_blend_win_k{k}", f"away_blend_win_k{k}", f"blend_win_diff_k{k}",
        ]

    results = {}
    for name, features in experiments.items():
        present = [c for c in features if c in match.columns]
        frame = match.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in present]
        )
        report = backtest(frame, present, test_from=2022)
        report["pooled_2023"] = backtest(frame, present, test_from=2023)["pooled"]
        results[name] = report
        p = report["pooled"]
        p23 = report["pooled_2023"]
        print(
            f"{name:26s} 2022+ ll={p['log_loss']:.4f} auc={p['auc']:.4f} | "
            f"2023+ ll={p23['log_loss']:.4f} auc={p23['auc']:.4f}"
        )

    # Standardised variants: mixed scales otherwise distort the linear fit.
    for name, features in {
        "prod_z": PROD,
        "prod+ngs_core": PROD + ngs_core,
        "prod+ngs_all": PROD + ngs_all,
        "prod+ngs_core_z": PROD + ngs_core,
        "prod+ngs_all_z": PROD + ngs_all,
    }.items():
        present = [c for c in features if c in match.columns]
        frame = match.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in present]
        )
        scale = name.endswith("_z")
        report = backtest(frame, present, test_from=2022, scale=scale)
        report["pooled_2023"] = backtest(frame, present, test_from=2023, scale=scale)["pooled"]
        report["scaled"] = scale
        results[name] = report
        p = report["pooled"]
        p23 = report["pooled_2023"]
        print(
            f"{name:26s} 2022+ ll={p['log_loss']:.4f} auc={p['auc']:.4f} | "
            f"2023+ ll={p23['log_loss']:.4f} auc={p23['auc']:.4f}"
        )

    best = min(
        results,
        key=lambda k: results[k]["pooled_2023"].get("log_loss", 9.9),
    )
    print(f"best: {best}")
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"best": best, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
