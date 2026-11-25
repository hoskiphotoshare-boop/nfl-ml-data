#!/usr/bin/env python3
"""Generate SHAP explanations for the match and player models.

Writes ``reports/model_explanations.json`` for the admin diagnostics page:

- global feature importance (mean |SHAP| and signed mean) per model
- local drivers for upcoming games and the top projected players

The match model is linear, so ``shap.LinearExplainer`` is exact.
Player models are gradient-boosted trees; a permutation explainer is used so
the values remain model-agnostic and additive without relying on internal
tree structure.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import polars as pl

try:
    import shap
except Exception as exc:  # pragma: no cover - shap is optional
    print(f"SHAP not installed; skipping explanations: {exc}")
    raise SystemExit(0)

from predict_upcoming import (
    apply_platt,
    build_feature_vector,
    load_match_features,
    load_platt,
    team_latest_row,
    team_stats_from_row,
)
from train_player_models import (
    POSITION_GROUPS,
    add_position_group,
    build_rolling,
    enrich_context,
    enrich_defense_rating,
    load_player_features,
)

MATCH_MODEL = "models/match_model.joblib"
PLAYER_MODELS = "models/player_models.joblib"
PLAYER_FEATURES = "models/player_features.json"
PREDICTIONS = "predictions/upcoming_games.json"
OUTPUT = "reports/model_explanations.json"

MATCH_SAMPLE = 250
PLAYER_SAMPLE = 60
LOCAL_GAMES = 200
LOCAL_PLAYERS = 6
TOP_DRIVERS = 8
INFERENCE_VECTORS = "models/upcoming_feature_vectors.json"
PLAYER_VECTORS = "models/player_feature_vectors.json"


def load_inference_vectors(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("games", {})
    except Exception:
        return {}

MATCH_ACRONYMS = {
    "epa": "EPA",
    "qb": "QB",
    "ol": "OL",
    "rb": "RB",
    "wr": "WR",
    "te": "TE",
    "inj": "Injury",
    "pg": "PG",
    "ng": "NG",
    "td": "TD",
    "tds": "TDs",
    "los": "LOS",
    "yac": "YAC",
    "pct": "Pct",
    "att": "Att",
}

SPECIAL_LABELS = {
    "diff_pg_diff": "Point diff (home − away)",
    "win_pct_diff": "Win pct (home − away)",
    "rest_diff": "Rest days (home − away)",
    "inj_impact_diff": "Injury impact (home − away)",
}


def pretty(name: str) -> str:
    words = []
    for part in name.split("_"):
        words.append(MATCH_ACRONYMS.get(part.lower(), part.capitalize()))
    return " ".join(words)


def match_label(feature: str) -> str:
    if feature in SPECIAL_LABELS:
        return SPECIAL_LABELS[feature]
    if feature.startswith("home_"):
        return f"Home · {pretty(feature[5:])}"
    if feature.startswith("away_"):
        return f"Away · {pretty(feature[5:])}"
    return pretty(feature)


def player_label(feature: str) -> str:
    if feature == "is_home":
        return "Home game"
    if feature == "rest_days":
        return "Rest days"
    if feature == "opp_def_rating":
        return "Opponent defense rating"
    if feature.startswith("roll_"):
        return f"Rolling {pretty(feature[5:]).lower()}"
    return pretty(feature)


def short_features(features: list[str], values: np.ndarray, shap_values: np.ndarray, labeler, limit=TOP_DRIVERS):
    order = np.argsort(np.abs(shap_values))[::-1][:limit]
    drivers = []
    for index in order:
        drivers.append(
            {
                "feature": features[index],
                "label": labeler(features[index]),
                "value": round(float(values[index]), 4),
                "shap": round(float(shap_values[index]), 4),
            }
        )
    return drivers


def match_explanations() -> dict | None:
    if not os.path.exists(MATCH_MODEL):
        return None
    bundle = joblib.load(MATCH_MODEL)
    model = bundle["model"]
    features = bundle["features"]
    frame = load_match_features()
    if frame is None:
        return None

    matrix = np.nan_to_num(frame.select(features).to_numpy(), nan=0.0)
    background = matrix[-300:]
    explainer = shap.LinearExplainer(model, background)
    sample = matrix[-MATCH_SAMPLE:]
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 1:
        values = values.reshape(1, -1)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    total = mean_abs.sum() or 1.0
    order = [i for i in np.argsort(mean_abs)[::-1] if mean_abs[i] > 1e-9][:20]
    global_drivers = [
        {
            "feature": features[i],
            "label": match_label(features[i]),
            "mean_abs_shap": round(float(mean_abs[i]), 4),
            "mean_shap": round(float(mean_signed[i]), 4),
            "share": round(float(mean_abs[i] / total), 4),
        }
        for i in order
    ]

    base_raw = float(np.asarray(explainer.expected_value).ravel()[0])
    base_probability = 1.0 / (1.0 + np.exp(-base_raw))

    games = []
    upcoming = json.load(open(PREDICTIONS, encoding="utf-8")) if os.path.exists(PREDICTIONS) else []
    upcoming = sorted(upcoming, key=lambda g: (g.get("gameday") or "", g.get("gametime") or ""))[:LOCAL_GAMES]
    if upcoming:
        stored = load_inference_vectors(INFERENCE_VECTORS)
        stats = {}
        rows = []
        for game in upcoming:
            vector = stored.get(str(game.get("game_id")))
            if vector:
                rows.append([float(vector.get(feature, 0.0)) for feature in features])
                continue
            for team in (game["home_team"], game["away_team"]):
                if team not in stats:
                    stats[team] = team_stats_from_row(team_latest_row(frame, team), team)
            rows.append(
                build_feature_vector(
                    features,
                    stats[game["home_team"]],
                    stats[game["away_team"]],
                    int(game.get("week") or 0),
                    int(game.get("season") or 0),
                )
            )
        game_values = np.asarray(explainer.shap_values(np.asarray(rows)))
        if game_values.ndim == 1:
            game_values = game_values.reshape(1, -1)
        platt = load_platt()
        for game, row, shap_row in zip(upcoming, rows, game_values):
            logit = base_raw + float(shap_row.sum())
            raw_probability = float(1.0 / (1.0 + np.exp(-logit)))
            games.append(
                {
                    "game_id": game.get("game_id"),
                    "matchup": f"{game['away_team']} @ {game['home_team']}",
                    "week": game.get("week"),
                    "home_team": game["home_team"],
                    "prediction": round(apply_platt(raw_probability, platt), 4),
                    "raw_prediction": round(raw_probability, 4),
                    "base_probability": round(float(base_probability), 4),
                    "drivers": short_features(features, np.asarray(row), shap_row, match_label),
                }
            )

    return {
        "model": bundle.get("name", type(model).__name__),
        "sample_size": int(len(sample)),
        "base_value_logit": round(base_raw, 4),
        "base_probability": round(float(base_probability), 4),
        "global": global_drivers,
        "games": games,
    }


def score_label(feature: str) -> str:
    if feature == "div_game":
        return "Division game"
    return match_label(feature)


def score_explanations() -> dict | None:
    if not os.path.exists("models/score_models.joblib"):
        return None
    bundle = joblib.load("models/score_models.joblib")
    features = bundle["features"]
    frame = load_match_features()
    if frame is None:
        return None
    matrix = np.nan_to_num(frame.select(features).to_numpy(), nan=0.0)
    sample = matrix[-250:]
    output = {"features": features, "global": {}, "games": []}

    explainers = {}
    for key in ("margin", "total"):
        model = bundle.get(f"{key}_model")
        if model is None:
            continue
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(sample, check_additivity=False))
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        total = mean_abs.sum() or 1.0
        order = [i for i in np.argsort(mean_abs)[::-1] if mean_abs[i] > 1e-9][:20]
        output["global"][key] = [
            {
                "feature": features[i],
                "label": score_label(features[i]),
                "mean_abs_shap": round(float(mean_abs[i]), 4),
                "mean_shap": round(float(mean_signed[i]), 4),
                "share": round(float(mean_abs[i] / total), 4),
            }
            for i in order
        ]
        explainers[key] = (model, explainer)

    upcoming = (
        json.load(open(PREDICTIONS, encoding="utf-8"))
        if os.path.exists(PREDICTIONS)
        else []
    )
    upcoming = sorted(upcoming, key=lambda g: (g.get("gameday") or "", g.get("gametime") or ""))
    if not upcoming or not explainers:
        return output

    stats = {}
    rows = []
    stored = load_inference_vectors(INFERENCE_VECTORS)
    for game in upcoming:
        vector = stored.get(str(game.get("game_id")))
        if vector:
            rows.append([float(vector.get(feature, 0.0)) for feature in features])
            continue
        for team in (game["home_team"], game["away_team"]):
            if team not in stats:
                stats[team] = team_stats_from_row(team_latest_row(frame, team), team)
        stats[game["home_team"]]["div_game"] = int(game.get("div_game") or 0)
        stats[game["away_team"]]["div_game"] = int(game.get("div_game") or 0)
        rows.append(
            build_feature_vector(
                features,
                stats[game["home_team"]],
                stats[game["away_team"]],
                int(game.get("week") or 0),
                int(game.get("season") or 0),
            )
        )
    matrix_games = np.asarray(rows)
    for game, row in zip(upcoming, matrix_games):
        entry = {
            "game_id": game.get("game_id"),
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "week": game.get("week"),
        }
        for key, (model, explainer) in explainers.items():
            values = np.asarray(explainer.shap_values(np.asarray([row]), check_additivity=False))[0]
            entry[key] = {
                "prediction": round(float(model.predict([row])[0]), 2),
                "base": round(float(np.asarray(explainer.expected_value).ravel()[0]), 2),
                "drivers": short_features(
                    features, np.asarray(row), values, score_label
                ),
            }
        output["games"].append(entry)
    return output


def latest_player_rows(data: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return data.sort(["season", "week"]).unique(subset=[id_col], keep="last")


def player_explanations() -> dict:
    if not (os.path.exists(PLAYER_MODELS) and os.path.exists(PLAYER_FEATURES)):
        return {}
    models = joblib.load(PLAYER_MODELS)
    features = json.load(open(PLAYER_FEATURES, encoding="utf-8"))["features"]
    frame = load_player_features()
    frame = enrich_defense_rating(add_position_group(enrich_context(frame)))
    rolling, id_col = build_rolling(frame)
    name_col = next(
        (c for c in ["player_display_name", "player_name", "full_name"] if c in rolling.columns),
        id_col,
    )
    team_col = "team" if "team" in rolling.columns else None

    output = {}
    player_records = []
    if os.path.exists("predictions/players_week.json"):
        try:
            with open("predictions/players_week.json", encoding="utf-8") as handle:
                player_records = json.load(handle)
        except Exception:
            player_records = []
    player_vectors = {}
    if os.path.exists(PLAYER_VECTORS):
        try:
            with open(PLAYER_VECTORS, encoding="utf-8") as handle:
                player_vectors = json.load(handle).get("players", {})
        except Exception:
            player_vectors = {}

    for group, positions in POSITION_GROUPS.items():
        model = models.get(group)
        if model is None:
            continue
        data = rolling.filter(
            pl.col("position").is_in(positions)
            & (pl.col("games_played") >= 1)
            & pl.col("roll_fantasy_points").is_not_null()
        )
        if data.height < 20:
            continue
        matrix = np.nan_to_num(data.select(features).to_numpy(), nan=0.0)
        background = matrix[-120:]
        explainer = shap.PermutationExplainer(model.predict, background)

        sample = matrix[-PLAYER_SAMPLE:]
        sample_explanation = explainer(sample)
        values = np.asarray(sample_explanation.values)
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        total = mean_abs.sum() or 1.0
        order = [i for i in np.argsort(mean_abs)[::-1] if mean_abs[i] > 1e-9][:20]
        global_drivers = [
            {
                "feature": features[i],
                "label": player_label(features[i]),
                "mean_abs_shap": round(float(mean_abs[i]), 4),
                "mean_shap": round(float(mean_signed[i]), 4),
                "share": round(float(mean_abs[i] / total), 4),
            }
            for i in order
        ]

        chosen = []
        if player_vectors:
            candidates = [
                record
                for record in player_records
                if record.get("position") in positions
                and str(record.get("player_id")) in player_vectors
            ]
            candidates.sort(
                key=lambda record: record.get("projected_fantasy_points") or 0,
                reverse=True,
            )
            chosen = candidates[:LOCAL_PLAYERS]

        if chosen:
            top_matrix = np.asarray(
                [
                    [
                        float(player_vectors[str(record["player_id"])].get(feature, 0.0))
                        for feature in features
                    ]
                    for record in chosen
                ]
            )
            predictions = model.predict(top_matrix)
            top_rows = [
                (record.get("player_id"), record.get("name"), record.get("team"))
                for record in chosen
            ]
        else:
            latest = latest_player_rows(data, id_col)
            latest_matrix = np.nan_to_num(latest.select(features).to_numpy(), nan=0.0)
            predictions = model.predict(latest_matrix)
            top_idx = np.argsort(predictions)[::-1][:LOCAL_PLAYERS]
            top_matrix = latest_matrix[top_idx]
            latest_rows = latest.select(
                [id_col, name_col] + ([team_col] if team_col else [])
            ).to_numpy()
            top_rows = [
                (
                    latest_rows[index][0],
                    latest_rows[index][1] if len(latest_rows[index]) > 1 else str(latest_rows[index][0]),
                    latest_rows[index][2] if len(latest_rows[index]) > 2 and team_col else "",
                )
                for index in top_idx
            ]

        explanation = explainer(top_matrix)
        local_values = np.asarray(explanation.values)

        players = []
        for position, (player_id, name, team) in enumerate(top_rows):
            players.append(
                {
                    "player_id": str(player_id),
                    "name": str(name),
                    "team": str(team),
                    "prediction": round(float(predictions[position]), 2),
                    "drivers": short_features(
                        features, top_matrix[position], local_values[position], player_label
                    ),
                }
            )

        base_value = float(np.asarray(explanation.base_values).ravel()[0])
        output[group] = {
            "model": type(model).__name__,
            "sample_size": int(len(sample)),
            "base_value": round(base_value, 3),
            "global": global_drivers,
            "players": players,
        }
    return output


def main() -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Linear SHAP (match) and permutation SHAP (players)",
    }
    match = match_explanations()
    if match:
        payload["match"] = match
        print(f"Match SHAP: {len(match['games'])} games, {len(match['global'])} global drivers")
    score = score_explanations()
    if score:
        payload["score"] = score
        print(f"Score SHAP: {len(score['games'])} games")
    payload["players"] = player_explanations()
    for group, data in payload["players"].items():
        print(f"{group} SHAP: {len(data['players'])} players")

    os.makedirs("reports", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"SHAP explanation error: {exc}")
    sys.exit(0)
