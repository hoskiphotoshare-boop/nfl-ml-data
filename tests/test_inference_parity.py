"""Regression tests for training/inference feature parity.

The pipeline stores the exact feature vectors used at inference time so SHAP
explanations and audit tools can reproduce the live predictions. These tests
fail if the stored vectors, models and JSON artifacts drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest


def load(path: str):
    file = Path(path)
    if not file.exists():
        pytest.skip(f"{path} not available")
    with file.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_match_vectors_reproduce_predictions():
    bundle = joblib.load("models/match_model.joblib")
    vectors = load("models/upcoming_feature_vectors.json")["games"]
    games = {game["game_id"]: game for game in load("predictions/upcoming_games.json")}
    assert vectors, "no stored inference vectors"
    worst = 0.0
    for game_id, vector in vectors.items():
        game = games.get(game_id)
        if not game or game.get("raw_home_win_prob") is None:
            continue
        row = [float(vector.get(feature, 0.0)) for feature in bundle["features"]]
        probability = float(bundle["model"].predict_proba([row])[0, 1])
        worst = max(worst, abs(probability - game["raw_home_win_prob"]))
    assert worst < 0.002, f"match vector mismatch {worst:.4f}"


def test_score_vectors_reproduce_totals():
    bundle = joblib.load("models/score_models.joblib")
    vectors = load("models/upcoming_feature_vectors.json")["games"]
    games = {game["game_id"]: game for game in load("predictions/upcoming_games.json")}
    worst = 0.0
    for game_id, vector in vectors.items():
        game = games.get(game_id)
        if not game or game.get("total") is None:
            continue
        row = [float(vector.get(feature, 0.0)) for feature in bundle["features"]]
        total = float(bundle["total_model"].predict([row])[0])
        worst = max(worst, abs(total - game["total"]))
    assert worst < 0.15, f"total vector mismatch {worst:.3f}"


def test_shap_predictions_match_live_predictions():
    explanations = load("reports/model_explanations.json")
    games = {game["game_id"]: game for game in load("predictions/upcoming_games.json")}
    worst_match = 0.0
    for game in explanations.get("match", {}).get("games", []):
        live = games.get(game.get("game_id"))
        if live and live.get("home_win_prob") is not None:
            worst_match = max(worst_match, abs(game["prediction"] - live["home_win_prob"]))
    assert worst_match < 0.002, f"match SHAP mismatch {worst_match:.4f}"

    worst_total = 0.0
    for game in explanations.get("score", {}).get("games", []):
        live = games.get(game.get("game_id"))
        if live and live.get("total") is not None:
            worst_total = max(
                worst_total, abs(game["total"]["prediction"] - live["total"])
            )
    assert worst_total < 0.15, f"total SHAP mismatch {worst_total:.3f}"


def test_player_vectors_cover_projections():
    vectors = load("models/player_feature_vectors.json")
    players = load("predictions/players_week.json")
    feature_config = load("models/player_features.json")
    assert vectors.get("features") == feature_config.get("features")
    missing = [
        player.get("name")
        for player in players
        if str(player.get("player_id")) not in vectors.get("players", {})
    ]
    assert not missing, f"missing player feature vectors: {missing[:5]}"
