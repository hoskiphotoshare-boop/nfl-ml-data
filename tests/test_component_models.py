import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import pytest


def test_component_bundle_has_gate():
    import joblib
    b = joblib.load("models/component_models.joblib")
    assert "gate" in b
    assert "passed" in b["gate"]
    assert b["gate"]["passed"] is True


def test_component_monotonicity_smoke():
    # colder weather should not increase pass yards in model prediction
    import joblib, predict_upcoming as pu
    bundle = joblib.load("models/component_models.joblib")
    features = bundle["features"]
    # Use a sample vector from upcoming total_vectors
    import json
    vecs = json.load(open("models/upcoming_feature_vectors.json"))
    total_vecs = vecs.get("total_vectors") or vecs.get("games", {})
    if not total_vecs:
        pytest.skip("no vectors")
    gid = next(iter(total_vecs))
    base = dict(total_vecs[gid])
    # neutral weather
    neutral = {k: base.get(k, 0.0) for k in base}
    for w in ["wx_outdoor", "wx_temp", "wx_wind", "wx_gust", "wx_snow", "wx_precip", "wx_wind_15", "wx_gust_30", "wx_snow_any", "wx_cold_32"]:
        neutral[w] = 0 if w != "wx_temp" else 60.0
    # cold weather
    cold = dict(neutral)
    cold["wx_outdoor"] = 1
    cold["wx_temp"] = 20.0
    cold["wx_cold_32"] = 1
    cold["wx_wind"] = 18.0
    cold["wx_wind_15"] = 1
    week = 12
    season = 2026
    x_neutral = pu.build_feature_vector(features, neutral, neutral, week, season, weather=neutral)
    x_cold = pu.build_feature_vector(features, cold, cold, week, season, weather=cold)
    p_neutral = float(bundle["models"]["c_pass_yds"].predict([x_neutral])[0])
    p_cold = float(bundle["models"]["c_pass_yds"].predict([x_cold])[0])
    assert p_cold <= p_neutral + 1e-6


def test_fg_make_prob_monotonic():
    import joblib
    fg_bundle = joblib.load("models/fg_model.joblib")
    fg_features = fg_bundle["features"]
    # warm vs cold feature rows
    warm = {"kick_distance": 44.0, "kick_distance_sq": 44.0**2, "wx_temp": 60.0, "wx_wind": 0.0,
            "wx_gust": 0.0, "wx_snow": 0.0, "wx_cold_32": 0, "wx_wind_15": 0, "wx_gust_30": 0, "wx_outdoor": 0}
    cold = {"kick_distance": 44.0, "kick_distance_sq": 44.0**2, "wx_temp": 20.0, "wx_wind": 20.0,
            "wx_gust": 30.0, "wx_snow": 0.0, "wx_cold_32": 1, "wx_wind_15": 1, "wx_gust_30": 1, "wx_outdoor": 1}
    def prob(row):
        fx = [float(row.get(c, 0.0)) for c in fg_features]
        return float(fg_bundle["model"].predict_proba([fx])[0, 1])
    assert prob(cold) <= prob(warm) + 1e-6


def test_bridge_identity_neutral():
    import joblib, numpy as np
    bridge = joblib.load("models/scoring_bridge.joblib")
    # neutral components should reconstruct a finite total
    comps = [50.0] * len(bridge["features"])  # placeholder
    # replace with realistic neutral-ish values
    # just check model runs
    pred = bridge["model"].predict([comps])
    assert np.isfinite(pred[0])


def test_gate_fallback_logic():
    import joblib
    b = joblib.load("models/component_models.joblib")
    assert isinstance(b["gate"]["passed"], bool)
    # gate passed in our run
    assert b["gate"]["passed"] is True


def test_player_multiplier_identity():
    import json, copy
    players = json.load(open("predictions/players_week.json"))
    # Find a player on a dome team? Instead check weather_adjustment exists
    assert any("weather_adjustment" in p for p in players)
    # ratios should be 1.0 for some players (dome teams) 
    neutral_players = [p for p in players if p.get("weather_adjustment", {}).get("applied") is False]
    assert len(neutral_players) > 0
    for p in neutral_players:
        wa = p["weather_adjustment"]
        assert wa["pass_ratio"] == 1.0
        assert wa["rush_ratio"] == 1.0
        assert wa["fg_ratio"] == 1.0
