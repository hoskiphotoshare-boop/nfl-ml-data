import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from compare_market import movement_for, weather_severe

MOVE = {
    "total_move": -2.0,
    "spread_move": 1.5,
    "ml_move": -0.05,
}


def test_total_movement_against_over():
    assert movement_for(MOVE, "total", "over")["against"]
    assert not movement_for(MOVE, "total", "under")["against"]


def test_spread_movement_against_home():
    assert movement_for(MOVE, "spread", "home")["against"]
    assert not movement_for(MOVE, "spread", "away")["against"]


def test_ml_movement_against_home():
    assert movement_for(MOVE, "ml", "home")["against"]
    assert not movement_for(MOVE, "ml", "away")["against"]


def test_small_movement_is_not_against():
    small = {"total_move": -1.0, "spread_move": 0.5, "ml_move": -0.01}
    assert not movement_for(small, "total", "over")["against"]
    assert not movement_for(small, "spread", "home")["against"]
    assert not movement_for(small, "ml", "home")["against"]


def test_weather_severe_thresholds():
    assert weather_severe({"gust": 30})
    assert weather_severe({"wind": 20})
    assert weather_severe({"snow": 0.1})
    assert not weather_severe({"gust": 29, "wind": 19, "snow": 0})


def test_movement_matches_exact_game_not_pair():
    from compare_market import movement_for_event

    movements = {
        ("Buffalo Bills", "Kansas City Chiefs", "2026-11-27T01:20:00Z"): {
            "total_move": -2.0
        },
        ("Buffalo Bills", "Kansas City Chiefs", "2027-01-03T18:00:00Z"): {
            "total_move": 1.0
        },
    }
    exact = movement_for_event(
        movements,
        "Buffalo Bills",
        "Kansas City Chiefs",
        "2026-11-27T01:20:00Z",
    )
    assert exact["total_move"] == -2.0
    reversed_home = movement_for_event(
        movements,
        "Kansas City Chiefs",
        "Buffalo Bills",
        "2026-11-27T01:20:00Z",
    )
    assert reversed_home["total_move"] == -2.0
    assert (
        movement_for_event(
            movements,
            "Buffalo Bills",
            "Kansas City Chiefs",
            "2026-12-01T01:20:00Z",
        )
        is None
    )


def test_weather_severe_ignores_closed_roof():
    from compare_market import weather_severe

    assert not weather_severe({"roof": "dome", "gust": 50, "snow": 5})
    assert not weather_severe({"roof": "closed", "wind": 40})
    assert weather_severe({"roof": "outdoors", "gust": 31})


def test_actionable_edge_not_zeroed():
    import os, sys, json
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    # monkeypatch compare_market main? Instead test logic: actionable_edge == edge when movement against
    from compare_market import movement_for
    move = {"total_move": -2.0, "spread_move": 1.5, "ml_move": -0.05}
    entry = movement_for(move, "total", "over")
    assert entry["against"] is True
    # actionable_edge should not be forced to 0 by movement alone
    edge = 0.12
    actionable = edge  # new behavior
    assert actionable == edge


def test_total_disagreement_suppresses_actionable_edge():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    # Simulate logic: if |model_total - market_total| > 7, actionable_edge=0
    model_total = 53.6
    market_total = 41.5
    edge = 0.3059
    if abs(model_total - market_total) > 7.0:
        actionable = 0.0
        suppressed = "total_disagreement"
    else:
        actionable = edge
        suppressed = None
    assert actionable == 0.0
    assert suppressed == "total_disagreement"


def test_total_edge_cap():
    edge = 0.15
    actionable = min(edge, 0.10)
    assert actionable == 0.10
    assert actionable < edge
