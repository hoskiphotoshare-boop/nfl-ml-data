import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import update_calls


def test_grade_market_ml():
    assert update_calls.grade_market("ml", "home", None, 24, 17) == "win"
    assert update_calls.grade_market("ml", "away", None, 24, 17) == "loss"
    assert update_calls.grade_market("ml", "home", None, 20, 20) == "push"


def test_grade_market_spread():
    # home covers -2.0
    assert update_calls.grade_market("spread", "home", -2.0, 24, 17) == "win"
    # home fails to cover
    assert update_calls.grade_market("spread", "home", -10.0, 24, 17) == "loss"
    # integer-line push
    assert update_calls.grade_market("spread", "home", -3.0, 20, 17) == "push"


def test_grade_market_total():
    assert update_calls.grade_market("total", "over", 40.5, 24, 17) == "win"
    assert update_calls.grade_market("total", "under", 40.5, 24, 17) == "loss"
    assert update_calls.grade_market("total", "over", 41.0, 24, 17) == "push"


def test_grade_market_no_line():
    assert update_calls.grade_market("spread", "home", None, 24, 17) is None
    assert update_calls.grade_market("total", "over", None, 24, 17) is None


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def test_snapshot_append_only_and_fill_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "data").mkdir()

    game = {
        "game_id": "2026_12_GB_LA",
        "season": 2026,
        "week": 12,
        "commence_time": "2026-11-26T01:00:00Z",
        "home_team": "LA",
        "away_team": "GB",
        "model_home_win_prob": 0.659,
        "markets": {
            "ml": {
                "side": "home",
                "team": "LA",
                "line": None,
                "price": 1.73,
            }
        },
    }
    market = {"odds_fetched_at": "2026-11-25T23:00:00+00:00", "games": [game]}
    upcoming = [
        {
            "game_id": "2026_12_GB_LA",
            "season": 2026,
            "week": 12,
            "home_team": "LA",
            "away_team": "GB",
            "home_win_prob": 0.659,
            "spread": 5.5,
            "total": 50.2,
            "gameday": "2026-11-25",
            "gametime": "20:00",
            "stadium": "SoFi Stadium",
        }
    ]
    _write_json("reports/market_comparison.json", market)
    _write_json("predictions/upcoming_games.json", upcoming)

    monkeypatch.setattr(update_calls, "MARKETS", "reports/market_comparison.json")
    monkeypatch.setattr(update_calls, "UPCOMING", "predictions/upcoming_games.json")
    monkeypatch.setattr(update_calls, "CALLS", "predictions/calls_history.json")
    monkeypatch.setattr(update_calls, "SCOREBOARD", "reports/calls_scoreboard.json")
    monkeypatch.setattr(update_calls, "PICKS", "predictions/game_picks.json")
    monkeypatch.setattr(update_calls, "SCHEDULES", "data/missing.parquet")
    monkeypatch.setattr(update_calls, "final_scores", lambda: {})

    update_calls.main()

    doc = json.load(open("predictions/game_picks.json", encoding="utf-8"))
    assert len(doc["games"]) == 1
    entry = doc["games"][0]
    assert entry["markets"].get("ml", {}).get("price") == 1.73
    assert "spread" not in entry["markets"]

    # Second run: changed ml price + new spread market key.
    game["markets"]["ml"]["price"] = 999.0
    game["markets"]["spread"] = {
        "side": "home",
        "team": "LA",
        "line": -2.0,
        "price": 1.83,
    }
    _write_json("reports/market_comparison.json", market)

    update_calls.main()

    doc2 = json.load(open("predictions/game_picks.json", encoding="utf-8"))
    entry2 = doc2["games"][0]
    # existing ml market not overwritten
    assert entry2["markets"]["ml"]["price"] == 1.73
    # missing market key filled
    assert entry2["markets"]["spread"]["price"] == 1.83
    # metadata not overwritten (snapshot_at stable)
    assert entry2["snapshot_at"] == entry["snapshot_at"]


def test_pick_grading_sets_final_and_grades(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "data").mkdir()

    game = {
        "game_id": "2026_12_GB_LA",
        "season": 2026,
        "week": 12,
        "commence_time": "2026-11-26T01:00:00Z",
        "home_team": "LA",
        "away_team": "GB",
        "model_home_win_prob": 0.659,
        "markets": {
            "ml": {"side": "home", "team": "LA", "line": None, "price": 1.73},
            "spread": {"side": "home", "team": "LA", "line": None, "price": 1.83},
            "total": {"side": "over", "team": "Over", "line": 43.0, "price": 1.93},
        },
    }
    market = {"odds_fetched_at": "2026-11-25T23:00:00+00:00", "games": [game]}
    upcoming = [{"game_id": "2026_12_GB_LA", "season": 2026, "week": 12, "home_team": "LA", "away_team": "GB"}]
    _write_json("reports/market_comparison.json", market)
    _write_json("predictions/upcoming_games.json", upcoming)

    monkeypatch.setattr(update_calls, "MARKETS", "reports/market_comparison.json")
    monkeypatch.setattr(update_calls, "UPCOMING", "predictions/upcoming_games.json")
    monkeypatch.setattr(update_calls, "CALLS", "predictions/calls_history.json")
    monkeypatch.setattr(update_calls, "SCOREBOARD", "reports/calls_scoreboard.json")
    monkeypatch.setattr(update_calls, "PICKS", "predictions/game_picks.json")
    monkeypatch.setattr(update_calls, "SCHEDULES", "data/missing.parquet")
    monkeypatch.setattr(update_calls, "final_scores", lambda: {
        (2026, 12, "GB", "LA"): {"home_score": 24, "away_score": 17}
    })

    update_calls.main()

    doc = json.load(open("predictions/game_picks.json", encoding="utf-8"))
    entry = doc["games"][0]
    assert entry["final"] == {"home": 24, "away": 17}
    assert entry["grades"]["ml"] == "win"
    assert entry["grades"]["spread"] is None
    assert entry["grades"]["total"] == "loss"
    graded_at = entry["graded_at"]
    assert graded_at is not None

    # second run with different final must not overwrite
    monkeypatch.setattr(update_calls, "final_scores", lambda: {
        (2026, 12, "GB", "LA"): {"home_score": 30, "away_score": 30}
    })
    update_calls.main()
    doc2 = json.load(open("predictions/game_picks.json", encoding="utf-8"))
    entry2 = doc2["games"][0]
    assert entry2["final"] == {"home": 24, "away": 17}
    assert entry2["graded_at"] == graded_at
