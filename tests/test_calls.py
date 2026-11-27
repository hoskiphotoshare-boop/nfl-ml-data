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
    monkeypatch.setattr(update_calls, "merged_final_scores", lambda: {})

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
    monkeypatch.setattr(update_calls, "merged_final_scores", lambda: {
        (2026, 12, "GB", "LA"): {
            "home_score": 24,
            "away_score": 17,
            "source": "espn",
        }
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
    monkeypatch.setattr(update_calls, "merged_final_scores", lambda: {
        (2026, 12, "GB", "LA"): {
            "home_score": 30,
            "away_score": 30,
            "source": "espn",
        }
    })
    update_calls.main()
    doc2 = json.load(open("predictions/game_picks.json", encoding="utf-8"))
    entry2 = doc2["games"][0]
    assert entry2["final"] == {"home": 24, "away": 17}
    assert entry2["graded_at"] == graded_at


def test_merged_scores_prefers_schedules(monkeypatch):
    key = (2026, 12, "CHI", "DET")
    monkeypatch.setattr(
        update_calls,
        "final_scores",
        lambda: {key: {"home_score": 35, "away_score": 39}},
    )
    monkeypatch.setattr(
        update_calls,
        "espn_final_scores",
        lambda: {key: {"home_score": 99, "away_score": 0, "source": "espn"}},
    )
    merged = update_calls.merged_final_scores()
    assert merged[key]["home_score"] == 35
    assert merged[key]["source"] == "schedules"


def test_merged_scores_uses_espn_fallback(monkeypatch):
    key = (2026, 12, "CHI", "DET")
    monkeypatch.setattr(update_calls, "final_scores", lambda: {})
    monkeypatch.setattr(
        update_calls,
        "espn_final_scores",
        lambda: {key: {"home_score": 35, "away_score": 39, "source": "espn"}},
    )
    merged = update_calls.merged_final_scores()
    assert merged[key]["away_score"] == 39
    assert merged[key]["source"] == "espn"


def test_closing_updates_model_probability_and_edge(monkeypatch):
    call = {
        "call_id": "x",
        "game_id": "g",
        "status": "open",
        "kickoff": "2026-11-26T18:00:00+00:00",
        "market": "spread",
        "side": "home",
        "team": "DET",
        "edge": 0.10,
        "edge_first": 0.10,
    }
    existing = {"x": call}
    games = [
        {
            "game_id": "g",
            "markets": {
                "spread": {
                    "side": "home",
                    "team": "DET",
                    "line": 1.5,
                    "price": 1.96,
                    "model_prob": 0.58,
                    "market_prob": 0.5,
                    "edge": 0.08,
                }
            },
        }
    ]
    now = update_calls.datetime(2026, 11, 26, 12, 0, tzinfo=update_calls.timezone.utc)
    monkeypatch.setattr(update_calls, "parse_time", lambda value: now)
    # parse_time is also used for the kickoff; a fixed kickoff after now keeps
    # the call open.
    monkeypatch.setattr(
        update_calls,
        "parse_time",
        lambda value: update_calls.datetime(
            2026, 11, 26, 18, 0, tzinfo=update_calls.timezone.utc
        ),
    )
    updated = update_calls.update_closing(games, existing, now)
    assert updated == 1
    assert call["closing_model_prob"] == 0.58
    assert call["closing_edge"] == 0.08
    assert call["edge_delta"] == -0.02


def test_grade_pick_archived_game_not_in_market():
    now = update_calls.datetime(2026, 11, 27, 5, 0, tzinfo=update_calls.timezone.utc)
    pick = {
        "game_id": "2026_12_KC_BUF",
        "season": 2026,
        "week": 12,
        "away_team": "KC",
        "home_team": "BUF",
        "markets": {
            "ml": {"side": "home", "team": "BUF", "line": None},
            "spread": {"side": "home", "team": "BUF", "line": -2.5},
            "total": {"side": "over", "line": 45.5},
        },
        "final": None,
        "grades": None,
        "graded_at": None,
        "final_source": None,
    }
    scores = {
        (2026, 12, "KC", "BUF"): {
            "home_score": 13,
            "away_score": 10,
            "source": "espn",
        }
    }
    assert update_calls.grade_pick(pick, scores, now)
    assert pick["final"] == {"home": 13, "away": 10}
    assert pick["final_source"] == "espn"
    assert pick["grades"] == {"ml": "win", "spread": "win", "total": "loss"}


def test_build_pick_entry_persists_market_guards():
    now = update_calls.datetime(2026, 11, 26, 12, 0, tzinfo=update_calls.timezone.utc)
    game = {
        "game_id": "2026_12_KC_BUF",
        "season": 2026,
        "week": 12,
        "away_team": "KC",
        "home_team": "BUF",
        "commence_time": "2026-11-27T01:20:00Z",
        "markets": {"total": {"side": "over", "edge": 0.12, "actionable_edge": 0.0}},
        "market_movement": {"total_move": -2.0},
        "movement_against_any": True,
        "weather_risk": True,
        "blend_conflict": True,
        "suppressed_edges": ["total"],
    }
    entry = update_calls.build_pick_entry(game, {}, now, "2026-11-26T12:00:00Z")
    assert entry["movement_against_any"] is True
    assert entry["weather_risk"] is True
    assert entry["blend_conflict"] is True
    assert entry["suppressed_edges"] == ["total"]
    assert entry["market_movement"] == {"total_move": -2.0}
