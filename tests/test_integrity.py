import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_market_has_no_games_started_at_fetch():
    market = load("reports/market_comparison.json")
    fetched = parse(market["odds_fetched_at"])
    started = [
        game["game_id"]
        for game in market.get("games", [])
        if game.get("commence_time") and parse(game["commence_time"]) <= fetched
    ]
    assert not started, f"pregame market report contains started games: {started}"


def test_every_final_pick_is_graded_with_a_source():
    picks = load("predictions/game_picks.json")
    for game in picks.get("games", []):
        if not game.get("final"):
            continue
        assert isinstance(game.get("grades"), dict), game["game_id"]
        assert set(game["grades"]) >= {"ml", "spread", "total"}, game["game_id"]
        assert game.get("final_source") in {"schedules", "espn"}, game["game_id"]


def test_graded_calls_have_result_profit_and_source():
    calls = load("predictions/calls_history.json")
    for call in calls:
        if call.get("status") != "graded":
            continue
        assert call.get("result") in {"win", "loss", "push"}, call.get("call_id")
        assert call.get("profit") is not None, call.get("call_id")
        assert call.get("final_source") in {"schedules", "espn"}, call.get("call_id")


def test_health_integrity_has_no_open_issues():
    health = load("reports/model_health.json")
    integrity = health.get("integrity", {})
    assert not integrity.get("market_started"), integrity.get("market_started")
    assert not integrity.get("open_calls_past_kickoff_6h"), integrity.get(
        "open_calls_past_kickoff_6h"
    )
    assert not integrity.get("final_picks_ungraded"), integrity.get(
        "final_picks_ungraded"
    )
    assert not integrity.get("final_picks_missing_source"), integrity.get(
        "final_picks_missing_source"
    )
