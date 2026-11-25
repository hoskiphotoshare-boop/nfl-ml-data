import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import backtest_market  # noqa: E402
import compare_market  # noqa: E402
import fetch_odds  # noqa: E402


def test_fallback_team_names_cover_the_league():
    names = compare_market.FALLBACK_TEAM_NAMES
    assert len(names) == 32
    assert len(set(names.values())) == 32
    assert names["LA"] == "Los Angeles Rams"
    assert names["WAS"] == "Washington Commanders"


def test_fallback_name_map_used_without_nflreadpy(monkeypatch):
    monkeypatch.setattr(compare_market, "TEAMS_AVAILABLE", False)
    names = compare_market.team_name_map()
    assert names["KC"] == "Kansas City Chiefs"


def test_compact_snapshot_keeps_market_core():
    row = fetch_odds.compact(
        {
            "commence_time": "2026-11-29T18:00:00Z",
            "home_team": "Cincinnati Bengals",
            "away_team": "New Orleans Saints",
            "home_spread": -9.5,
            "away_spread": 9.5,
            "total": 50.5,
            "home_moneyline": 1.18,
            "away_moneyline": 5.5,
            "best_home_ml": {"price": 1.2, "bookmaker": "x"},
        }
    )
    assert row["total"] == 50.5
    assert "best_home_ml" not in row
    assert set(row) == {
        "commence_time",
        "home_team",
        "away_team",
        "home_spread",
        "away_spread",
        "total",
        "home_moneyline",
        "away_moneyline",
    }


def test_movement_handles_missing_values():
    assert backtest_market.movement({"home_spread": -2.5}, {"home_spread": -3.0}, "home_spread") == -0.5
    assert backtest_market.movement({"home_spread": None}, {"home_spread": -3.0}, "home_spread") is None
    assert backtest_market.movement({}, {}, "total") is None
