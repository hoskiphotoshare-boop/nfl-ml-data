import polars as pl

from src.features import add_team_rolling_features, build_player_features


def test_player_fantasy_points_includes_passing_and_rushing():
    df = pl.DataFrame(
        {
            "player_id": ["q1"],
            "player_name": ["QB One"],
            "position": ["QB"],
            "team": ["KC"],
            "season": [2026],
            "week": [1],
            "passing_yards": [300],
            "passing_tds": [3],
            "passing_interceptions": [1],
            "rushing_yards": [20],
            "rushing_tds": [1],
            "receptions": [0],
            "targets": [0],
            "yards": [0],
            "touchdowns": [0],
            "fumbles_lost_total": [0],
        }
    )
    out = build_player_features(df)
    # 300*0.04 + 3*4 - 1*2 + 20*0.1 + 1*6 = 30.0
    assert out["fantasy_points"][0] == 30.0


def test_add_team_rolling_features_uses_prior_games_only():
    match = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["BUF", "BUF"],
        }
    )
    df = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "team": ["KC", "KC"],
            "passing_epa": [1.0, 2.0],
        }
    )
    out = add_team_rolling_features(match, df, "team", ["passing_epa"], "test")
    assert "home_test_passing_epa" in out.columns

    week1 = out.filter(pl.col("week") == 1)["home_test_passing_epa"][0]
    week2 = out.filter(pl.col("week") == 2)["home_test_passing_epa"][0]
    assert week1 == 0.0
    assert week2 == 1.0


def test_weather_requests_are_batched_and_skip_domes():
    import os
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    from fetch_weather import collect_requests

    now = datetime(2026, 11, 27, 5, 0, tzinfo=timezone.utc)
    games = [
        {"home_team": "BUF", "gameday": "2026-11-27", "gametime": "15:00", "roof": "outdoors"},
        {"home_team": "KC", "gameday": "2026-11-27", "gametime": "16:25", "roof": "outdoors"},
        {"home_team": "DET", "gameday": "2026-11-27", "gametime": "15:00", "roof": "dome"},
    ]
    requests = collect_requests(games, now)
    assert set(requests) == {"2026-11-27"}
    assert len(requests["2026-11-27"]) == 2
