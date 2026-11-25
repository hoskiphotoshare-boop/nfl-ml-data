import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import polars as pl  # noqa: E402

import simulate_playoffs as sp  # noqa: E402


def test_rating_probability_gives_home_edge():
    home = sp.rating_probability({}, "SF", "CHI")
    neutral = sp.rating_probability({}, "SF", "CHI", neutral=True)
    assert 0.5 < home < 0.6
    assert neutral == 0.5


def test_rating_probability_respects_strength():
    rating = {"SF": 10.0, "CHI": 0.0}
    assert sp.rating_probability(rating, "SF", "CHI") > 0.75
    assert sp.rating_probability(rating, "CHI", "SF") < 0.30


def test_srs_ratings_are_opponent_adjusted():
    schedules = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "home_team": ["SF", "SF", "CHI"],
            "away_team": ["CHI", "ARI", "ARI"],
            "home_score": [30, 20, 17],
            "away_score": [10, 10, 10],
        }
    )
    ratings = sp.srs_ratings(schedules, 2026)
    assert ratings["SF"] > ratings["CHI"] > ratings["ARI"]
