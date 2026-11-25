import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import compare_futures  # noqa: E402
import fetch_futures  # noqa: E402


EVENTS = [
    {
        "commence_time": "2027-02-08T00:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "outrights",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": 6.5},
                            {"name": "San Francisco 49ers", "price": 7.5},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "outrights",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": 7.0},
                            {"name": "San Francisco 49ers", "price": 7.0},
                        ],
                    }
                ],
            },
        ],
    }
]


def test_summarise_picks_best_price_and_normalises():
    payload = fetch_futures.summarise(EVENTS)
    assert payload["sport"] == "americanfootball_nfl_super_bowl_winner"
    assert len(payload["teams"]) == 2
    assert abs(sum(t["market_prob"] for t in payload["teams"]) - 1.0) < 1e-6
    chiefs = next(t for t in payload["teams"] if t["team"] == "Kansas City Chiefs")
    assert chiefs["best_price"] == 7.0
    assert chiefs["best_bookmaker"] == "fanduel"
    assert chiefs["books"] == 2


def test_team_map_has_all_franchises():
    assert len(compare_futures.FALLBACK_TEAM_NAMES) == 32
    assert compare_futures.FALLBACK_TEAM_NAMES["SF"] == "San Francisco 49ers"
