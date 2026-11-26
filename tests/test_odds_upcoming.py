import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from datetime import datetime, timedelta, timezone

from compare_market import is_blend_conflict
from fetch_odds import is_upcoming

NOW = datetime(2026, 11, 26, 18, 30, tzinfo=timezone.utc)


def test_odds_filter_drops_started_games():
    assert not is_upcoming({"commence_time": "2026-11-26T18:00:00Z"}, NOW)
    assert is_upcoming({"commence_time": "2026-11-26T19:00:00Z"}, NOW)
    assert is_upcoming({}, NOW)


def test_blend_conflict_threshold():
    assert is_blend_conflict(-1.58, 0.641)
    assert not is_blend_conflict(-1.40, 0.641)
    assert not is_blend_conflict(-1.58, 0.58)
    assert not is_blend_conflict(None, 0.641)
