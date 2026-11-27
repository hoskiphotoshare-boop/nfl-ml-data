import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from datetime import datetime, timezone

from predict_upcoming import kickoff_utc


def test_kickoff_utc_uses_eastern_time():
    got = kickoff_utc("2026-11-26", "13:00")
    assert got == datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc)


def test_kickoff_utc_handles_missing_time():
    got = kickoff_utc("2026-11-26", None)
    assert got.hour == 5  # 00:00 ET is 05:00 UTC in November


def test_kickoff_utc_rejects_bad_date():
    assert kickoff_utc("not-a-date", "13:00") is None


def test_merge_weather_keeps_previous_forecast():
    from predict_upcoming import merge_weather

    previous = {
        "temp": 33.5,
        "wind": 13.8,
        "gust": 40.5,
        "snow": 2.9,
        "weather_fetched_at": "2026-11-27T05:00:00Z",
    }
    merged = merge_weather({"temp": None, "wind": None}, previous)
    assert merged["temp"] == 33.5
    assert merged["gust"] == 40.5
    assert merged["snow"] == 2.9
    assert merged["weather_fetched_at"] == "2026-11-27T05:00:00Z"

    fresh = merge_weather({"temp": 40.0, "wind": 8.0}, previous)
    assert fresh["temp"] == 40.0
    assert fresh["wind"] == 8.0
    assert fresh["gust"] == 40.5
