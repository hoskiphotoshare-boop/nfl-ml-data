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
