#!/usr/bin/env python3
"""Attach kickoff weather forecasts to upcoming games.

Uses the free Open-Meteo API (no key) for outdoor games within the forecast
horizon. Indoor games are left alone; their roof is already shown.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

PREDICTIONS = "predictions/upcoming_games.json"
FORECAST_DAYS = 16

STADIUMS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2581),
    "NE": (42.0909, -71.2643),
    "NO": (29.9509, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9694),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9077, -76.8645),
}


def kickoff_utc(game: dict) -> datetime | None:
    day = game.get("gameday")
    if not day:
        return None
    clock = game.get("gametime") or "13:00"
    try:
        local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    try:
        from zoneinfo import ZoneInfo

        local = local.replace(tzinfo=ZoneInfo("America/New_York"))
    except Exception:
        local = local.replace(tzinfo=timezone(timedelta(hours=-4)))
    return local.astimezone(timezone.utc)


def fetch_forecast(team: str, day: str, cache: dict) -> tuple[float, float, float] | None:
    coords = STADIUMS.get(team)
    if not coords:
        return None
    key = (coords, day)
    if key in cache:
        return cache[key]
    query = urllib.parse.urlencode(
        {
            "latitude": coords[0],
            "longitude": coords[1],
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
            "start_date": day,
            "end_date": day,
            "timezone": "UTC",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
        hourly = data.get("hourly", {})
        cache[key] = hourly
        return hourly
    except Exception:
        cache[key] = None
        return None


def hourly_value(hourly: dict, key: str, timestamp: datetime) -> float | None:
    times = hourly.get("time") or []
    values = hourly.get(key) or []
    if not times:
        return None
    target = timestamp.strftime("%Y-%m-%dT%H:00")
    try:
        index = times.index(target)
    except ValueError:
        index = min(
            range(len(times)),
            key=lambda i: abs(
                datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - timestamp
            ),
        )
    if index >= len(values):
        return None
    value = values[index]
    return round(float(value), 1) if value is not None else None


def main() -> None:
    if not os.path.exists(PREDICTIONS):
        print("No upcoming games; skipping weather.")
        return
    games = json.load(open(PREDICTIONS, encoding="utf-8"))
    now = datetime.now(timezone.utc)
    cache: dict = {}
    updated = 0

    for game in games:
        if (game.get("roof") or "").lower() in {"dome", "closed"}:
            continue
        kickoff = kickoff_utc(game)
        if kickoff is None or kickoff < now or kickoff > now + timedelta(days=FORECAST_DAYS):
            continue
        day = kickoff.strftime("%Y-%m-%d")
        hourly = fetch_forecast(game.get("home_team", ""), day, cache)
        if not hourly:
            continue
        game["temp"] = hourly_value(hourly, "temperature_2m", kickoff)
        game["wind"] = hourly_value(hourly, "wind_speed_10m", kickoff)
        game["precip"] = hourly_value(hourly, "precipitation_probability", kickoff)
        if game.get("temp") is not None or game.get("wind") is not None:
            updated += 1

    with open(PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
    print(f"Added weather forecasts to {updated}/{len(games)} games")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Weather fetch error: {exc}")
