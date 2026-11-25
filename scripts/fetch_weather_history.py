#!/usr/bin/env python3
"""Backfill historical game weather from Open-Meteo.

Fetches hourly reanalysis (ERA5) and historical forecast-archive data for each
outdoor home game, then records the values at kickoff. Requests cover a small
window around game day, are cached per stadium-day-source, and run with modest
concurrency so the free API stays reliable.

Writes ``features/weather_games.parquet`` keyed by ``game_id``. Dome/closed
games are marked but left without weather values.

Run once manually; the daily workflow only uses ``fetch_weather.py`` for the
current 16-day forecast window.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_weather import STADIUMS, kickoff_utc  # noqa: E402

SCHEDULES = "data/schedules.parquet"
OUTPUT = "features/weather_games.parquet"
CACHE_DIR = "data/weather_cache"
SEASONS_BACK = 5
MAX_WORKERS = 4
TIMEOUT = 40
RETRIES = 3

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
]

SOURCES = {
    "actual": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}


def cache_path(source: str, team: str, day: str) -> str:
    return os.path.join(CACHE_DIR, f"{source}_{team}_{day}.json")


_local = threading.local()


def session() -> requests.Session:
    """One keep-alive session per worker to avoid repeated TLS handshakes."""
    current = getattr(_local, "session", None)
    if current is None:
        current = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        current.mount("https://", adapter)
        _local.session = current
    return current


def fetch_hourly(url: str, lat: float, lon: float, start: str, end: str, path: str):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                hourly = json.load(handle)
            if hourly and hourly.get("time"):
                return hourly
        except Exception:
            pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
    }
    for attempt in range(RETRIES):
        try:
            response = session().get(url, params=params, timeout=(10, TIMEOUT))
            response.raise_for_status()
            hourly = response.json().get("hourly")
            if hourly and hourly.get("time"):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(hourly, handle)
                return hourly
        except Exception:
            if attempt + 1 < RETRIES:
                time.sleep(1.5 * (attempt + 1))
    return None


def value_at(hourly: dict, key: str, timestamp: datetime):
    if not hourly:
        return None
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
    if index >= len(values) or values[index] is None:
        return None
    return round(float(values[index]), 2)


def main() -> None:
    if not os.path.exists(SCHEDULES):
        print("No schedules; skipping weather backfill.")
        return
    schedules = pl.read_parquet(SCHEDULES)
    current = int(schedules["season"].max())
    games = schedules.filter(
        (pl.col("season") >= current - SEASONS_BACK + 1)
        & pl.col("home_score").is_not_null()
    )
    if games.is_empty():
        print("No played games in range; skipping weather backfill.")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    tasks = {}
    for game in games.iter_rows(named=True):
        if str(game.get("roof") or "").lower() in {"dome", "closed"}:
            continue
        team = game.get("home_team")
        day = game.get("gameday")
        coords = STADIUMS.get(team)
        if not coords or not day:
            continue
        start = (datetime.fromisoformat(str(day)) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        end = (datetime.fromisoformat(str(day)) + timedelta(days=1)).strftime("%Y-%m-%d")
        for source, url in SOURCES.items():
            key = (source, team, str(day))
            if key in tasks:
                continue
            tasks[key] = (
                url,
                coords[0],
                coords[1],
                start,
                end,
                cache_path(source, team, str(day)),
            )

    cache: dict = {}
    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_hourly, *task): key for key, task in tasks.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                hourly = future.result()
            except Exception:
                hourly = None
            cache[key] = hourly
            if hourly:
                ok += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  weather requests {done}/{len(futures)} ({ok} usable)")

    rows = []
    for game in games.iter_rows(named=True):
        team = game["home_team"]
        season = int(game["season"])
        day = str(game.get("gameday"))
        kickoff = kickoff_utc(game)
        dome = str(game.get("roof") or "").lower() in {"dome", "closed"}

        record = {
            "game_id": game.get("game_id"),
            "season": season,
            "week": game.get("week"),
            "home_team": team,
            "away_team": game.get("away_team"),
            "is_dome": int(dome),
        }
        for source in SOURCES:
            data = cache.get((source, team, day))
            for name, key in (
                ("temp", "temperature_2m"),
                ("wind", "wind_speed_10m"),
                ("gust", "wind_gusts_10m"),
                ("precip", "precipitation"),
                ("snow", "snowfall"),
                ("humidity", "relative_humidity_2m"),
            ):
                record[f"{source}_{name}"] = (
                    None if dome or kickoff is None else value_at(data, key, kickoff)
                )
        rows.append(record)

    frame = pl.DataFrame(rows)
    frame.write_parquet(OUTPUT)
    covered = frame.filter(pl.col("forecast_temp").is_not_null()).height
    actual = frame.filter(pl.col("actual_temp").is_not_null()).height
    print(
        f"Wrote {OUTPUT} ({frame.height} games, "
        f"{actual} with actual weather, {covered} with forecast weather, "
        f"{int(frame['is_dome'].sum())} dome/closed)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Weather backfill error: {exc}")
