#!/usr/bin/env python3
"""Fetch NFL odds from The Odds API and write ``market/odds_latest.json``.

Requires the ``ODDS_API_KEY`` environment variable (GitHub Actions secret).
If the key is missing or the API call fails, the script exits 0 so the rest of
the pipeline can continue.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"
BASE_URL = "https://api.the-odds-api.com/v4/sports"
META_PATH = "market/odds_meta.json"
CACHE_HOURS = float(os.environ.get("ODDS_CACHE_HOURS", "6"))


def recent_cache() -> bool:
    """Return True when a recent odds snapshot exists and force-fetch is off."""
    if os.environ.get("FORCE_ODDS_FETCH") == "1":
        return False
    if not os.path.exists(META_PATH):
        return False
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours < CACHE_HOURS:
            print(
                f"Odds cache is {age_hours:.1f}h old (< {CACHE_HOURS:g}h); "
                "skipping Odds API call."
            )
            return True
    except Exception:
        pass
    return False


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fetch_odds(api_key: str) -> list[dict]:
    url = (
        f"{BASE_URL}/{SPORT}/odds/"
        f"?apiKey={api_key}&regions={REGIONS}&markets={MARKETS}"
        f"&oddsFormat={ODDS_FORMAT}"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        data = json.load(resp)
    print(f"Odds API requests — used: {used}, remaining: {remaining}")
    return data


def summarise_event(event: dict) -> dict:
    home = event["home_team"]
    away = event["away_team"]

    home_spreads: list[float] = []
    away_spreads: list[float] = []
    totals: list[float] = []
    home_prices: list[float] = []
    away_prices: list[float] = []
    books: set[str] = set()

    for bookmaker in event.get("bookmakers", []):
        books.add(bookmaker.get("key", "unknown"))
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])
            if key == "spreads":
                for o in outcomes:
                    if o.get("name") == home and o.get("point") is not None:
                        home_spreads.append(float(o["point"]))
                    elif o.get("name") == away and o.get("point") is not None:
                        away_spreads.append(float(o["point"]))
            elif key == "totals":
                for o in outcomes:
                    if o.get("point") is not None:
                        totals.append(float(o["point"]))
            elif key == "h2h":
                for o in outcomes:
                    price = o.get("price")
                    if price is None:
                        continue
                    if o.get("name") == home:
                        home_prices.append(float(price))
                    elif o.get("name") == away:
                        away_prices.append(float(price))

    home_spread = median(home_spreads)
    away_spread = median(away_spreads)
    total = median(totals)
    home_ml = median(home_prices)
    away_ml = median(away_prices)

    market_home_prob = None
    if home_ml and away_ml and home_ml > 0 and away_ml > 0:
        raw_home = 1.0 / home_ml
        raw_away = 1.0 / away_ml
        total_raw = raw_home + raw_away
        if total_raw > 0:
            market_home_prob = raw_home / total_raw

    return {
        "commence_time": event.get("commence_time"),
        "home_team": home,
        "away_team": away,
        "home_spread": home_spread,
        "away_spread": away_spread,
        "total": total,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "market_home_win_prob": round(market_home_prob, 4) if market_home_prob else None,
        "bookmakers": sorted(books),
    }


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set; skipping odds fetch.")
        return

    if recent_cache():
        return

    try:
        data = fetch_odds(api_key)
    except urllib.error.HTTPError as exc:
        print(f"Odds API HTTP error: {exc.code} {exc.reason}")
        return
    except Exception as exc:  # pragma: no cover
        print(f"Odds API error: {exc}")
        return

    events = [summarise_event(e) for e in data]
    os.makedirs("market", exist_ok=True)
    with open("market/odds_latest.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "events": len(events),
            },
            f,
            indent=2,
        )
    print(f"Wrote market/odds_latest.json ({len(events)} events)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Odds fetch error: {exc}")
    sys.exit(0)