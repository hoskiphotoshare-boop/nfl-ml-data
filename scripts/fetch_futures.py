#!/usr/bin/env python3
"""Fetch Super Bowl futures (outrights) from The Odds API.

Sport key: ``americanfootball_nfl_super_bowl_winner``
Market:    ``outrights`` (1 credit per request, single region)

Writes ``market/futures_latest.json`` with per-team median/best prices and a
vig-free implied probability, plus a daily snapshot for movement tracking.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

SPORT = "americanfootball_nfl_super_bowl_winner"
REGIONS = "us"
MARKETS = "outrights"
ODDS_FORMAT = "decimal"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
META_PATH = "market/futures_meta.json"
OUT_PATH = "market/futures_latest.json"
PREDICTIONS_PATH = "predictions/playoffs.json"

CACHE_HOURS = float(os.environ.get("FUTURES_CACHE_HOURS", "12"))
MIN_REMAINING = int(os.environ.get("ODDS_MIN_REMAINING", "12"))
CREDIT_COST = 1


def read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def recent_cache() -> bool:
    if os.environ.get("FORCE_ODDS_FETCH") == "1":
        return False
    meta = read_json(META_PATH, {})
    fetched_at = meta.get("fetched_at")
    if not fetched_at:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
    except Exception:
        return False
    if age < CACHE_HOURS:
        print(f"Futures cache is {age:.1f}h old (< {CACHE_HOURS:g}h); skipping API call.")
        return True
    return False


def budget_exhausted() -> bool:
    meta = read_json("market/odds_meta.json", {})
    remaining = meta.get("requests_remaining")
    try:
        remaining = int(remaining)
    except Exception:
        return False
    if remaining < MIN_REMAINING:
        print(f"Odds quota low ({remaining} left); skipping futures fetch.")
        return True
    return False


def fetch(api_key: str):
    query = (
        f"{URL}?apiKey={api_key}&regions={REGIONS}&markets={MARKETS}"
        f"&oddsFormat={ODDS_FORMAT}"
    )
    req = urllib.request.Request(query, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        data = json.load(resp)
    print(f"Futures API — used: {used}, remaining: {remaining} (cost {CREDIT_COST})")
    return data, used, remaining


def summarise(events: list[dict]) -> dict:
    prices: dict[str, dict[str, float]] = {}
    books: set[str] = set()
    commence = None
    title = "NFL Super Bowl Winner"

    for event in events:
        commence = commence or event.get("commence_time")
        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key", "unknown")
            books.add(book)
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    team = outcome.get("name")
                    price = outcome.get("price")
                    if not team or price is None:
                        continue
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    if price <= 1:
                        continue
                    prices.setdefault(team, {})[book] = price

    rows = []
    for team, book_prices in prices.items():
        values = list(book_prices.values())
        median_price = statistics.median(values)
        best_book = max(book_prices, key=book_prices.get)
        rows.append(
            {
                "team": team,
                "best_price": round(book_prices[best_book], 2),
                "best_bookmaker": best_book,
                "median_price": round(median_price, 2),
                "books": len(book_prices),
            }
        )

    total_implied = sum(1.0 / row["median_price"] for row in rows if row["median_price"] > 1)
    for row in rows:
        implied = 1.0 / row["median_price"] if row["median_price"] > 1 else None
        row["raw_implied"] = round(implied, 5) if implied else None
        row["market_prob"] = round(implied / total_implied, 4) if implied and total_implied else None

    rows.sort(key=lambda r: r.get("market_prob") or 0, reverse=True)
    return {
        "sport": SPORT,
        "title": title,
        "commence_time": commence,
        "bookmakers": sorted(books),
        "overround": round(total_implied - 1.0, 4) if total_implied else None,
        "teams": rows,
    }


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set; skipping futures fetch.")
        return

    # Only fetch when the model has playoff odds to compare against.
    if not os.path.exists(PREDICTIONS_PATH):
        print("No playoff predictions found; skipping futures fetch.")
        return

    if recent_cache() or budget_exhausted():
        return

    try:
        data, used, remaining = fetch(api_key)
    except urllib.error.HTTPError as exc:
        print(f"Futures API HTTP error: {exc.code} {exc.reason}")
        return
    except Exception as exc:  # pragma: no cover
        print(f"Futures API error: {exc}")
        return

    if not data:
        print("Futures API returned no events.")
        return

    payload = summarise(data)
    now = datetime.now(timezone.utc)
    payload["fetched_at"] = now.isoformat()

    os.makedirs("market", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fetched_at": now.isoformat(),
                "teams": len(payload["teams"]),
                "credits": CREDIT_COST,
                "requests_used": int(used) if str(used).isdigit() else used,
                "requests_remaining": int(remaining) if str(remaining).isdigit() else remaining,
            },
            f,
            indent=2,
        )

    history_dir = "market/futures_history"
    os.makedirs(history_dir, exist_ok=True)
    with open(f"{history_dir}/{now.strftime('%Y-%m-%d')}.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "team": row["team"],
                    "market_prob": row.get("market_prob"),
                    "best_price": row.get("best_price"),
                }
                for row in payload["teams"]
            ],
            f,
            indent=2,
        )

    print(
        f"Wrote {OUT_PATH} ({len(payload['teams'])} teams, "
        f"overround {payload['overround']})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Futures fetch error: {exc}")
    sys.exit(0)
