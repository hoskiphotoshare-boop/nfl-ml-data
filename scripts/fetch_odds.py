#!/usr/bin/env python3
"""Fetch NFL odds from The Odds API and write ``market/odds_latest.json``.

Requires the ``ODDS_API_KEY`` environment variable (GitHub Actions secret).
If the key is missing or the API call fails, the script exits 0 so the rest of
the pipeline can continue.

Quota model (free tier): a request costs ``markets × regions`` credits, so the
three markets used here (h2h, spreads, totals) in one region cost 3 credits.
The scheduled workflow is budgeted to stay around half of the 500 credits/month.

Optional guards (set by the workflow) keep off-season and bye-week runs from
burning credits:

- ``ODDS_REQUIRE_GAME_WITHIN_DAYS``: skip unless a game kicks off soon.
- ``ODDS_REQUIRE_GAME_WITHIN_HOURS``: tighter guard for pre-game runs.
- ``ODDS_MIN_REMAINING``: skip when the last known quota is nearly exhausted.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"
BASE_URL = "https://api.the-odds-api.com/v4/sports"
META_PATH = "market/odds_meta.json"
PREDICTIONS_PATH = "predictions/upcoming_games.json"

CACHE_HOURS = float(os.environ.get("ODDS_CACHE_HOURS", "6"))
MIN_REMAINING = int(os.environ.get("ODDS_MIN_REMAINING", "12"))
REQUIRE_DAYS = os.environ.get("ODDS_REQUIRE_GAME_WITHIN_DAYS")
REQUIRE_HOURS = os.environ.get("ODDS_REQUIRE_GAME_WITHIN_HOURS")
CREDIT_COST = len([m for m in MARKETS.split(",") if m]) * len(REGIONS.split(","))


def read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def game_datetime_utc(game: dict) -> datetime | None:
    day = game.get("gameday")
    if not day:
        return None
    clock = game.get("gametime") or "00:00"
    try:
        local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            local = datetime.strptime(str(day), "%Y-%m-%d")
        except ValueError:
            return None
    try:
        from zoneinfo import ZoneInfo

        local = local.replace(tzinfo=ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - tzdata fallback
        local = local.replace(tzinfo=timezone(timedelta(hours=-4)))
    return local.astimezone(timezone.utc)


def next_kickoff() -> datetime | None:
    games = read_json(PREDICTIONS_PATH, [])
    now = datetime.now(timezone.utc)
    upcoming = [dt for dt in (game_datetime_utc(g) for g in games) if dt and dt >= now]
    return min(upcoming) if upcoming else None


def guard_allows_fetch() -> bool:
    """Return False when the current run should not spend credits."""
    kickoff = next_kickoff()
    now = datetime.now(timezone.utc)

    if REQUIRE_DAYS:
        limit = timedelta(days=float(REQUIRE_DAYS))
        if kickoff is None or kickoff - now > limit:
            print(
                f"Odds guard: no kickoff within {float(REQUIRE_DAYS):g} days "
                "(off-season or stale predictions); skipping API call."
            )
            return False

    if REQUIRE_HOURS:
        limit = timedelta(hours=float(REQUIRE_HOURS))
        if kickoff is None or kickoff - now > limit:
            print(
                f"Odds guard: next kickoff is "
                f"{'none' if kickoff is None else f'{(kickoff - now).total_seconds() / 3600:.1f}h away'}"
                f" (> {float(REQUIRE_HOURS):g}h); skipping API call."
            )
            return False

    return True


def recent_cache() -> bool:
    """Return True when a recent odds snapshot exists and force-fetch is off."""
    if os.environ.get("FORCE_ODDS_FETCH") == "1":
        return False
    meta = read_json(META_PATH, {})
    fetched_at = meta.get("fetched_at")
    if not fetched_at:
        return False
    try:
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
    except Exception:
        return False
    if age_hours < CACHE_HOURS:
        print(
            f"Odds cache is {age_hours * 60:.0f}m old (< {CACHE_HOURS * 60:g}m); "
            "skipping Odds API call."
        )
        return True
    return False


def budget_exhausted() -> bool:
    meta = read_json(META_PATH, {})
    remaining = meta.get("requests_remaining")
    if remaining is None:
        return False
    try:
        remaining = int(remaining)
    except Exception:
        return False
    if remaining < MIN_REMAINING:
        print(
            f"Odds quota low ({remaining} credits left, floor {MIN_REMAINING}); "
            "skipping API call to preserve the monthly allowance."
        )
        return True
    return False


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fetch_odds(api_key: str) -> tuple[list[dict], str | None, str | None]:
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
    print(f"Odds API requests — used: {used}, remaining: {remaining} (cost {CREDIT_COST})")
    return data, used, remaining


def best_ml(entries):
    return max(entries, key=lambda x: x[0]) if entries else None


def best_spread_entry(entries):
    if not entries:
        return None
    best_point = max(e[0] for e in entries)
    same = [e for e in entries if e[0] == best_point]
    return max(same, key=lambda x: x[1])


def best_total_entry(entries, side: str):
    if not entries:
        return None
    target = min(e[0] for e in entries) if side == "over" else max(e[0] for e in entries)
    same = [e for e in entries if e[0] == target]
    return max(same, key=lambda x: x[1])


def summarise_event(event: dict) -> dict:
    home = event["home_team"]
    away = event["away_team"]

    home_ml_entries = []
    away_ml_entries = []
    home_spread_entries = []
    away_spread_entries = []
    over_entries = []
    under_entries = []
    books: set[str] = set()

    for bookmaker in event.get("bookmakers", []):
        book = bookmaker.get("key", "unknown")
        books.add(book)
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            for o in market.get("outcomes", []):
                name = o.get("name")
                price = o.get("price")
                point = o.get("point")
                if price is None:
                    continue
                if key == "h2h":
                    if name == home:
                        home_ml_entries.append((float(price), book))
                    elif name == away:
                        away_ml_entries.append((float(price), book))
                elif key == "spreads" and point is not None:
                    if name == home:
                        home_spread_entries.append((float(point), float(price), book))
                    elif name == away:
                        away_spread_entries.append((float(point), float(price), book))
                elif key == "totals" and point is not None:
                    if str(name).lower() == "over":
                        over_entries.append((float(point), float(price), book))
                    elif str(name).lower() == "under":
                        under_entries.append((float(point), float(price), book))

    home_spread = median([e[0] for e in home_spread_entries])
    away_spread = median([e[0] for e in away_spread_entries])
    total = median([e[0] for e in over_entries + under_entries])
    home_ml = median([e[0] for e in home_ml_entries])
    away_ml = median([e[0] for e in away_ml_entries])

    market_home_prob = None
    if home_ml and away_ml and home_ml > 0 and away_ml > 0:
        raw_home = 1.0 / home_ml
        raw_away = 1.0 / away_ml
        total_raw = raw_home + raw_away
        if total_raw > 0:
            market_home_prob = raw_home / total_raw

    best_home_ml = best_ml(home_ml_entries)
    best_away_ml = best_ml(away_ml_entries)
    best_home_spread = best_spread_entry(home_spread_entries)
    best_away_spread = best_spread_entry(away_spread_entries)
    best_over = best_total_entry(over_entries, "over")
    best_under = best_total_entry(under_entries, "under")

    def spread_payload(entry):
        return (
            {"point": entry[0], "price": entry[1], "bookmaker": entry[2]}
            if entry
            else None
        )

    def total_payload(entry):
        return (
            {"point": entry[0], "price": entry[1], "bookmaker": entry[2]}
            if entry
            else None
        )

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
        "best_home_ml": {"price": best_home_ml[0], "bookmaker": best_home_ml[1]}
        if best_home_ml
        else None,
        "best_away_ml": {"price": best_away_ml[0], "bookmaker": best_away_ml[1]}
        if best_away_ml
        else None,
        "best_home_spread": spread_payload(best_home_spread),
        "best_away_spread": spread_payload(best_away_spread),
        "best_over": total_payload(best_over),
        "best_under": total_payload(best_under),
        "bookmakers": sorted(books),
    }


def compact(event: dict) -> dict:
    """Small snapshot record used for line-movement history."""
    return {
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "home_spread": event.get("home_spread"),
        "away_spread": event.get("away_spread"),
        "total": event.get("total"),
        "home_moneyline": event.get("home_moneyline"),
        "away_moneyline": event.get("away_moneyline"),
    }


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set; skipping odds fetch.")
        return

    if not guard_allows_fetch():
        return

    if recent_cache() or budget_exhausted():
        return

    try:
        data, used, remaining = fetch_odds(api_key)
    except urllib.error.HTTPError as exc:
        print(f"Odds API HTTP error: {exc.code} {exc.reason}")
        return
    except Exception as exc:  # pragma: no cover
        print(f"Odds API error: {exc}")
        return

    events = [summarise_event(e) for e in data]
    now = datetime.now(timezone.utc)
    os.makedirs("market", exist_ok=True)
    with open("market/odds_latest.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fetched_at": now.isoformat(),
                "events": len(events),
                "credits": CREDIT_COST,
                "requests_used": used,
                "requests_remaining": remaining,
            },
            f,
            indent=2,
        )

    history_dir = "market/history"
    os.makedirs(history_dir, exist_ok=True)
    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    with open(f"{history_dir}/{stamp}.json", "w", encoding="utf-8") as f:
        json.dump([compact(e) for e in events], f, indent=2)

    print(
        f"Wrote market/odds_latest.json ({len(events)} events) and "
        f"history/{stamp}.json (cost {CREDIT_COST} credits)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Odds fetch error: {exc}")
    sys.exit(0)
