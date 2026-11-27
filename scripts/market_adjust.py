#!/usr/bin/env python3
"""Inference-time market-informed adjustment of model outputs.

No historical movement coefficient is trained, so this is a bounded,
transparent blend toward market prices/lines when the market has moved.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(__file__))

try:
    import joblib

    _SCORE_BUNDLE = joblib.load("models/score_models.joblib")
    SIGMA_MARGIN = float(_SCORE_BUNDLE.get("sigma_margin", 13.5))
except Exception:
    _SCORE_BUNDLE = None
    SIGMA_MARGIN = 13.5

try:
    from compare_market import (
        team_name_map,
        pregame_history,
        movement_for_event,
        choose_event,
        MOVEMENT_TOTAL,
        MOVEMENT_SPREAD,
        MOVEMENT_ML,
    )
except Exception:
    team_name_map = lambda: {}
    pregame_history = lambda: {}
    movement_for_event = lambda *a, **k: None
    choose_event = lambda c, g: (c[0] if c else None)
    MOVEMENT_TOTAL = 1.4
    MOVEMENT_SPREAD = 0.9
    MOVEMENT_ML = 0.03


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def movement_weight(move, threshold):
    """Bounded weight: 0 below threshold, 15% at 1x, 30% cap at 2x+."""
    if move is None:
        return 0.0
    try:
        m = float(move)
    except (TypeError, ValueError):
        return 0.0
    if abs(m) < threshold:
        return 0.0
    return min(0.30, 0.15 * abs(m) / threshold)


# Aliases for tests.
weight_for_move = movement_weight
compute_weight = movement_weight
move_weight = movement_weight


def blend_toward(model_value, market_value, weight):
    if model_value is None or market_value is None:
        return model_value
    return (1.0 - weight) * float(model_value) + weight * float(market_value)


def derive_spread_from_prob(prob, sigma=SIGMA_MARGIN):
    try:
        p = min(max(float(prob), 0.01), 0.99)
        return NormalDist().inv_cdf(p) * float(sigma)
    except Exception:
        return None


def apply_market_adjustment(predictions, now):
    """Mutate ``predictions`` (list of game dicts) in place and return it."""
    if not isinstance(predictions, list):
        return predictions

    odds = load_json("market/odds_latest.json") or []
    names = team_name_map()
    movements = pregame_history()

    odds_index = {}
    for event in odds:
        key = (event.get("home_team"), event.get("away_team"))
        odds_index.setdefault(key, []).append(event)

    odds_meta = load_json("market/odds_meta.json") or {}
    source_ts = odds_meta.get("fetched_at")

    for game in predictions:
        if not isinstance(game, dict):
            continue
        home_abbr = game.get("home_team")
        away_abbr = game.get("away_team")
        home_full = names.get(home_abbr, home_abbr)
        away_full = names.get(away_abbr, away_abbr)

        candidates = list(odds_index.get((home_full, away_full), [])) + list(
            odds_index.get((away_full, home_full), [])
        )
        event = choose_event(candidates, game) if candidates else None

        # Not started / pre-kickoff gate.
        pre_kickoff = True
        if event is not None:
            try:
                kickoff = datetime.fromisoformat(
                    str(event.get("commence_time", "")).replace("Z", "+00:00")
                )
                if kickoff <= now:
                    pre_kickoff = False
                    event = None
            except Exception:
                pass

        # Preserve pure model values.
        pure_prob = game.get("home_win_prob")
        pure_total = game.get("total")
        pure_spread = game.get("spread")
        game["model_home_win_prob"] = game.get("model_home_win_prob", pure_prob)
        game["model_total"] = game.get("model_total", pure_total)
        game["model_spread"] = game.get("model_spread", pure_spread)

        ml_move = None
        total_move = None
        spread_move = None
        market_prob = None
        market_total = None
        market_spread = None

        if event is not None:
            reversed_event = (
                event.get("home_team") == away_full
                and event.get("away_team") == home_full
            )
            market_prob = event.get("market_home_win_prob")
            if reversed_event and market_prob is not None:
                market_prob = 1.0 - float(market_prob)
            if reversed_event:
                market_spread = event.get("away_spread")
            else:
                market_spread = event.get("home_spread")
            market_total = event.get("total")

            movement = movement_for_event(
                movements, home_full, away_full, event.get("commence_time")
            )
            if movement:
                ml_move = movement.get("ml_move")
                total_move = movement.get("total_move")
                spread_move = movement.get("spread_move")

        w_ml = movement_weight(ml_move, MOVEMENT_ML)
        w_total = movement_weight(total_move, MOVEMENT_TOTAL)
        w_spread = movement_weight(spread_move, MOVEMENT_SPREAD)
        w_prob = max(w_ml, w_spread)

        applied = False
        new_prob = pure_prob
        new_total = pure_total
        new_spread = pure_spread

        if pre_kickoff and event is not None:
            if market_prob is not None and w_prob > 0:
                new_prob = blend_toward(pure_prob, market_prob, w_prob)
                applied = True
            if market_total is not None and w_total > 0:
                new_total = blend_toward(pure_total, market_total, w_total)
                applied = True
            if applied and new_prob is not None and pure_prob is not None and new_prob != pure_prob:
                derived = derive_spread_from_prob(new_prob, SIGMA_MARGIN)
                if derived is not None:
                    new_spread = derived

        game["market_home_prob"] = market_prob
        game["market_total"] = market_total
        game["market_spread"] = market_spread
        game["ml_move"] = ml_move
        game["total_move"] = total_move
        game["spread_move"] = spread_move

        if applied:
            if new_prob is not None:
                game["home_win_prob"] = round(float(new_prob), 3)
            if new_total is not None:
                game["total"] = round(float(new_total), 1)
            if new_spread is not None:
                game["spread"] = round(float(new_spread), 1)

        game["market_adjustment"] = {
            "applied": applied,
            "weights": {
                "ml": round(w_ml, 4),
                "total": round(w_total, 4),
                "spread": round(w_spread, 4),
                "prob": round(w_prob, 4),
            },
            "w_ml": round(w_ml, 4),
            "w_total": round(w_total, 4),
            "w_spread": round(w_spread, 4),
            "w_prob": round(w_prob, 4),
            "market_home_prob": market_prob,
            "market_total": market_total,
            "market_spread": market_spread,
            "ml_move": ml_move,
            "total_move": total_move,
            "spread_move": spread_move,
            "source_timestamp": source_ts,
        }

    return predictions


if __name__ == "__main__":
    # CLI convenience: adjust the upcoming predictions file in place.
    path = "predictions/upcoming_games.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            preds = json.load(f)
        apply_market_adjustment(preds, datetime.now(timezone.utc))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2)
        print(f"Adjusted {len(preds)} games")
