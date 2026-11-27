#!/usr/bin/env python3
"""Apply bounded weather multipliers to player projections from component models."""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PLAYERS = "predictions/players_week.json"
UPCOMING = "predictions/upcoming_games.json"
STATE = "predictions/.players_weather_state.json"
META = "predictions/meta.json"


def load(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


NUMERIC_FIELDS = [
    "projected_targets", "projected_receptions", "projected_yards", "projected_tds",
    "projected_fantasy_points", "floor_fantasy_points", "ceiling_fantasy_points",
    "projected_pass_yards", "projected_pass_tds", "projected_interceptions",
    "projected_rush_yards", "projected_rush_tds",
    "projected_fg", "projected_fg_att", "projected_xp",
]


def sig():
    h = hashlib.sha256()
    for path in (PLAYERS, META, UPCOMING):
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def fantasy_ratio(pos, p, pr, rr, fr):
    pos = (pos or "").upper()
    if pos == "QB":
        py = float(p.get("projected_pass_yards") or 0.0)
        pt = float(p.get("projected_pass_tds") or 0.0)
        ry = float(p.get("projected_rush_yards") or 0.0)
        rt = float(p.get("projected_rush_tds") or 0.0)
        base = 0.04 * py + 4.0 * pt + 0.1 * ry + 6.0 * rt
        adj = 0.04 * py * pr + 4.0 * pt * pr + 0.1 * ry * rr + 6.0 * rt * rr
        return clamp(adj / base, 0.85, 1.10) if base > 0 else 1.0
    if pos in ("WR", "TE"):
        return clamp(pr, 0.85, 1.10)
    if pos == "RB":
        ry = float(p.get("projected_rush_yards") or p.get("projected_yards") or 0.0)
        rt = float(p.get("projected_rush_tds") or p.get("projected_tds") or 0.0)
        rec = float(p.get("projected_receptions") or 0.0)
        base = 0.1 * ry + 6.0 * rt + 1.0 * rec
        adj = rr * (0.1 * ry + 6.0 * rt) + pr * (1.0 * rec)
        return clamp(adj / base, 0.85, 1.10) if base > 0 else 1.0
    if pos == "K":
        fg = float(p.get("projected_fg") or 0.0)
        xp = float(p.get("projected_xp") or 0.0)
        base = 3.0 * fg + 1.0 * xp
        adj = fr * (3.0 * fg) + 1.0 * xp
        return clamp(adj / base, 0.85, 1.10) if base > 0 else 1.0
    return 1.0


def main():
    players = load(PLAYERS, [])
    upcoming = load(UPCOMING, [])
    if not isinstance(players, list) or not isinstance(upcoming, list):
        print("Missing data; skipping.")
        return

    # Idempotency: skip if source unchanged.
    current_sig = sig()
    state = load(STATE, {})
    if isinstance(state, dict) and state.get("signature") == current_sig:
        print("Player weather adjustment already applied for this input; skipping.")
        return

    # team -> ratios (first non-neutral game wins)
    team_ratios = {}
    for game in upcoming:
        sp = game.get("stat_projection") or {}
        if "pass_ratio" not in sp:
            continue
        pr = float(sp.get("pass_ratio", 1.0))
        rr = float(sp.get("rush_ratio", 1.0))
        fr = float(sp.get("fg_ratio", 1.0))
        for team in (game.get("home_team"), game.get("away_team")):
            if team and team not in team_ratios:
                team_ratios[team] = (pr, rr, fr)

    applied_count = 0
    for p in players:
        if not isinstance(p, dict):
            continue
        # Restore base if previously adjusted.
        wa = p.get("weather_adjustment")
        if isinstance(wa, dict) and isinstance(wa.get("base"), dict):
            for k, v in wa["base"].items():
                p[k] = v

        team = p.get("team")
        pr, rr, fr = team_ratios.get(team, (1.0, 1.0, 1.0))
        material = any(abs(r - 1.0) > 0.005 for r in (pr, rr, fr))
        pos = (p.get("position") or "").upper()

        # snapshot base before any mutation
        base_snapshot = {k: p.get(k) for k in NUMERIC_FIELDS}

        if material:
            applied_count += 1
            # scale stat fields
            if pos == "QB":
                if p.get("projected_pass_yards") is not None:
                    p["projected_pass_yards"] = round(float(p["projected_pass_yards"]) * pr, 1)
                if p.get("projected_pass_tds") is not None:
                    p["projected_pass_tds"] = round(float(p["projected_pass_tds"]) * pr, 2)
                if p.get("projected_rush_yards") is not None:
                    p["projected_rush_yards"] = round(float(p["projected_rush_yards"]) * rr, 1)
                if p.get("projected_rush_tds") is not None:
                    p["projected_rush_tds"] = round(float(p["projected_rush_tds"]) * rr, 2)
            elif pos in ("WR", "TE"):
                if p.get("projected_targets") is not None:
                    p["projected_targets"] = round(float(p["projected_targets"]) * pr, 1)
                if p.get("projected_receptions") is not None:
                    p["projected_receptions"] = round(float(p["projected_receptions"]) * pr, 1)
                if p.get("projected_yards") is not None:
                    p["projected_yards"] = round(float(p["projected_yards"]) * pr, 1)
                if p.get("projected_tds") is not None:
                    p["projected_tds"] = round(float(p["projected_tds"]) * pr, 2)
            elif pos == "RB":
                if p.get("projected_rush_yards") is not None:
                    p["projected_rush_yards"] = round(float(p["projected_rush_yards"]) * rr, 1)
                if p.get("projected_rush_tds") is not None:
                    p["projected_rush_tds"] = round(float(p["projected_rush_tds"]) * rr, 2)
                if p.get("projected_receptions") is not None:
                    p["projected_receptions"] = round(float(p["projected_receptions"]) * pr, 1)
                if p.get("projected_targets") is not None:
                    p["projected_targets"] = round(float(p["projected_targets"]) * pr, 1)
                if p.get("projected_yards") is not None:
                    p["projected_yards"] = round(float(p["projected_yards"]) * rr, 1)
                if p.get("projected_tds") is not None:
                    p["projected_tds"] = round(float(p["projected_tds"]) * rr, 2)
            if pos == "K":
                if p.get("projected_fg") is not None:
                    p["projected_fg"] = round(float(p["projected_fg"]) * fr, 2)
                if p.get("projected_fg_att") is not None:
                    p["projected_fg_att"] = round(float(p["projected_fg_att"]) * fr, 2)

            fratio = fantasy_ratio(pos, p, pr, rr, fr)
            # apply fantasy ratio to derived totals (use original base snapshot values)
            for field in ("projected_fantasy_points", "floor_fantasy_points", "ceiling_fantasy_points"):
                orig = base_snapshot.get(field)
                if orig is not None:
                    p[field] = round(float(orig) * fratio, 1 if field != "projected_fantasy_points" else 1)
            if p.get("projected_fantasy_points") is not None and base_snapshot.get("projected_fantasy_points") is not None:
                p["projected_fantasy_points"] = round(float(base_snapshot["projected_fantasy_points"]) * fratio, 1)

        p["weather_adjustment"] = {
            "pass_ratio": round(pr, 4),
            "rush_ratio": round(rr, 4),
            "fg_ratio": round(fr, 4),
            "fantasy_ratio": round(fantasy_ratio(pos, p if not material else p, pr, rr, fr), 4) if False else round(fantasy_ratio(pos, {**p, **{k: base_snapshot.get(k) for k in NUMERIC_FIELDS}}, pr, rr, fr), 4),
            "applied": material,
            "base": base_snapshot if material else None,
        }
        # Fix fantasy_ratio computation using base snapshot for material case
        if material:
            base_p = {**p}
            for k in NUMERIC_FIELDS:
                base_p[k] = base_snapshot.get(k)
            p["weather_adjustment"]["fantasy_ratio"] = round(fantasy_ratio(pos, base_p, pr, rr, fr), 4)

    dump(PLAYERS, players)
    dump(STATE, {"signature": current_sig})
    applied_count = sum(1 for p in players if (p.get("weather_adjustment") or {}).get("applied"))
    print(f"Adjusted {applied_count}/{len(players)} players")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Player weather adjustment error: {exc}")
