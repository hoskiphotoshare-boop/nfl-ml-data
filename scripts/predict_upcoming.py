#!/usr/bin/env python3
"""Predict upcoming NFL games using the trained match model.

Builds a feature vector for each unplayed game from the most recent
team-level feature row, then writes ``predictions/upcoming_games.json``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__) + "/../")

import joblib
import polars as pl


ET = ZoneInfo("America/New_York")


def kickoff_utc(gameday, gametime) -> datetime | None:
    """Convert a schedule gameday/gametime (ET) to an aware UTC timestamp."""
    try:
        day = datetime.strptime(str(gameday)[:10], "%Y-%m-%d")
    except Exception:
        return None
    hour = minute = 0
    if gametime:
        try:
            parts = str(gametime).strip().split(":")
            hour, minute = int(parts[0]), int(parts[1])
        except Exception:
            hour = minute = 0
    return day.replace(hour=hour, minute=minute, tzinfo=ET).astimezone(timezone.utc)


WEATHER_FIELDS = (
    "temp",
    "wind",
    "gust",
    "precip",
    "precip_mm",
    "snow",
    "weather_code",
    "humidity",
    "weather_fetched_at",
)


def merge_weather(row: dict, previous: dict | None) -> dict:
    """Keep the last fetched forecast when a prediction row is rebuilt."""
    weather = {
        field: previous.get(field)
        for field in WEATHER_FIELDS
        if previous and previous.get(field) is not None
    }
    if row.get("temp") is not None:
        weather["temp"] = row["temp"]
    if row.get("wind") is not None:
        weather["wind"] = row["wind"]
    return weather


def load_schedules() -> pl.DataFrame | None:
    for path in ["data/schedules.parquet", "data/schedules.csv"]:
        if os.path.exists(path):
            return pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
    return None


def load_platt():
    path = "models/platt.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if data.get("enabled") else None
    except Exception:
        return None


def apply_platt(prob: float, platt: dict | None) -> float:
    if not platt:
        return prob
    p = min(max(prob, 1e-6), 1 - 1e-6)
    value = platt["a"] * math.log(p / (1 - p)) + platt["b"]
    return 1.0 / (1.0 + math.exp(-value))


def load_score_models():
    path = "models/score_models.joblib"
    if not os.path.exists(path):
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def load_match_features() -> pl.DataFrame | None:
    path = "features/match_features.parquet"
    if not os.path.exists(path):
        return None
    return pl.read_parquet(path)


def current_records(schedules: pl.DataFrame, season: int) -> dict[str, dict]:
    """Current wins, games and point differential per team."""
    played = schedules.filter(
        (pl.col("season") == season)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    home = played.select(
        pl.col("home_team").alias("team"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("win"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
    )
    away = played.select(
        pl.col("away_team").alias("team"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Int8).alias("win"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
    )
    long = pl.concat([home, away])
    agg = long.group_by("team").agg(
        pl.col("win").sum().alias("wins"),
        pl.len().alias("games"),
        pl.col("pf").sum().alias("pf"),
        pl.col("pa").sum().alias("pa"),
    )
    return {
        row["team"]: {
            "win_pct": round(row["wins"] / max(row["games"], 1), 4),
            "diff_pg": round((row["pf"] - row["pa"]) / max(row["games"], 1), 3),
            "prior_games": int(row["games"]),
        }
        for row in agg.iter_rows(named=True)
    }


def current_injury_context(season: int) -> dict[str, dict]:
    try:
        from add_match_context import injury_table

        table = injury_table()
        if table is None:
            return {}
        latest = (
            table.filter(pl.col("season") == season)
            .sort("week")
            .group_by("team", maintain_order=True)
            .agg(pl.all().last())
        )
        keys = [
            "inj_impact",
            "inj_qb",
            "inj_ol",
            "inj_skill",
            "inj_def",
        ]
        return {
            row["team"]: {
                **{key: float(row.get(key) or 0.0) for key in keys},
                "inj_missing": 0,
            }
            for row in latest.iter_rows(named=True)
        }
    except Exception:
        return {}


def current_roster_context(season: int) -> dict[str, dict]:
    try:
        from add_match_context import roster_table

        table = roster_table()
        if table is None:
            return {}
        latest = (
            table.filter(pl.col("season") == season)
            .sort("week")
            .group_by("team", maintain_order=True)
            .agg(pl.all().last())
        )
        return {
            row["team"]: {
                "roster_qb": int(row.get("roster_qb") or 0),
                "roster_ol": int(row.get("roster_ol") or 0),
                "roster_db": int(row.get("roster_db") or 0),
                "roster_missing": 0,
            }
            for row in latest.iter_rows(named=True)
        }
    except Exception:
        return {}


def current_qb_context(season: int) -> dict[str, dict]:
    """Trailing starter quality and QB-change flags for each team."""
    path = "features/player_features.parquet"
    if not os.path.exists(path):
        return {}
    players = pl.read_parquet(path)
    qbs = players.filter(
        (pl.col("position") == "QB")
        & (pl.col("season") == season)
        & (pl.col("attempts").is_not_null())
        & (pl.col("attempts") > 0)
    )
    if qbs.height == 0:
        return {}

    primary = (
        qbs.sort(
            ["team", "week", "attempts"],
            descending=[False, False, True],
        )
        .group_by(["team", "week"], maintain_order=True)
        .agg(
            pl.col("player_id").first().alias("qb_id"),
            pl.col("player_name").first().alias("qb_name"),
            pl.col("attempts").first().alias("attempts"),
            pl.col("passing_epa").first().alias("passing_epa"),
        )
    )
    latest = (
        primary.sort(["team", "week"])
        .group_by("team", maintain_order=True)
        .agg(pl.all().last())
    )

    # Prefer the current depth-chart starter when available.
    starters = {}
    dc_path = "data/depth_charts_latest.parquet"
    if os.path.exists(dc_path):
        dc = pl.read_parquet(dc_path)
        latest_dt = dc["dt"].max()
        dc = dc.filter((pl.col("dt") == latest_dt) & (pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1))
        for row in dc.iter_rows(named=True):
            starters[row["team"]] = {
                "id": row.get("gsis_id"),
                "name": row.get("player_name"),
            }

    context = {}
    for row in latest.iter_rows(named=True):
        team = row["team"]
        starter = starters.get(team)
        qb_id = row["qb_id"]
        qb_name = row["qb_name"]
        if starter and (starter.get("id") or starter.get("name")):
            # Match the depth-chart starter in the player table.
            candidates = qbs.filter(
                (pl.col("team") == team)
                & (
                    (pl.col("player_id") == starter.get("id"))
                    if starter.get("id")
                    else (pl.col("player_name") == starter.get("name"))
                )
            )
            if candidates.height:
                qb_id = candidates["player_id"][-1]
                qb_name = candidates["player_name"][-1]

        hist = primary.filter((pl.col("team") == team) & (pl.col("qb_id") == qb_id)).sort("week")
        if hist.height == 0:
            context[team] = {
                "qb_prior_epa": 0.0,
                "qb_starts": 0,
                "qb_changed": 1,
                "qb_missing": 1,
            }
            continue
        epa = hist["passing_epa"].to_list()[-3:]
        att = hist["attempts"].to_list()[-3:]
        values = [e / a for e, a in zip(epa, att) if a and e is not None]
        previous = primary.filter(
            (pl.col("team") == team) & (pl.col("week") < row["week"])
        ).sort("week")
        changed = 0
        if previous.height:
            changed = int(previous["qb_id"][-1] != qb_id)

        context[team] = {
            "qb_prior_epa": round(sum(values) / len(values), 4) if values else 0.0,
            "qb_starts": int(hist.height - 1),
            "qb_changed": changed,
            "qb_missing": 0,
        }
    return context


def team_latest_row(df: pl.DataFrame, team: str) -> dict:
    rows = df.filter((pl.col("home_team") == team) | (pl.col("away_team") == team)).sort(
        ["season", "week"]
    )
    if rows.height == 0:
        return {}
    return rows.tail(1).to_dicts()[0]


def team_stats_from_row(row: dict, team: str) -> dict:
    if not row:
        return {}
    prefix = "home_" if row.get("home_team") == team else "away_"
    stats = {}
    for key, value in row.items():
        if key.startswith(prefix):
            stats[key[len(prefix):]] = value
    return stats


def season_records(schedules: pl.DataFrame, season: int) -> dict[str, dict]:
    """W-L-T and last-five form for every team in the given season."""
    played = schedules.filter(
        (pl.col("season") == season)
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    ).sort(["week", "gameday"] if "gameday" in schedules.columns else ["week"])

    records: dict[str, dict] = {}

    def bucket(team: str) -> dict:
        return records.setdefault(
            team, {"w": 0, "l": 0, "t": 0, "form": []}
        )

    for row in played.iter_rows(named=True):
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        hb, ab = bucket(home), bucket(away)
        if hs > as_:
            hb["w"] += 1
            ab["l"] += 1
            hb["form"].append("W")
            ab["form"].append("L")
        elif as_ > hs:
            ab["w"] += 1
            hb["l"] += 1
            ab["form"].append("W")
            hb["form"].append("L")
        else:
            hb["t"] += 1
            ab["t"] += 1
            hb["form"].append("T")
            ab["form"].append("T")

    out = {}
    for team, rec in records.items():
        suffix = f"-{rec['t']}" if rec["t"] else ""
        out[team] = {
            "record": f"{rec['w']}-{rec['l']}{suffix}",
            "form": "".join(rec["form"][-5:]),
        }
    return out


def build_feature_vector(features: list[str], home_stats: dict, away_stats: dict, week: int, season: int) -> list[float]:
    values = []
    for feat in features:
        if feat == "week":
            values.append(float(week))
        elif feat == "season":
            values.append(float(season))
        elif feat == "div_game":
            values.append(float(home_stats.get("div_game") or 0.0))
        elif feat == "rest_diff":
            values.append(
                float(home_stats.get("rest_days") or 0) - float(away_stats.get("rest_days") or 0)
            )
        elif feat.startswith("home_"):
            values.append(float(home_stats.get(feat[5:]) or 0.0))
        elif feat.startswith("away_"):
            values.append(float(away_stats.get(feat[5:]) or 0.0))
        elif feat.endswith("_diff"):
            base = feat[: -len("_diff")]
            values.append(
                float(home_stats.get(base) or 0.0) - float(away_stats.get(base) or 0.0)
            )
        else:
            values.append(0.0)
    return values


def main() -> None:
    schedules = load_schedules()
    features_df = load_match_features()
    if schedules is None or features_df is None:
        print("Missing schedules or features; skipping upcoming predictions.")
        return
    if not os.path.exists("models/match_model.joblib"):
        print("No trained model found; skipping upcoming predictions.")
        return

    bundle = joblib.load("models/match_model.joblib")
    model = bundle["model"]
    feature_names = bundle["features"]
    score_bundle = load_score_models()
    platt = load_platt()

    current_season = int(schedules["season"].max())
    upcoming = schedules.filter(
        (pl.col("season") == current_season)
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    ).sort(["week", "gameday"] if "gameday" in schedules.columns else ["week"])

    if upcoming.height == 0:
        print("No upcoming games found.")
        return

    stats_cache: dict[str, dict] = {}
    records = season_records(schedules, current_season)
    team_records = current_records(schedules, current_season)
    qb_context = current_qb_context(current_season)
    injury_context = current_injury_context(current_season)
    roster_context = current_roster_context(current_season)

    # Actual rest days for the upcoming game, using the previous scheduled game
    # (including unplayed weeks) so future weeks do not all look like long rests.
    team_dates: dict[str, list[datetime]] = {}
    season_games = schedules.filter(pl.col("season") == current_season)
    for scheduled in season_games.iter_rows(named=True):
        day = scheduled.get("gameday")
        if not day:
            continue
        try:
            game_day = datetime.fromisoformat(str(day)[:10])
        except ValueError:
            continue
        for team in (scheduled["home_team"], scheduled["away_team"]):
            team_dates.setdefault(team, []).append(game_day)
    for team in team_dates:
        team_dates[team] = sorted(set(team_dates[team]))

    def rest_days(team: str, gameday) -> int:
        if not gameday:
            return 7
        try:
            game_day = datetime.fromisoformat(str(gameday)[:10])
        except ValueError:
            return 7
        prior = [day for day in team_dates.get(team, []) if day < game_day]
        if not prior:
            return 7
        return max(0, min(21, (game_day - prior[-1]).days))

    now = datetime.now(timezone.utc)
    previous_by_id: dict[str, dict] = {}
    previous_vectors: dict[str, dict] = {}
    if os.path.exists("predictions/upcoming_games.json"):
        try:
            with open("predictions/upcoming_games.json", encoding="utf-8") as f:
                for game in json.load(f):
                    if isinstance(game, dict) and game.get("game_id"):
                        previous_by_id[str(game["game_id"])] = game
        except Exception:
            previous_by_id = {}
    if os.path.exists("models/upcoming_feature_vectors.json"):
        try:
            with open("models/upcoming_feature_vectors.json", encoding="utf-8") as f:
                previous_vectors = (json.load(f) or {}).get("games", {})
        except Exception:
            previous_vectors = {}

    prediction_rows = []
    feature_vectors: dict[str, dict] = {}
    for row in upcoming.iter_rows(named=True):
        game_id = str(row.get("game_id"))
        # A started game must keep its last pre-kickoff prediction. Rebuilding
        # features after kickoff risks using the game itself and makes the card
        # disagree with the archived call.
        kickoff = kickoff_utc(row.get("gameday"), row.get("gametime"))
        if kickoff is not None and kickoff <= now:
            locked = previous_by_id.get(game_id)
            if locked:
                prediction_rows.append(locked)
                if game_id in previous_vectors:
                    feature_vectors[game_id] = previous_vectors[game_id]
            continue

        home = row["home_team"]
        away = row["away_team"]
        week = int(row["week"])

        if home not in stats_cache:
            stats_cache[home] = team_stats_from_row(
                team_latest_row(features_df, home), home
            )
            stats_cache[home].update(team_records.get(home, {}))
            stats_cache[home].update(qb_context.get(home, {}))
            stats_cache[home].update(injury_context.get(home, {}))
        if away not in stats_cache:
            stats_cache[away] = team_stats_from_row(
                team_latest_row(features_df, away), away
            )
            stats_cache[away].update(team_records.get(away, {}))
            stats_cache[away].update(qb_context.get(away, {}))
            stats_cache[away].update(injury_context.get(away, {}))

        # These change every week, so set them for both teams on every row.
        home_rest = rest_days(home, row.get("gameday"))
        away_rest = rest_days(away, row.get("gameday"))
        stats_cache[home].update(
            {
                "div_game": int(row.get("div_game") or 0),
                "rest_days": home_rest,
                "short_week": int(home_rest <= 4),
                **roster_context.get(home, {}),
            }
        )
        stats_cache[away].update(
            {
                "div_game": int(row.get("div_game") or 0),
                "rest_days": away_rest,
                "short_week": int(away_rest <= 4),
                **roster_context.get(away, {}),
            }
        )

        x = build_feature_vector(
            feature_names, stats_cache[home], stats_cache[away], week, current_season
        )
        feature_vectors[str(row.get("game_id"))] = {
            name: float(value) for name, value in zip(feature_names, x)
        }
        raw_proba = float(model.predict_proba([x])[0, 1])
        proba = apply_platt(raw_proba, platt)

        home_avg = float(stats_cache[home].get("avg_points") or 0.0)
        away_avg = float(stats_cache[away].get("avg_points") or 0.0)
        home_allowed = float(stats_cache[home].get("avg_allowed") or 0.0)
        away_allowed = float(stats_cache[away].get("avg_allowed") or 0.0)
        proj_home = (home_avg + away_allowed) / 2.0 if home_avg and away_allowed else None
        proj_away = (away_avg + home_allowed) / 2.0 if away_avg and home_allowed else None
        spread = (proj_home - proj_away) if proj_home is not None and proj_away is not None else None
        total = (proj_home + proj_away) if proj_home is not None and proj_away is not None else None

        # Keep the spread consistent with the win probability: convert the
        # model probability to a margin with the calibrated margin sigma. The
        # separate margin regressor can disagree with the classifier (e.g. a
        # 41% home team being projected as a 4-point favorite).
        sigma_margin = float(score_bundle.get("sigma_margin", 13.5)) if score_bundle else 13.5
        try:
            probit = NormalDist().inv_cdf(min(max(proba, 0.01), 0.99))
            spread = probit * sigma_margin
        except Exception:
            pass

        if score_bundle:
            try:
                sx = build_feature_vector(
                    score_bundle["features"],
                    stats_cache[home],
                    stats_cache[away],
                    week,
                    current_season,
                )
                total = float(score_bundle["total_model"].predict([sx])[0])
            except Exception:
                pass

        weather = merge_weather(row, previous_by_id.get(game_id))
        prediction_rows.append(
            {
                "game_id": row.get("game_id"),
                "season": current_season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(proba, 3),
                "raw_home_win_prob": round(raw_proba, 3),
                "home_record": records.get(home, {}).get("record"),
                "away_record": records.get(away, {}).get("record"),
                "home_form": records.get(home, {}).get("form"),
                "away_form": records.get(away, {}).get("form"),
                "spread": round(spread, 1) if spread is not None else None,
                "total": round(total, 1) if total is not None else None,
                "gameday": str(row.get("gameday")) if row.get("gameday") is not None else None,
                "gametime": str(row.get("gametime")) if row.get("gametime") is not None else None,
                "location": row.get("location"),
                "stadium": row.get("stadium"),
                "roof": row.get("roof"),
                "surface": row.get("surface"),
                "div_game": row.get("div_game"),
                **weather,
            }
        )

    os.makedirs("predictions", exist_ok=True)
    with open("predictions/upcoming_games.json", "w", encoding="utf-8") as f:
        json.dump(prediction_rows, f, indent=2)
    os.makedirs("models", exist_ok=True)
    with open("models/upcoming_feature_vectors.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "season": current_season,
                "features": feature_names,
                "games": feature_vectors,
            },
            f,
            indent=2,
        )
    with open("predictions/meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "games": len(prediction_rows),
                "season": current_season,
                "week": int(prediction_rows[0]["week"]) if prediction_rows else None,
            },
            f,
            indent=2,
        )
    print(f"Wrote predictions/upcoming_games.json ({len(prediction_rows)} games)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Upcoming prediction error: {exc}")
    sys.exit(0)