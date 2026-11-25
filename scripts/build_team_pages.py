#!/usr/bin/env python3
"""Generate static team pages for SEO.

Reads ``predictions/playoffs.json`` and ``predictions/upcoming_games.json`` and
writes ``frontend/teams/index.html`` plus one page per team.
"""

from __future__ import annotations

import html
import json
import os


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="../style.css" />
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle"><a href="/">← Back to NFL ML Predictions</a></p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""


def main() -> None:
    playoffs = load_json("predictions/playoffs.json", [])
    games = load_json("predictions/upcoming_games.json", [])
    if not playoffs:
        print("No playoff predictions found; skipping team pages.")
        return

    out_dir = "frontend/teams"
    os.makedirs(out_dir, exist_ok=True)

    # Index page
    rows = []
    for team in sorted(playoffs, key=lambda t: t.get("team", "")):
        abbr = team.get("team", "")
        rows.append(
            f"<tr><td><a href=\"{html.escape(abbr)}.html\">{html.escape(abbr)}</a></td>"
            f"<td>{html.escape(str(team.get('conference') or ''))}</td>"
            f"<td>{fmt_pct(team.get('playoff_prob'))}</td>"
            f"<td>{fmt_pct(team.get('super_bowl_prob'))}</td></tr>"
        )
    index_body = f"""
    <section>
      <h2>Teams</h2>
      <table class="team-table">
        <thead><tr><th>Team</th><th>Conference</th><th>Playoff</th><th>Super Bowl</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("NFL Teams — Playoff Odds", index_body))

    # Per-team pages
    for team in playoffs:
        abbr = team.get("team", "")
        if not abbr:
            continue
        team_games = [
            g for g in games if g.get("home_team") == abbr or g.get("away_team") == abbr
        ]
        game_rows = []
        for g in sorted(team_games, key=lambda x: (x.get("week", 0), x.get("gameday") or ""))[:10]:
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            game_rows.append(
                f"<tr><td>Week {html.escape(str(g.get('week')))}</td>"
                f"<td>{html.escape(away)} @ {html.escape(home)}</td>"
                f"<td>{html.escape(str(g.get('gameday') or ''))} {html.escape(str(g.get('gametime') or ''))}</td>"
                f"<td>{html.escape(str(g.get('stadium') or ''))}</td></tr>"
            )
        body = f"""
        <section>
          <h2>{html.escape(abbr)}</h2>
          <p>Conference: {html.escape(str(team.get('conference') or '—'))}</p>
          <p>Playoff probability: {fmt_pct(team.get('playoff_prob'))}</p>
          <p>Super Bowl probability: {fmt_pct(team.get('super_bowl_prob'))}</p>
        </section>
        <section>
          <h2>Upcoming games</h2>
          <table class="team-table">
            <thead><tr><th>Week</th><th>Matchup</th><th>Start</th><th>Venue</th></tr></thead>
            <tbody>{''.join(game_rows) or '<tr><td colspan="4">No upcoming games</td></tr>'}</tbody>
          </table>
        </section>
        """
        with open(os.path.join(out_dir, f"{abbr}.html"), "w", encoding="utf-8") as f:
            f.write(page(f"{abbr} — NFL Predictions", body))

    print(f"Wrote {len(playoffs)} team pages to {out_dir}/")


if __name__ == "__main__":
    main()
