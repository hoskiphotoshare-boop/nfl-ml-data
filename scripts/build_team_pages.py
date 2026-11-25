#!/usr/bin/env python3
"""Generate static team pages for SEO and sharing.

Reads ``predictions/playoffs.json`` and ``predictions/upcoming_games.json`` and
writes ``frontend/teams/index.html`` plus one page per team.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime

TEAMS = {
    "ARI": ("Cardinals", "#97233F"),
    "ATL": ("Falcons", "#A71930"),
    "BAL": ("Ravens", "#241773"),
    "BUF": ("Bills", "#00338D"),
    "CAR": ("Panthers", "#0085CA"),
    "CHI": ("Bears", "#0B162A"),
    "CIN": ("Bengals", "#FB4F14"),
    "CLE": ("Browns", "#311D00"),
    "DAL": ("Cowboys", "#041E42"),
    "DEN": ("Broncos", "#FB4F14"),
    "DET": ("Lions", "#0076B6"),
    "GB": ("Packers", "#203731"),
    "HOU": ("Texans", "#03202F"),
    "IND": ("Colts", "#002C5F"),
    "JAX": ("Jaguars", "#006778"),
    "KC": ("Chiefs", "#E31837"),
    "LA": ("Rams", "#003594"),
    "LAC": ("Chargers", "#0080C6"),
    "LV": ("Raiders", "#1F2937"),
    "MIA": ("Dolphins", "#008E97"),
    "MIN": ("Vikings", "#4F2683"),
    "NE": ("Patriots", "#002244"),
    "NO": ("Saints", "#D3BC8D"),
    "NYG": ("Giants", "#0B2265"),
    "NYJ": ("Jets", "#125740"),
    "PHI": ("Eagles", "#004C54"),
    "PIT": ("Steelers", "#FFB612"),
    "SEA": ("Seahawks", "#002244"),
    "SF": ("49ers", "#AA0000"),
    "TB": ("Buccaneers", "#D50A0A"),
    "TEN": ("Titans", "#0C2340"),
    "WAS": ("Commanders", "#5A1414"),
}


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


def american(prob) -> str:
    try:
        p = float(prob)
    except Exception:
        return "—"
    if p <= 0 or p >= 1:
        return "—"
    raw = (-100 * p / (1 - p)) if p >= 0.5 else (100 * (1 - p) / p)
    return f"{raw:+.0f}"


def luminance(hex_color: str) -> float:
    clean = hex_color.lstrip("#")
    r, g, b = (int(clean[i : i + 2], 16) / 255 for i in (0, 2, 4))
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def badge(abbr: str, small: bool = False) -> str:
    color = TEAMS.get(abbr, ("", "#334155"))[1]
    text = "#0a1020" if luminance(color) > 0.55 else "#ffffff"
    size = " sm" if small else ""
    return (
        f'<span class="team-badge{size}" style="background:{color};color:{text}">'
        f"{html.escape(abbr)}</span>"
    )


def full_name(abbr: str) -> str:
    return TEAMS.get(abbr, (abbr, ""))[0]


def kickoff(gameday, gametime) -> str:
    label = html.escape(str(gameday or "TBD"))
    try:
        label = datetime.strptime(str(gameday), "%Y-%m-%d").strftime("%a, %b %d").replace(" 0", " ")
    except Exception:
        pass
    if gametime:
        try:
            hour, minute = (int(part) for part in str(gametime).split(":")[:2])
            suffix = "PM" if hour >= 12 else "AM"
            return f"{label} · {hour % 12 or 12}:{minute:02d} {suffix} ET"
        except Exception:
            pass
    return label


def page(title: str, description: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="{html.escape(description)}" />
  <meta name="theme-color" content="#070b14" />
  <title>{html.escape(title)}</title>
  <link rel="icon" href='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🏈</text></svg>' />
  <link rel="stylesheet" href="../style.css" />
</head>
<body>
  <header class="appbar">
    <div class="appbar-inner">
      <div class="brand-row">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4.5 19.5 19.5 4.5" /><path d="M7.8 4.9 19.1 16.2" /><path d="M4.9 7.8 16.2 19.1" />
            </svg>
          </span>
          <div class="brand-text">
            <h1 class="brand-title">NFL <span>ML Predictions</span></h1>
            <p class="brand-sub">Model odds · player projections · playoff chances</p>
          </div>
        </div>
        <div class="spacer"></div>
        <div class="topbar-meta">
          <a class="chip" href="/">← All games</a>
        </div>
      </div>
    </div>
  </header>
  <main>
    {body}
  </main>
  <footer>
    <div class="foot-row"><span>Model-implied probabilities · not betting advice</span></div>
    <div class="foot-row"><span>Data: nflverse (CC-BY 4.0)</span></div>
  </footer>
</body>
</html>
"""


def stat(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (
        f'<div class="stat {cls}">'
        f'<div class="k">{html.escape(label)}</div>'
        f'<div class="v">{value}</div>'
        f'<div class="s">{html.escape(sub)}</div>'
        "</div>"
    )


def meter(value, cls: str = "") -> str:
    try:
        pct = max(0.0, min(1.0, float(value))) * 100
        label = f"{pct:.0f}%"
    except Exception:
        pct, label = 0.0, "—"
    return (
        f'<div class="meter {cls}">'
        f'<div class="track"><i style="width:{pct:.1f}%"></i></div>'
        f'<span class="val">{label}</span>'
        "</div>"
    )


def main() -> None:
    playoffs = load_json("predictions/playoffs.json", [])
    games = load_json("predictions/upcoming_games.json", [])
    if not playoffs:
        print("No playoff predictions found; skipping team pages.")
        return

    out_dir = "frontend/teams"
    os.makedirs(out_dir, exist_ok=True)

    ordered = sorted(playoffs, key=lambda t: (t.get("super_bowl_prob") or 0), reverse=True)
    rows = []
    for index, team in enumerate(ordered, start=1):
        abbr = team.get("team", "")
        conf = team.get("conference") or "—"
        rows.append(
            f'<tr><td class="muted-2">{index}</td>'
            f'<td><div class="player">{badge(abbr, True)}<div><div class="nm">'
            f'<a href="{html.escape(abbr)}.html">{html.escape(abbr)}</a></div>'
            f'<div class="tm">{html.escape(full_name(abbr))}</div></div></div></td>'
            f'<td class="hide-sm muted">{html.escape(conf)}</td>'
            f'<td>{meter(team.get("playoff_prob"), "green")}</td>'
            f'<td class="hide-sm">{meter(team.get("conference_prob"))}</td>'
            f'<td>{meter(team.get("super_bowl_prob"), "green")}</td>'
            f'<td class="num hide-sm muted">{american(team.get("super_bowl_prob"))}</td></tr>'
        )

    index_body = f"""
    <div class="panel-head">
      <div>
        <h2>NFL playoff odds by team</h2>
        <p>Every team’s simulated chance of reaching the playoffs, winning its conference and lifting the Super Bowl, from 10,000 Monte Carlo runs.</p>
      </div>
    </div>
    <div class="table-wrap"><div class="table-scroll"><table class="dtable">
      <thead><tr><th>#</th><th>Team</th><th class="hide-sm">Conf</th><th>Playoff</th>
      <th class="hide-sm">Conf. title</th><th>Super Bowl</th><th class="num hide-sm">SB odds</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div></div>
    """
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            page(
                "NFL Playoff Odds by Team — NFL ML Predictions",
                "Simulated playoff and Super Bowl probabilities for all 32 NFL teams.",
                index_body,
            )
        )

    for team in ordered:
        abbr = team.get("team", "")
        if not abbr:
            continue
        team_games = [g for g in games if g.get("home_team") == abbr or g.get("away_team") == abbr]
        game_rows = []
        for g in sorted(team_games, key=lambda x: (x.get("week", 0), x.get("gameday") or "")):
            home, away = g.get("home_team", ""), g.get("away_team", "")
            opponent = away if home == abbr else home
            venue = "vs" if home == abbr else "@"
            model = g.get("home_win_prob")
            win_prob = model if home == abbr else (1 - model if model is not None else None)
            game_rows.append(
                f'<tr><td class="muted-2">Wk {html.escape(str(g.get("week")))}</td>'
                f'<td><div class="player">{badge(opponent, True)}'
                f'<div><div class="nm">{venue} {html.escape(full_name(opponent))}</div>'
                f'<div class="tm">{html.escape(opponent)} · {html.escape(str(g.get("stadium") or ""))}</div></div></div></td>'
                f'<td class="hide-sm">{kickoff(g.get("gameday"), g.get("gametime"))}</td>'
                f'<td class="num">{fmt_pct(win_prob)}</td>'
                f'<td class="num hide-sm muted">{american(win_prob)}</td></tr>'
            )

        body = f"""
        <div class="panel-head">
          <div>
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.35rem">
              {badge(abbr)}
              <h2>{html.escape(abbr)} — {html.escape(full_name(abbr))}</h2>
            </div>
            <p>{html.escape(team.get("conference") or "")} · simulated playoff, conference and Super Bowl chances plus the remaining schedule.</p>
          </div>
        </div>
        <div class="stat-row">
          {stat("Playoff chance", fmt_pct(team.get("playoff_prob")), "make the postseason", "good")}
          {stat("Conference title", fmt_pct(team.get("conference_prob")), "win the conference")}
          {stat("Super Bowl", fmt_pct(team.get("super_bowl_prob")), "win the title")}
          {stat("SB odds", american(team.get("super_bowl_prob")), "model-implied moneyline")}
        </div>
        <div class="section-title">Remaining schedule</div>
        <div class="table-wrap"><div class="table-scroll"><table class="dtable">
          <thead><tr><th>Week</th><th>Matchup</th><th class="hide-sm">Kickoff</th>
          <th class="num">Win prob</th><th class="num hide-sm">Odds</th></tr></thead>
          <tbody>{''.join(game_rows) or '<tr><td colspan="5" class="muted">No remaining games</td></tr>'}</tbody>
        </table></div></div>
        """
        with open(os.path.join(out_dir, f"{abbr}.html"), "w", encoding="utf-8") as f:
            f.write(
                page(
                    f"{abbr} {full_name(abbr)} Playoff Odds & Remaining Schedule — NFL ML Predictions",
                    f"{full_name(abbr)} playoff probability, Super Bowl odds and remaining NFL schedule with model win probabilities.",
                    body,
                )
            )

    print(f"Wrote {len(ordered)} team pages to {out_dir}/")


if __name__ == "__main__":
    main()
