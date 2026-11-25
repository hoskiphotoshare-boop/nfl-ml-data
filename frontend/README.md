# NFL ML Predictions — Frontend

Static, no-build web app deployed to Cloudflare Pages. It renders:

- **Games** — win probabilities, model spread/total, records/form, kickoff weather,
  and a **Model edges** view ranked by ML/spread/total edge.
- **Players** — per-position projections with matchup, recent form, injury/roster
  status, and **floor–ceiling** ranges.
- **Playoffs** — model playoff/conference/Super Bowl probabilities alongside
  vig-free Super Bowl futures and the edge per team.
- **Team pages** — generated static pages under `/teams/`.
- **Admin** — Google OAuth-gated diagnostics at `/admin.html` (noindex).

All odds are displayed as **decimal** (model fair odds and market prices).
Light theme is the default; a dark theme is available via the header toggle.

## Running locally

The app is plain HTML/CSS/JS; no build step. Data is served same-origin from
`/data/...`, so seed the ignored data folder before previewing:

```bash
mkdir -p frontend/data/predictions frontend/data/reports
cp predictions/*.json frontend/data/predictions/
cp reports/*.json frontend/data/reports/
python3 -m http.server 8080 --directory frontend
# open http://localhost:8080
```

The public site does **not** call GitHub, a proxy, or any upstream data URL.
The deployed `frontend/data/` is populated by the Actions workflows before
upload.

## Deploying

```bash
mkdir -p frontend/data/predictions frontend/data/reports
cp predictions/*.json frontend/data/predictions/
cp reports/*.json frontend/data/reports/
python scripts/build_team_pages.py
npx wrangler pages deploy frontend --project-name nfl-ml-data --commit-dirty=true
```

Cloudflare Pages settings: framework preset **None**, build command empty,
output directory `frontend`.

## Data contract

| Path | Contents |
|---|---|
| `/data/predictions/upcoming_games.json` | Upcoming games: probabilities, spread, total, records/form, kickoff, venue, roof/surface, weather |
| `/data/predictions/players_week.json` | Player projections: stats, matchup, last-six history per stat, 80% floor/ceiling range, injury/roster status |
| `/data/predictions/playoffs.json` | Playoff, conference and Super Bowl probabilities |
| `/data/predictions/meta.json` | Generated timestamp, season, week |
| `/data/reports/market_comparison.json` | Model vs market by game/market, outlier flag |
| `/data/reports/futures_comparison.json` | Model vs Super Bowl futures |
| `/data/reports/backtest.json` | Match-model walk-forward metrics |
| `/data/reports/player_backtest.json` | Player MAE/RMSE and interval coverage |
| `/data/reports/playoff_backtest.json` | Playoff/conference/SB Brier scores |
| `/data/reports/model_explanations.json` | SHAP global/local drivers |
| `/data/reports/model_health.json` | Calibration, outliers, data freshness, budget |
| `/data/reports/line_movement.json` | Opening-to-latest movement by game |
| `/data/reports/feature_ablation.json` | Feature-set comparison |
| `/data/reports/weather_ablation.json` | Historical weather ablation by model (not used in the UI) |

## Behaviour notes

- **Filters** default to the current week; Players defaults to QB. Tabs are
  addressable via the URL hash (`#games`, `#players`, `#playoffs`); week/team
  filters are not persisted across reloads.
- **Model outliers**: games with a 20+ percentage-point disagreement on any
  market are flagged amber. They stay in their sorted position and in the edge
  stats, and the card shows the top SHAP drivers for the specific market
  (winner/spread or total). Spread drivers come from the win-probability model
  because the displayed spread is derived from it. Treat them as
  missing-information flags, not picks.
- **Weather**: kickoff forecasts are shown for outdoor games but are not model
  inputs. A 2024–2026 walk-forward ablation (ERA5 and forecast archive) found
  only a small, non-robust totals gain and flat-or-worse margin, match and
  player models; see `reports/weather_ablation.json`.
- **Player table**: the small numbers under each stat are the last six games,
  oldest to newest (bold = most recent). “Range” under Proj is the 80%
  projection interval (floor 10th percentile to ceiling 90th percentile).
- **Glossary**: each tab has a collapsible legend for its abbreviations,
  symbols and model terms.
- **Mobile tables**: stat and playoff columns stay reachable by swiping the
  table sideways; the player/team column is pinned and a hint appears above
  the table.
- **Picks** (`nfl_ml_picks`) and theme (`nfl_theme`) live in `localStorage`.
- **Admin** is a client-side gate only. The underlying JSON remains public.
- **Team pages** are generated at deploy time by `scripts/build_team_pages.py`.

## Attribution

Data: nflverse (CC-BY 4.0; FTN/participation CC-BY-SA 4.0 where applicable).
Weather forecasts and historical reanalysis: Open-Meteo.com (CC BY 4.0).
Model-implied probabilities are informational, not betting advice.

## Roadmap

- Closing-line/CLV scoreboard before treating large edges as actionable.
- Validate the live playoff blend (match model + SRS) against SRS-only.
- Per-position player interval calibration to hit ~80% coverage.
- Outlier triage for stale lines and late QB/roster news.
- Monitoring alerts for calibration drift, data freshness and Odds API budget.
