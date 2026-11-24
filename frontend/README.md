# NFL ML Predictions – Frontend

Static frontend for the NFL predictive‑modeling pipeline. It renders:

- Match predictions (winner probability, spread, total)
- Player projections (targets, receptions, yards, TDs)
- Playoff / Super‑Bowl probabilities
- Model‑vs‑market comparison

Prediction and report JSON are served **same‑origin** under `/data/...` by a
Cloudflare Pages Function (`functions/data/[[path]].js`). The browser never
sees an upstream data URL.

## Local preview

```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

Note: the `/data` proxy only exists when deployed to Cloudflare Pages. For
local preview, seed `frontend/data/` with the JSON files or run a small proxy.

## Deploy to Cloudflare Pages (free)

1. Go to the Cloudflare dashboard → **Pages** → **Create a project**.
2. Connect the project repository (or use direct upload).
3. Configure:
   - **Framework preset:** None
   - **Build command:** *(leave empty)*
   - **Build output directory:** `frontend`
4. Deploy. Cloudflare provides a free `*.pages.dev` URL.
5. Optionally add a subdomain of a domain already on Cloudflare for $0.

Alternatively, deploy from the command line:

```bash
npx wrangler pages deploy frontend --project-name nfl-ml-data
```

## Data contract

The frontend expects:

| Path | Contents |
|------|----------|
| `/data/predictions/upcoming_games.json` | Upcoming game predictions |
| `/data/predictions/games_week.json` | Latest completed week |
| `/data/predictions/players_week.json` | Player projections |
| `/data/predictions/playoffs.json` | Playoff/Super Bowl odds |
| `/data/reports/market_comparison.json` | Model‑vs‑market comparison |

## Notes

- Picks are stored in `localStorage` (no auth).
- A later enhancement can add a D1‑backed API for multi‑user picks and leaderboards.
- No build step is required; this is plain HTML/CSS/JS.
