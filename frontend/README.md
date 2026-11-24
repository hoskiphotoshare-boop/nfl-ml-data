# NFL ML Predictions – Frontend

Static frontend for the NFL predictive‑modeling pipeline. It fetches prediction
JSON from **jsDelivr** (a free CDN for GitHub) and renders:

- Match predictions (winner probability, spread, total)
- Player projections (targets, receptions, yards, TDs)
- Playoff / Super‑Bowl probabilities

## Local preview

```bash
# from the repo root
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy to Cloudflare Pages (free)

1. Push this repo to GitHub (already done).
2. Go to the Cloudflare dashboard → **Pages** → **Create a project** → **Connect to Git**.
3. Select the `nfl-ml-data` repository.
4. Configure:
   - **Framework preset:** None
   - **Build command:** *(leave empty)*
   - **Build output directory:** `frontend`
5. Click **Save and Deploy**.
6. Cloudflare gives you a free URL such as `https://nfl-ml-data.pages.dev`.
   You can later add a **custom domain** (subdomain of a domain you already own)
   for $0, or purchase a new domain (the only paid item).

## How predictions are updated

The GitHub Actions workflow `.github/workflows/ingest-and-train.yml` runs weekly
and on manual dispatch. It:

1. Attempts to download the latest data (`scripts/download_data.py`).
2. Builds `predictions/*.json` (`scripts/build_predictions.py`).
3. Commits the JSON back to `main`.

The frontend reads the JSON via jsDelivr:

```
https://cdn.jsdelivr.net/gh/hoskiphotoshare-boop/nfl-ml-data@main/predictions/games_week.json
```

Because jsDelivr caches GitHub content globally, the page loads fast without
any backend server.

## Notes

- Picks are currently stored in `localStorage` (no auth).
- A later enhancement can add a Cloudflare Worker + D1 for multi‑user picks and
  leaderboards, still within the free tier.
- No build step is required; this is plain HTML/CSS/JS.
