# NFL Picks API – Cloudflare Worker + D1

Optional backend for multi‑user picks and leaderboards. Everything stays within
Cloudflare’s free tier.

## Prerequisites

- Cloudflare account (free)
- `wrangler` CLI (`npm install -g wrangler`)

## 1. Create the D1 database

```bash
wrangler d1 create nfl_ml_picks
```

Copy the printed `database_id` into `worker/wrangler.toml`.

## 2. Apply the schema

```bash
wrangler d1 execute nfl_ml_picks --file=worker/schema.sql
```

## 3. Deploy the Worker

```bash
cd worker
wrangler deploy
```

Wrangler prints a URL such as:

```
https://nfl-ml-picks-api.<your-subdomain>.workers.dev
```

## 4. Connect the frontend

In `frontend/index.html`, add **before** loading `app.js`:

```html
<script>
  window.NFL_API_BASE = 'https://nfl-ml-picks-api.<your-subdomain>.workers.dev';
</script>
```

When `NFL_API_BASE` is set, the frontend can use the Worker for picks and
leaderboards. Without it, picks stay in `localStorage` (fully offline).

## API

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| `GET` | `/api/health` | – | Health check |
| `GET` | `/api/picks?userId=alice` | – | List picks for a user |
| `POST` | `/api/picks` | `{ userId, gameId, pick, confidence? }` | Upsert a pick |
| `GET` | `/api/leaderboard` | – | Top users by total picks |

`pick` must be `"home"` or `"away"`.

## Free‑tier notes

- D1: 5 GB storage, 25 M reads/day, 50 k writes/day.
- Workers: 100 k requests/day.
- No servers, no paid add‑ons.
