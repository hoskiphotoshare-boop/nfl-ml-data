# NFL Picks API – Cloudflare Worker + D1

Optional backend for multi-user picks and leaderboards. Everything stays within
Cloudflare’s free tier.

**This worker is not deployed by the site pipeline.** It is gated so an
accidental deploy cannot expose an open write API: without `API_ENABLED` set to
`"true"` every request returns `503`, and without a matching `API_TOKEN` every
request returns `401`.

## Prerequisites

- Cloudflare account (free)
- `wrangler` CLI (`npm install -g wrangler@4.169.0`)

## 1. Create the D1 database

```bash
wrangler d1 create nfl_ml_picks
```

Copy the printed `database_id` into `worker/wrangler.toml`.

## 2. Apply the schema

```bash
wrangler d1 execute nfl_ml_picks --file=worker/schema.sql
```

## 3. Configure the gate

In `worker/wrangler.toml`:

```toml
[vars]
API_ENABLED = "true"
```

Store the bearer token as a secret (use a long random value; do not commit it):

```bash
cd worker
wrangler secret put API_TOKEN
```

## 4. Deploy the Worker

```bash
cd worker
wrangler deploy
```

Wrangler prints a URL such as:

```
https://nfl-ml-picks-api.<your-subdomain>.workers.dev
```

## 5. Connect the frontend

In `frontend/index.html`, add **before** loading `app.js`:

```html
<script>
  window.NFL_API_BASE = 'https://nfl-ml-picks-api.<your-subdomain>.workers.dev';
</script>
```

When `NFL_API_BASE` is set, the frontend can use the Worker for picks and
leaderboards. Without it, picks stay in `localStorage` (fully offline).

The frontend does not ship a way to send `Authorization` today, so wiring the
Worker into the UI also requires passing `API_TOKEN` from a same-origin proxy —
do not embed the token in `frontend/`.

## API

Every request needs `Authorization: Bearer <API_TOKEN>`.

| Method | Path | Body / Query | Description |
|--------|------|--------------|-------------|
| `GET` | `/api/health` | – | Health check |
| `GET` | `/api/picks?userId=alice` | – | List picks for a user |
| `POST` | `/api/picks` | `{ userId, gameId, pick, confidence? }` | Upsert a pick |
| `GET` | `/api/leaderboard` | – | Top users by total picks |

Validation:

- `pick` must be `"home"` or `"away"`.
- `userId` and `gameId` are trimmed, capped at 64 characters, and limited to
  `[A-Za-z0-9_.:-]`.
- `confidence` is numeric and must be within `0–100`.

## Free‑tier notes

- D1: 5 GB storage, 25 M reads/day, 50 k writes/day.
- Workers: 100 k requests/day.
- No servers, no paid add‑ons.
