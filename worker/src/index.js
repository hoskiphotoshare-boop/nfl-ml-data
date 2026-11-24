/**
 * Cloudflare Worker for NFL ML Picks API.
 *
 * Endpoints:
 *   GET  /api/health
 *   GET  /api/picks?userId=...
 *   POST /api/picks       { userId, gameId, pick, confidence? }
 *   GET  /api/leaderboard
 *
 * Bindings (see wrangler.toml):
 *   DB  – Cloudflare D1 database
 */

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

function error(message, status = 400) {
  return json({ error: message }, status);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (url.pathname === '/api/health') {
      return json({ ok: true, service: 'nfl-ml-picks-api' });
    }

    if (url.pathname === '/api/picks' && request.method === 'GET') {
      const userId = url.searchParams.get('userId');
      if (!userId) return error('userId is required');
      const { results } = await env.DB.prepare(
        'SELECT game_id, pick, confidence, created_at FROM picks WHERE user_id = ? ORDER BY game_id'
      )
        .bind(userId)
        .all();
      return json(results);
    }

    if (url.pathname === '/api/picks' && request.method === 'POST') {
      let body;
      try {
        body = await request.json();
      } catch {
        return error('Invalid JSON body');
      }
      const { userId, gameId, pick, confidence = 0 } = body || {};
      if (!userId || gameId === undefined || !pick) {
        return error('userId, gameId and pick are required');
      }
      if (pick !== 'home' && pick !== 'away') {
        return error("pick must be 'home' or 'away'");
      }

      await env.DB.prepare(
        `INSERT INTO picks (user_id, game_id, pick, confidence)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(user_id, game_id) DO UPDATE SET
           pick = excluded.pick,
           confidence = excluded.confidence,
           created_at = CURRENT_TIMESTAMP`
      )
        .bind(userId, gameId, pick, confidence)
        .run();

      return json({ ok: true });
    }

    if (url.pathname === '/api/leaderboard') {
      const { results } = await env.DB.prepare(
        `SELECT user_id, COUNT(*) AS total_picks
         FROM picks
         GROUP BY user_id
         ORDER BY total_picks DESC
         LIMIT 100`
      ).all();
      return json(results);
    }

    return error('Not found', 404);
  },
};
