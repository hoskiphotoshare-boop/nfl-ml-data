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
 *   DB        – Cloudflare D1 database
 *   API_TOKEN – bearer token required for every request
 *   API_ENABLED – must be the string "true" to accept traffic
 *
 * This worker is not deployed by the site pipeline. It is gated so an
 * accidental deploy cannot expose an unauthenticated write API.
 */

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

const MAX_ID = 64;
const MAX_CONFIDENCE = 100;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

function error(message, status = 400) {
  return json({ error: message }, status);
}

function cleanId(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_ID) return null;
  if (!/^[A-Za-z0-9_.:-]+$/.test(trimmed)) return null;
  return trimmed;
}

function authorized(request, env) {
  const expected = env.API_TOKEN;
  if (!expected) return false;
  const header = request.headers.get('authorization') || '';
  const provided = header.replace(/^Bearer\s+/i, '').trim();
  if (!provided || provided.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i += 1) {
    diff |= provided.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (env.API_ENABLED !== 'true') {
      return error('disabled', 503);
    }
    if (!authorized(request, env)) {
      return error('unauthorized', 401);
    }

    if (url.pathname === '/api/health') {
      return json({ ok: true, service: 'nfl-ml-picks-api' });
    }

    if (url.pathname === '/api/picks' && request.method === 'GET') {
      const userId = cleanId(url.searchParams.get('userId'));
      if (!userId) return error('valid userId is required');
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
      const userId = cleanId(body && body.userId);
      const gameId = cleanId(body && body.gameId);
      const pick = body && body.pick;
      const confidence = body && body.confidence === undefined ? 0 : body.confidence;
      if (!userId || !gameId || !pick) {
        return error('valid userId, gameId and pick are required');
      }
      if (pick !== 'home' && pick !== 'away') {
        return error("pick must be 'home' or 'away'");
      }
      const confidenceValue = Number(confidence);
      if (!Number.isFinite(confidenceValue) || confidenceValue < 0 || confidenceValue > MAX_CONFIDENCE) {
        return error(`confidence must be between 0 and ${MAX_CONFIDENCE}`);
      }

      await env.DB.prepare(
        `INSERT INTO picks (user_id, game_id, pick, confidence)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(user_id, game_id) DO UPDATE SET
           pick = excluded.pick,
           confidence = excluded.confidence,
           created_at = CURRENT_TIMESTAMP`
      )
        .bind(userId, gameId, pick, Math.round(confidenceValue))
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
