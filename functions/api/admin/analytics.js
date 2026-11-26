/**
 * Admin-only traffic summary.
 *
 * Requires a Google ID token whose audience is the app's web client, whose
 * email is verified and allow-listed, and whose expiry is still in the future.
 * Data comes from the ANALYTICS_KV batches written by /api/analytics/collect.
 */

const MAX_BATCHES = 6000;
const MAX_DAYS = 90;

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "private, max-age=60",
      ...headers,
    },
  });
}

async function digest(value) {
  const bytes = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(hash)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function verifyToken(token, env) {
  if (!token || !env.GOOGLE_CLIENT_ID) return false;
  const cacheKey = new Request(`https://analytics.internal/token/${await digest(token)}`);
  try {
    const hit = await caches.default.match(cacheKey);
    if (hit) return hit.ok;
  } catch {
    /* cache is optional */
  }

  let ok = false;
  try {
    const res = await fetch(
      `https://oauth2.googleapis.com/tokeninfo?id_token=${encodeURIComponent(token)}`
    );
    if (res.ok) {
      const info = await res.json();
      const emails = String(env.ADMIN_EMAILS || "")
        .split(",")
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean);
      ok =
        info.aud === env.GOOGLE_CLIENT_ID &&
        String(info.email_verified) === "true" &&
        Number(info.exp) * 1000 > Date.now() &&
        emails.includes(String(info.email || "").toLowerCase());
    }
  } catch {
    ok = false;
  }

  const out = new Response(ok ? "ok" : "no", {
    status: ok ? 200 : 401,
    headers: { "cache-control": "max-age=300" },
  });
  try {
    await caches.default.put(cacheKey, out.clone());
  } catch {
    /* cache is optional */
  }
  return ok;
}

function top(counter, limit = 10) {
  return Object.entries(counter)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => ({ label, count }));
}

function bump(counter, key) {
  const label = String(key || "unknown").slice(0, 80);
  counter[label] = (counter[label] || 0) + 1;
}

async function listBatches(kv, prefix) {
  const names = [];
  let cursor;
  for (let page = 0; page < 20 && names.length < MAX_BATCHES; page += 1) {
    const result = await kv.list({ prefix, limit: 1000, cursor });
    for (const key of result.keys || []) {
      names.push(key.name);
      if (names.length >= MAX_BATCHES) break;
    }
    if (result.list_complete || !result.cursor) break;
    cursor = result.cursor;
  }
  return names;
}

function dayList(days) {
  const now = new Date();
  const list = [];
  for (let offset = days - 1; offset >= 0; offset -= 1) {
    const date = new Date(
      Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - offset)
    );
    list.push(date.toISOString().slice(0, 10));
  }
  return list;
}

export async function onRequestGet(context) {
  const { request, env } = context;
  if (!env.ANALYTICS_KV) {
    return json({ error: "analytics_not_configured" }, 503);
  }

  const auth = request.headers.get("authorization") || "";
  const token = auth.replace(/^Bearer\s+/i, "").trim();
  // Local-only test hook. Production never sets ANALYTICS_DEV_MODE.
  const devMode =
    env.ANALYTICS_DEV_MODE === "1" &&
    request.headers.get("x-analytics-dev") === "local";
  if (!devMode && !(await verifyToken(token, env))) {
    return json({ error: "unauthorized" }, 401);
  }

  const url = new URL(request.url);
  const requested = Number.parseInt(url.searchParams.get("days") || "7", 10);
  const days = Math.min(MAX_DAYS, Math.max(1, Number.isFinite(requested) ? requested : 7));
  const dates = dayList(days);

  const names = [];
  for (const day of dates) {
    const prefix = `b:${day}:`;
    const keys = await listBatches(env.ANALYTICS_KV, prefix);
    names.push(...keys);
  }
  const truncated = names.length >= MAX_BATCHES;
  const selected = names.slice(0, MAX_BATCHES);

  const daily = new Map(
    dates.map((day) => [
      day,
      { day, pageviews: 0, visits: new Set(), events: 0, clicks: 0 },
    ])
  );
  const pages = {};
  const tabs = {};
  const views = {};
  const clicks = {};
  const referrers = {};
  const countries = {};
  const devices = {};
  let pageviews = 0;
  let eventsTotal = 0;
  let clicksTotal = 0;

  for (let index = 0; index < selected.length; index += 40) {
    const chunk = selected.slice(index, index + 40);
    const records = await Promise.all(
      chunk.map((key) => env.ANALYTICS_KV.get(key, { type: "json" }).catch(() => null))
    );
    for (const record of records) {
      if (!record || !record.day) continue;
      const day = daily.get(record.day);
      if (!day) continue;
      const events = Array.isArray(record.events) ? record.events : [];
      day.events += events.length;
      eventsTotal += events.length;
      if (record.visitor) day.visits.add(record.visitor);

      for (const event of events) {
        const name = String(event && event.e ? event.e : "");
        const label = String(event && event.l ? event.l : "");
        if (name === "pageview") {
          day.pageviews += 1;
          pageviews += 1;
          bump(pages, record.path || "/");
          bump(referrers, record.ref || "direct");
          bump(countries, record.country || "XX");
          bump(devices, record.device || "unknown");
        } else if (name === "tab_view") {
          bump(tabs, label);
        } else if (name === "view_change") {
          bump(views, label);
        } else if (
          name === "game_open" ||
          name === "player_open" ||
          name === "team_link"
        ) {
          day.clicks += 1;
          clicksTotal += 1;
          bump(clicks, `${name}:${label}`);
        }
      }
    }
  }

  const byDay = dates.map((date) => {
    const day = daily.get(date);
    return {
      day: date,
      pageviews: day.pageviews,
      visits: day.visits.size,
      events: day.events,
      clicks: day.clicks,
    };
  });
  const visits = byDay.reduce((sum, day) => sum + day.visits, 0);

  return json({
    ok: true,
    generated_at: new Date().toISOString(),
    range_days: days,
    batches: selected.length,
    truncated,
    totals: {
      pageviews,
      visits,
      events: eventsTotal,
      clicks: clicksTotal,
    },
    by_day: byDay,
    pages: top(pages),
    tabs: top(tabs),
    views: top(views),
    clicks: top(clicks),
    referrers: top(referrers),
    countries: top(countries),
    devices: top(devices),
    note:
      "Cookie-free first-party counts. Visits are daily unique hashes and are " +
      "summed across days; no IP or user agent is stored. UTC days.",
  });
}

export function onRequestPost() {
  return json({ error: "method_not_allowed" }, 405);
}
