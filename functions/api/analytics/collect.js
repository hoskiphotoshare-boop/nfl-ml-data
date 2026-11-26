/**
 * First-party, cookie-free analytics collector.
 *
 * One KV write per batched page interaction. No IPs, user agents, cookies or
 * persistent identifiers are stored: the visitor value is a daily-salted hash
 * truncated to 64 bits, and raw events are only kept as per-batch JSON.
 */

const ALLOWED_EVENTS = new Set([
  "pageview",
  "tab_view",
  "view_change",
  "game_open",
  "player_open",
  "team_link",
  "sort_change",
  "filter_change",
  "theme_toggle",
]);

const BOT_RE = /bot|crawler|spider|slurp|bingpreview|headless|python-requests|curl|wget/i;
const PROD_ORIGIN = "https://nfl-ml-data.pages.dev";
const MAX_BODY = 16 * 1024;
const MAX_EVENTS = 25;
const RETENTION_SECONDS = 365 * 24 * 60 * 60;

function noContent() {
  return new Response(null, { status: 204 });
}

function clean(value, max) {
  return String(value == null ? "" : value)
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .trim()
    .slice(0, max);
}

function allowedOrigin(origin) {
  if (!origin) return true;
  if (origin === PROD_ORIGIN) return true;
  if (/^https:\/\/[a-z0-9-]+\.nfl-ml-data\.pages\.dev$/.test(origin)) return true;
  if (/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return true;
  return false;
}

async function visitorHash(env, day, ip, ua) {
  const salt = env.ANALYTICS_SALT;
  if (!salt) return null;
  const bytes = new TextEncoder().encode(`${salt}|${day}|${ip}|${ua}`);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .slice(0, 8)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function referrerHost(raw, requestHost) {
  if (!raw) return "direct";
  try {
    const host = new URL(raw).hostname.replace(/^www\./, "");
    return host === requestHost ? "internal" : host;
  } catch {
    return "direct";
  }
}

function deviceClass(body, ua) {
  const value = String(body.device || "");
  if (["mobile", "tablet", "desktop"].includes(value)) return value;
  if (/iPad|Tablet/i.test(ua)) return "tablet";
  if (/Mobi|Android|iPhone/i.test(ua)) return "mobile";
  return "desktop";
}

export async function onRequestPost(context) {
  const { request, env } = context;
  try {
    if (!env.ANALYTICS_KV || !env.ANALYTICS_SALT) return noContent();
    if (request.headers.get("dnt") === "1" || request.headers.get("sec-gpc") === "1") {
      return noContent();
    }
    if (!allowedOrigin(request.headers.get("origin"))) {
      return new Response("forbidden", { status: 403 });
    }

    const ua = request.headers.get("user-agent") || "";
    if (BOT_RE.test(ua)) return noContent();

    const text = await request.text();
    if (!text || text.length > MAX_BODY) return noContent();

    let body;
    try {
      body = JSON.parse(text);
    } catch {
      return noContent();
    }

    const rawEvents = Array.isArray(body.events) ? body.events : [];
    const events = [];
    for (const event of rawEvents.slice(0, MAX_EVENTS)) {
      const name = clean(event && event.e, 32);
      if (!ALLOWED_EVENTS.has(name)) continue;
      events.push({ e: name, l: clean(event.l, 64) });
    }
    if (!events.length) return noContent();

    const now = new Date();
    const day = now.toISOString().slice(0, 10);
    const hour = now.toISOString().slice(11, 13);
    const ip = request.headers.get("cf-connecting-ip") || "";
    const visitor = await visitorHash(env, day, ip, ua);
    if (!visitor) return noContent();

    let path = clean(body.path || "/", 120);
    if (!path.startsWith("/")) path = "/" + path;
    path = path.split("?")[0].split("#")[0];
    if (!path) path = "/";

    const requestHost = new URL(request.url).hostname;
    const record = {
      v: 1,
      day,
      path,
      ref: referrerHost(body.referrer || request.headers.get("referer") || "", requestHost),
      device: deviceClass(body, ua),
      country: (request.cf && request.cf.country) || "XX",
      visitor,
      events,
    };

    const key = `b:${day}:${hour}:${Date.now().toString(36)}:${crypto
      .randomUUID()
      .slice(0, 8)}`;
    await env.ANALYTICS_KV.put(key, JSON.stringify(record), {
      expirationTtl: RETENTION_SECONDS,
    });
  } catch {
    // Analytics must never break the page.
  }
  return noContent();
}

export function onRequestGet() {
  return new Response("Method not allowed", { status: 405 });
}
