const SUMMARY =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary";
const CACHE_PREFIX = "https://player-stats.internal/";
const TEAM_ALIASES = { LAR: "LA", WSH: "WAS", JAC: "JAX" };
const GROUPS = ["passing", "rushing", "receiving", "kicking"];

function teamAbbr(team) {
  const abbr = (team && team.abbreviation) || "";
  return TEAM_ALIASES[abbr] || abbr;
}

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status: status || 200,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "public, max-age=15, stale-while-revalidate=30",
    },
  });
}

function statusOf(payload) {
  const competition = ((payload.header || {}).competitions || [])[0] || {};
  const status = competition.status || {};
  const type = status.type || {};
  return {
    state: type.state || "pre",
    completed: Boolean(type.completed),
    period: status.period || null,
    clock: status.displayClock || null,
    detail: type.shortDetail || type.detail || type.description || null,
  };
}

function shape(payload, eventId) {
  const teams = {};
  for (const side of (payload.boxscore || {}).players || []) {
    const abbr = teamAbbr(side.team);
    if (!abbr) continue;
    const groups = {};
    for (const group of side.statistics || []) {
      if (!GROUPS.includes(group.name)) continue;
      const players = (group.athletes || [])
        .filter((row) => row && row.athlete && row.athlete.id)
        .map((row) => ({
          id: String(row.athlete.id),
          name: row.athlete.displayName || row.athlete.shortName || "",
          stats: row.stats || [],
        }));
      if (!players.length) continue;
      groups[group.name] = { labels: group.labels || [], players };
    }
    if (Object.keys(groups).length) teams[abbr] = groups;
  }
  return {
    updated_at: new Date().toISOString(),
    event_id: String(eventId),
    source: "espn",
    status: statusOf(payload),
    teams,
  };
}

export async function onRequestGet(context) {
  const url = new URL(context.request.url);
  const eventId = (url.searchParams.get("event") || "").trim();
  if (!/^\d{1,12}$/.test(eventId)) {
    return json({ error: "event query parameter is required", teams: {} }, 400);
  }

  const cache = caches.default;
  const key = new Request(`${CACHE_PREFIX}${eventId}`, { method: "GET" });
  const cached = await cache.match(key);
  if (cached) return cached;

  try {
    const response = await fetch(`${SUMMARY}?event=${eventId}`, {
      headers: { Accept: "application/json", "User-Agent": "nfl-ml-stats" },
      cf: { cacheTtl: 20 },
    });
    if (!response.ok) throw new Error(`ESPN ${response.status}`);
    const output = json(shape(await response.json(), eventId));
    context.waitUntil(cache.put(key, output.clone()));
    return output;
  } catch (error) {
    if (cached) return cached;
    return json(
      { error: String(error), event_id: eventId, teams: {} },
      502
    );
  }
}
