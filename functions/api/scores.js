const ESPN =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const CACHE_KEY = "https://scores.internal/nfl-scoreboard";
const TEAM_ALIASES = { LAR: "LA", WSH: "WAS", JAC: "JAX" };

function normalizeTeam(abbr) {
  return TEAM_ALIASES[abbr] || abbr;
}

function gameId(season, week, away, home) {
  return `${season}_${String(week).padStart(2, "0")}_${normalizeTeam(
    away
  )}_${normalizeTeam(home)}`;
}

function scoreValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function shape(payload) {
  const games = (payload.events || []).map((event) => {
    const competition = (event.competitions || [])[0] || {};
    const competitors = competition.competitors || [];
    const home = competitors.find((item) => item.homeAway === "home") || {};
    const away = competitors.find((item) => item.homeAway === "away") || {};
    const season = event.season && event.season.year;
    const week = event.week && event.week.number;
    if (!season || !week || !home.team || !away.team) return null;
    const status = competition.status || event.status || {};
    return {
      game_id: gameId(
        season,
        week,
        away.team.abbreviation,
        home.team.abbreviation
      ),
      espn_id: event.id || null,
      date: event.date || competition.date || null,
      season,
      week,
      status: {
        state: status.type && status.type.state ? status.type.state : "pre",
        completed: Boolean(status.type && status.type.completed),
        period: status.period || null,
        clock: status.displayClock || null,
        detail: status.type
          ? status.type.shortDetail || status.type.detail || status.type.description
          : null,
      },
      away: {
        team: normalizeTeam(away.team.abbreviation),
        score: scoreValue(away.score),
      },
      home: {
        team: normalizeTeam(home.team.abbreviation),
        score: scoreValue(home.score),
      },
    };
  });
  return {
    updated_at: new Date().toISOString(),
    source: "espn",
    games: games.filter(Boolean),
  };
}

export async function onRequestGet(context) {
  const cache = caches.default;
  const key = new Request(CACHE_KEY, { method: "GET" });
  const cached = await cache.match(key);
  if (cached) return cached;

  try {
    const response = await fetch(ESPN, {
      headers: { Accept: "application/json", "User-Agent": "nfl-ml-scores" },
      cf: { cacheTtl: 20 },
    });
    if (!response.ok) throw new Error(`ESPN ${response.status}`);
    const body = shape(await response.json());
    const output = new Response(JSON.stringify(body), {
      headers: {
        "content-type": "application/json; charset=utf-8",
        "cache-control": "public, max-age=15, stale-while-revalidate=30",
      },
    });
    context.waitUntil(cache.put(key, output.clone()));
    return output;
  } catch (error) {
    if (cached) return cached;
    return new Response(
      JSON.stringify({ error: "upstream_unavailable", games: [] }),
      {
        status: 502,
        headers: { "content-type": "application/json; charset=utf-8" },
      }
    );
  }
}
