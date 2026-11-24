// Frontend for NFL ML Predictions.
// Prediction and report JSON are served same-origin under /data.
// Picks are stored in localStorage; an optional API can be enabled by setting
// `window.NFL_API_BASE` before this script loads.

const BASE = '/data/predictions';
const REPORTS_BASE = '/data/reports';
const API_BASE = window.NFL_API_BASE || null;
const PICKS_KEY = 'nfl_ml_picks';

let picks = {};
try {
  picks = JSON.parse(localStorage.getItem(PICKS_KEY) || '{}');
} catch {
  picks = {};
}

async function fetchJSON(name) {
  // Cache-bust so the browser always gets the latest predictions.
  const url = `${BASE}/${name}?t=${Date.now()}`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json();
}

async function fetchReport(name) {
  const url = `${REPORTS_BASE}/${name}?t=${Date.now()}`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function decimalOdds(prob) {
  if (prob === undefined || prob === null || prob <= 0 || prob >= 1) return null;
  return (1 / prob).toFixed(2);
}

function savePicks() {
  localStorage.setItem(PICKS_KEY, JSON.stringify(picks));
}

function setPick(gameId, side) {
  picks[gameId] = side;
  savePicks();
  renderGames(window.__games || []);
  renderPicks();
}

function renderPicks() {
  const container = document.getElementById('picks-list');
  if (!container) return;
  container.innerHTML = '';
  const entries = Object.entries(picks);
  if (!entries.length) {
    container.appendChild(el('p', 'muted', 'No picks yet. Click “Pick Home” or “Pick Away” on a game.'));
    return;
  }
  entries.forEach(([gameId, side]) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', `Game ${gameId}`));
    card.appendChild(el('p', '', `Your pick: ${side === 'home' ? 'Home team' : 'Away team'}`));
    container.appendChild(card);
  });
}

function renderGames(games) {
  window.__games = games;
  const container = document.getElementById('games-list');
  if (!container) return;
  container.innerHTML = '';
  if (!games.length) {
    container.appendChild(el('p', 'muted', 'No games available yet.'));
    return;
  }
  games.forEach((g) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', `${g.away_team} @ ${g.home_team}`));
    card.appendChild(el('p', '', `Week ${g.week} · ${g.season}`));
    const homeOdds = decimalOdds(g.home_win_prob);
    const awayOdds = decimalOdds(1 - g.home_win_prob);
    const homeLine = `Home: ${(g.home_win_prob * 100).toFixed(0)}%${homeOdds ? ` (${homeOdds})` : ''}`;
    const awayLine = `Away: ${((1 - g.home_win_prob) * 100).toFixed(0)}%${awayOdds ? ` (${awayOdds})` : ''}`;
    card.appendChild(el('p', 'prob' + (g.home_win_prob < 0.5 ? ' low' : ''), homeLine));
    card.appendChild(el('p', 'odds', awayLine));
    card.appendChild(el('p', '', `Spread: ${g.spread > 0 ? '+' : ''}${g.spread}`));
    card.appendChild(el('p', '', `Total: ${g.total}`));

    const marketGame = window.__marketById?.get(g.game_id);
    if (marketGame && marketGame.market_home_win_prob !== undefined) {
      const edgePct = (marketGame.edge * 100).toFixed(1);
      card.appendChild(
        el(
          'p',
          'odds',
          `Market: ${(marketGame.market_home_win_prob * 100).toFixed(1)}% · Edge ${
            edgePct >= 0 ? '+' : ''
          }${edgePct}%`
        )
      );
    }
    if (g.calibrated_home_win_prob !== undefined) {
      card.appendChild(
        el('p', 'odds', `Calibrated: ${(g.calibrated_home_win_prob * 100).toFixed(1)}%`)
      );
    }

    const buttons = el('div', 'pick-buttons');
    const homeBtn = el('button', 'pick-btn', 'Pick Home');
    const awayBtn = el('button', 'pick-btn', 'Pick Away');
    if (picks[g.game_id] === 'home') homeBtn.classList.add('selected');
    if (picks[g.game_id] === 'away') awayBtn.classList.add('selected');
    homeBtn.addEventListener('click', () => setPick(g.game_id, 'home'));
    awayBtn.addEventListener('click', () => setPick(g.game_id, 'away'));
    buttons.appendChild(homeBtn);
    buttons.appendChild(awayBtn);
    card.appendChild(buttons);

    container.appendChild(card);
  });
  renderPicks();
}

function renderPlayers(players) {
  const container = document.getElementById('players-list');
  if (!container) return;
  container.innerHTML = '';
  if (!players.length) {
    container.appendChild(el('p', 'muted', 'No player projections yet.'));
    return;
  }
  players.forEach((p) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', `${p.name} (${p.position})`));
    card.appendChild(el('p', '', `${p.team}`));
    card.appendChild(el('p', '', `Targets: ${p.projected_targets}`));
    card.appendChild(el('p', '', `Receptions: ${p.projected_receptions}`));
    card.appendChild(el('p', '', `Yards: ${p.projected_yards}`));
    card.appendChild(el('p', '', `TDs: ${p.projected_tds}`));
    if (p.projected_pass_yards !== undefined) {
      card.appendChild(el('p', '', `Pass yards: ${p.projected_pass_yards}`));
      card.appendChild(el('p', '', `Pass TDs: ${p.projected_pass_tds}`));
    }
    if (p.projected_fantasy_points !== undefined) {
      card.appendChild(el('p', '', `Fantasy pts: ${p.projected_fantasy_points}`));
    }
    container.appendChild(card);
  });
}

function renderPlayoffs(data) {
  const container = document.getElementById('playoffs-list');
  if (!container) return;
  container.innerHTML = '';
  if (!data.length) {
    container.appendChild(el('p', 'muted', 'No playoff odds yet.'));
    return;
  }
  data.forEach((t) => {
    const card = el('div', 'card');
    const title = t.conference ? `${t.team} (${t.conference})` : t.team;
    card.appendChild(el('h3', '', title));
    const po = decimalOdds(t.playoff_prob);
    const cpo = decimalOdds(t.conference_prob);
    const sbo = decimalOdds(t.super_bowl_prob);
    card.appendChild(
      el('p', 'prob' + (t.playoff_prob < 0.5 ? ' low' : ''), `Playoff: ${(t.playoff_prob * 100).toFixed(0)}%${po ? ` (${po})` : ''}`)
    );
    if (t.conference_prob !== undefined) {
      card.appendChild(
        el('p', 'prob' + (t.conference_prob < 0.25 ? ' low' : ''), `Conference: ${(t.conference_prob * 100).toFixed(0)}%${cpo ? ` (${cpo})` : ''}`)
      );
    }
    card.appendChild(
      el('p', 'prob' + (t.super_bowl_prob < 0.1 ? ' low' : ''), `Super Bowl: ${(t.super_bowl_prob * 100).toFixed(0)}%${sbo ? ` (${sbo})` : ''}`)
    );
    container.appendChild(card);
  });
}

function renderMarket(data) {
  const summaryEl = document.getElementById('market-summary');
  const listEl = document.getElementById('market-list');
  if (!summaryEl || !listEl) return;

  summaryEl.innerHTML = '';
  listEl.innerHTML = '';

  const summary = data.summary || {};
  const summaryItems = [
    ['Games compared', summary.games_compared ?? '—'],
    ['Mean edge', summary.mean_edge !== undefined ? `${(summary.mean_edge * 100).toFixed(1)}%` : '—'],
    ['Mean abs edge', summary.mean_abs_edge !== undefined ? `${(summary.mean_abs_edge * 100).toFixed(1)}%` : '—'],
    ['Max abs edge', summary.max_abs_edge !== undefined ? `${(summary.max_abs_edge * 100).toFixed(1)}%` : '—'],
  ];
  summaryItems.forEach(([label, value]) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', label));
    card.appendChild(el('p', 'prob', String(value)));
    summaryEl.appendChild(card);
  });

  const games = data.games || [];
  if (!games.length) {
    listEl.appendChild(el('p', 'muted', 'No model-vs-market games available yet.'));
    return;
  }

  games.forEach((g) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', `${g.away_team} @ ${g.home_team}`));
    card.appendChild(el('p', '', `Week ${g.week} · ${g.season}`));
    card.appendChild(
      el('p', 'prob', `Model: ${(g.model_home_win_prob * 100).toFixed(1)}%`)
    );
    card.appendChild(
      el('p', 'odds', `Market: ${(g.market_home_win_prob * 100).toFixed(1)}%`)
    );
    const edge = g.edge * 100;
    const edgeClass = Math.abs(edge) >= 10 ? 'prob low' : 'prob';
    card.appendChild(
      el('p', edgeClass, `Edge: ${edge >= 0 ? '+' : ''}${edge.toFixed(1)}%`)
    );
    if (g.model_spread !== null && g.market_spread !== null) {
      card.appendChild(el('p', '', `Spread: model ${g.model_spread} · market ${g.market_spread}`));
    }
    if (g.model_total !== null && g.market_total !== null) {
      card.appendChild(el('p', '', `Total: model ${g.model_total} · market ${g.market_total}`));
    }
    listEl.appendChild(card);
  });
}


function setupTabs() {
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      panels.forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tab.dataset.tab).classList.add('active');
    });
  });
}

async function loadGames() {
  try {
    return await fetchJSON('upcoming_games.json');
  } catch {
    return fetchJSON('games_week.json');
  }
}

async function loadAll() {
  setupTabs();
  renderPicks();
  try {
    const [games, players, playoffs, market] = await Promise.all([
      loadGames(),
      fetchJSON('players_week.json'),
      fetchJSON('playoffs.json'),
      fetchReport('market_comparison.json').catch(() => null),
    ]);
    if (market) {
      window.__marketById = new Map(
        (market.games || []).map((g) => [g.game_id, g])
      );
    }
    renderGames(games);
    renderPlayers(players);
    renderPlayoffs(playoffs);
    if (market) renderMarket(market);
  } catch (err) {
    console.error(err);
    const container = document.getElementById('games-list');
    if (container) {
      container.innerHTML =
        '<p class="muted">Could not load predictions. The backend may still be generating data.</p>';
    }
  }
}

document.addEventListener('DOMContentLoaded', loadAll);