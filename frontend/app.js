// NFL ML Predictions — frontend
// Data is served same-origin from /data (prediction + report JSON).
// Picks are stored in localStorage; every view is rendered client-side.

const BASE = '/data/predictions';
const REPORTS_BASE = '/data/reports';
const PICKS_KEY = 'nfl_ml_picks';

const TEAMS = {
  ARI: ['Cardinals', '#97233F'],
  ATL: ['Falcons', '#A71930'],
  BAL: ['Ravens', '#241773'],
  BUF: ['Bills', '#00338D'],
  CAR: ['Panthers', '#0085CA'],
  CHI: ['Bears', '#0B162A'],
  CIN: ['Bengals', '#FB4F14'],
  CLE: ['Browns', '#311D00'],
  DAL: ['Cowboys', '#041E42'],
  DEN: ['Broncos', '#FB4F14'],
  DET: ['Lions', '#0076B6'],
  GB: ['Packers', '#203731'],
  HOU: ['Texans', '#03202F'],
  IND: ['Colts', '#002C5F'],
  JAX: ['Jaguars', '#006778'],
  KC: ['Chiefs', '#E31837'],
  LA: ['Rams', '#003594'],
  LAC: ['Chargers', '#0080C6'],
  LV: ['Raiders', '#1F2937'],
  MIA: ['Dolphins', '#008E97'],
  MIN: ['Vikings', '#4F2683'],
  NE: ['Patriots', '#002244'],
  NO: ['Saints', '#D3BC8D'],
  NYG: ['Giants', '#0B2265'],
  NYJ: ['Jets', '#125740'],
  PHI: ['Eagles', '#004C54'],
  PIT: ['Steelers', '#FFB612'],
  SEA: ['Seahawks', '#002244'],
  SF: ['49ers', '#AA0000'],
  TB: ['Buccaneers', '#D50A0A'],
  TEN: ['Titans', '#0C2340'],
  WAS: ['Commanders', '#5A1414'],
};

const INJURY_LABEL = {
  out: 'Out',
  doubtful: 'Doubtful',
  questionable: 'Questionable',
  probable: 'Probable',
  ir: 'Injured reserve',
  na: 'Not active',
};

const state = {
  games: [],
  players: [],
  playoffs: [],
  market: null,
  meta: null,
  marketById: new Map(),
  gamesById: new Map(),
  playersPos: 'QB',
  playoffsConf: 'all',
  picks: {},
};

/* ---------------- persistence ---------------- */

try {
  state.picks = JSON.parse(localStorage.getItem(PICKS_KEY) || '{}');
} catch {
  state.picks = {};
}

/* ---------------- small helpers ---------------- */

const $ = (sel) => document.querySelector(sel);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clear(node) {
  if (node) node.innerHTML = '';
}

function pct(value, digits = 0) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function signed(value, digits = 1, suffix = '') {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`;
}

function formatEv(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  if (value >= 99.95) return '99%+';
  if (value <= -99.95) return '-99%+';
  return signed(value, 1, '%');
}

function decimalOdds(prob) {
  if (prob === undefined || prob === null || prob <= 0 || prob >= 1) return null;
  return (1 / prob).toFixed(2);
}

function american(prob) {
  if (prob === undefined || prob === null || prob <= 0 || prob >= 1) return null;
  const raw = prob >= 0.5 ? (-100 * prob) / (1 - prob) : (100 * (1 - prob)) / prob;
  return `${raw > 0 ? '+' : ''}${Math.round(raw)}`;
}

function americanFromDecimal(dec) {
  if (!dec || dec <= 1) return null;
  const raw = dec >= 2 ? (dec - 1) * 100 : -100 / (dec - 1);
  return `${raw > 0 ? '+' : ''}${Math.round(raw)}`;
}

function teamAbbr(value) {
  return String(value || '').toUpperCase();
}

function teamFull(abbr) {
  const meta = TEAMS[teamAbbr(abbr)];
  return meta ? meta[0] : abbr;
}

function teamColor(abbr) {
  const meta = TEAMS[teamAbbr(abbr)];
  return meta ? meta[1] : '#334155';
}

function hexToRgb(hex) {
  const clean = hex.replace('#', '');
  const n = parseInt(clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance([r, g, b]) {
  const a = [r, g, b].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
}

function badgeColor(abbr) {
  let rgb = hexToRgb(teamColor(abbr));
  if (luminance(rgb) < 0.06) {
    // Lift near-black team colors so the badge reads on the dark surface.
    rgb = rgb.map((v) => Math.round(v + (255 - v) * 0.42));
  }
  return rgb;
}

function teamBadge(abbr, small = false) {
  const rgb = badgeColor(abbr);
  const node = el('span', `team-badge${small ? ' sm' : ''}`, teamAbbr(abbr));
  node.style.background = `rgb(${rgb.join(',')})`;
  node.style.color = luminance(rgb) > 0.55 ? '#0a1020' : '#ffffff';
  node.title = `${teamFull(abbr)} (${teamAbbr(abbr)})`;
  return node;
}

function formDots(form) {
  if (!form) return null;
  const wrap = el('span', 'form');
  wrap.setAttribute('aria-label', `Last five: ${form.split('').join(' ')}`);
  form.split('').slice(-5).forEach((raw) => {
    const letter = raw.toUpperCase();
    const cls = letter === 'W' ? 'w' : letter === 'T' ? 't' : 'l';
    wrap.appendChild(el('i', cls, letter));
  });
  return wrap;
}

function dateLabel(iso) {
  if (!iso) return 'Date TBD';
  const d = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}

function kickoffLabel(game) {
  if (!game) return '';
  const day = dateLabel(game.gameday);
  const time = timeLabel(game.gametime);
  return time ? `${day} · ${time}` : day;
}

function timeLabel(value) {
  if (!value) return '';
  const [h, m] = String(value).split(':').map(Number);
  if (Number.isNaN(h)) return '';
  const suffix = h >= 12 ? 'PM' : 'AM';
  const hour = h % 12 === 0 ? 12 : h % 12;
  return `${hour}:${String(Number.isNaN(m) ? 0 : m).padStart(2, '0')} ${suffix} ET`;
}

function relativeTime(iso) {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/* ---------------- shared components ---------------- */

function statTile({ label, value, sub, cls = '' }) {
  const tile = el('div', `stat ${cls}`.trim());
  tile.appendChild(el('div', 'k', label));
  const v = el('div', 'v');
  v.textContent = value ?? '—';
  tile.appendChild(v);
  if (sub) tile.appendChild(el('div', 's', sub));
  else {
    const s = el('div', 's', '\u00a0');
    tile.appendChild(s);
  }
  return tile;
}

function sparkline(values, max) {
  const wrap = el('div', 'spark');
  if (!values || !values.length) {
    wrap.appendChild(el('span', 'muted-2', '—'));
    return wrap;
  }
  const top = max || Math.max(...values, 1);
  values.slice(-6).forEach((value, index, arr) => {
    const bar = el('span', `bar${index === arr.length - 1 ? ' hi' : ''}`);
    bar.style.height = `${Math.max(3, (Math.max(0, value) / top) * 24)}px`;
    bar.title = `${value.toFixed(1)} pts`;
    bar.setAttribute('aria-hidden', 'true');
    wrap.appendChild(bar);
  });
  return wrap;
}

function meter(value, cls = '') {
  const wrap = el('div', `meter ${cls}`.trim());
  const track = el('div', 'track');
  const fill = el('i');
  fill.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
  track.appendChild(fill);
  wrap.appendChild(track);
  wrap.appendChild(el('span', 'val', pct(value)));
  return wrap;
}

function emptyState({ icon = '—', title, message, actionLabel, onAction }) {
  const box = el('div', 'empty');
  box.appendChild(el('div', 'big', icon));
  box.appendChild(el('strong', '', title));
  if (message) box.appendChild(el('span', '', message));
  if (actionLabel && onAction) {
    const btn = el('button', 'pick-btn', actionLabel);
    btn.style.maxWidth = '180px';
    btn.style.marginTop = '0.35rem';
    btn.type = 'button';
    btn.addEventListener('click', onAction);
    box.appendChild(btn);
  }
  return box;
}

/* ---------------- games ---------------- */

function probsFor(game) {
  const home = game.home_win_prob;
  if (home === undefined || home === null) return { home: null, away: null };
  return { home, away: 1 - home };
}

function spreadLine(game) {
  if (game.spread === undefined || game.spread === null) return '—';
  const value = Math.abs(game.spread);
  const favorite = game.spread >= 0 ? game.home_team : game.away_team;
  if (value < 0.05) return 'Pick’em';
  return `${favorite} -${value.toFixed(1)}`;
}

function gameCard(game) {
  const card = el('article', 'game-card');
  const market = state.marketById.get(game.game_id);
  const { home, away } = probsFor(game);
  const homeFav = home !== null && home >= 0.5;

  const top = el('div', 'game-top');
  top.appendChild(el('span', 'when', kickoffLabel(game)));
  const venue = game.stadium || (game.location && game.location !== 'Home' ? game.location : '');
  if (venue) {
    const v = el('span', 'venue', `· ${venue}`);
    v.title = [game.roof, game.surface].filter(Boolean).join(' · ');
    top.appendChild(v);
  }
  top.appendChild(el('span', 'spacer'));
  if (game.div_game) top.appendChild(el('span', 'tag div', 'Div'));
  if (market && market.edge !== undefined && market.edge !== null && Math.abs(market.edge) >= 0.05) {
    top.appendChild(
      el(
        'span',
        `tag ${market.edge >= 0 ? 'edge' : 'neg'}`,
        `${market.edge >= 0 ? '+' : ''}${(market.edge * 100).toFixed(1)}%`
      )
    );
  }
  card.appendChild(top);

  const rows = [
    { team: game.away_team, prob: away, record: game.away_record, form: game.away_form },
    { team: game.home_team, prob: home, record: game.home_record, form: game.home_form },
  ];

  rows.forEach((row) => {
    const isFav = row.prob !== null && row.prob >= 0.5;
    const line = el('div', `team-row${isFav ? ' fav' : ' dog'}`);
    line.appendChild(teamBadge(row.team));

    const names = el('div');
    names.style.minWidth = '0';
    const nameWrap = el('div', 'team-name');
    nameWrap.appendChild(el('span', '', teamAbbr(row.team)));
    if (row.record) nameWrap.appendChild(el('span', 'team-sub', ` ${row.record}`));
    if (row.form) nameWrap.appendChild(formDots(row.form));
    names.appendChild(nameWrap);
    const oddsLine = el(
      'div',
      'team-sub',
      row.prob === null ? '—' : `${american(row.prob)}`
    );
    if (row.prob !== null) oddsLine.title = `Decimal odds ${decimalOdds(row.prob)}`;
    names.appendChild(oddsLine);
    line.appendChild(names);
    line.appendChild(el('span', 'prob', pct(row.prob)));
    card.appendChild(line);
  });

  if (home !== null) {
    const bar = el('div', 'winbar');
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', `${teamAbbr(game.away_team)} ${pct(away)} · ${teamAbbr(game.home_team)} ${pct(home)}`);
    const awaySeg = el('span', 'away');
    awaySeg.style.width = `${Math.max(6, (away || 0) * 100)}%`;
    awaySeg.style.background = `linear-gradient(90deg, rgb(${badgeColor(game.away_team).join(',')}), rgba(${badgeColor(
      game.away_team
    ).join(',')}, 0.65))`;
    const homeSeg = el('span', 'home');
    homeSeg.style.width = `${Math.max(6, (home || 0) * 100)}%`;
    homeSeg.style.background = `linear-gradient(90deg, rgba(${badgeColor(game.home_team).join(',')}, 0.65), rgb(${badgeColor(
      game.home_team
    ).join(',')}))`;
    bar.appendChild(awaySeg);
    bar.appendChild(homeSeg);
    card.appendChild(bar);
  }

  const meta = el('div', 'meta-grid');
  [
    ['Model spread', spreadLine(game)],
    ['Model total', game.total !== null && game.total !== undefined ? game.total.toFixed(1) : '—'],
    [
      'Confidence',
      home === null ? '—' : `${Math.round(Math.abs(home - 0.5) * 200)}%`,
    ],
  ].forEach(([k, v]) => {
    const cell = el('div', 'meta-cell');
    cell.appendChild(el('div', 'k', k));
    cell.appendChild(el('div', 'v', v));
    meta.appendChild(cell);
  });
  card.appendChild(meta);

  if (market) {
    const line = el('div', 'market-line');
    line.appendChild(
      el('span', 'tag neutral', `Market ${pct(market.market_home_win_prob, 1)}`)
    );
    if (market.spread_edge !== null && market.spread_edge !== undefined && Math.abs(market.spread_edge) >= 0.5) {
      const side = market.spread_edge >= 0 ? game.home_team : game.away_team;
      line.appendChild(
        el('span', 'tag neutral', `${teamAbbr(side)} +${Math.abs(market.spread_edge).toFixed(1)}`)
      );
    }
    card.appendChild(line);
  }

  const buttons = el('div', 'pick-buttons');
  [
    { side: 'away', team: game.away_team },
    { side: 'home', team: game.home_team },
  ].forEach(({ side, team }) => {
    const btn = el('button', 'pick-btn', `${side === 'home' ? 'Pick' : 'Pick'} ${teamAbbr(team)}`);
    btn.type = 'button';
    btn.setAttribute('aria-pressed', String(state.picks[game.game_id] === side));
    if (state.picks[game.game_id] === side) btn.classList.add('selected');
    btn.addEventListener('click', () => setPick(game.game_id, side));
    buttons.appendChild(btn);
  });
  card.appendChild(buttons);

  return card;
}

function setPick(gameId, side) {
  if (state.picks[gameId] === side) delete state.picks[gameId];
  else state.picks[gameId] = side;
  localStorage.setItem(PICKS_KEY, JSON.stringify(state.picks));
  renderGames();
  renderPicks();
}

function renderPicks() {
  const container = $('#picks-list');
  if (!container) return;
  clear(container);
  const entries = Object.entries(state.picks);
  if (!entries.length) {
    container.appendChild(
      el('p', 'muted', 'No picks yet. Tap “Pick” on any game — picks stay in this browser.')
    );
    return;
  }
  entries.forEach(([gameId, side]) => {
    const game = state.gamesById.get(gameId);
    const row = el('div', 'pick-row');
    if (game) {
      row.appendChild(teamBadge(side === 'home' ? game.home_team : game.away_team, true));
      const text = el('div');
      text.style.flex = '1';
      text.style.minWidth = '0';
      text.appendChild(el('div', 'team-name', `${game.away_team} @ ${game.home_team}`));
      const prob = side === 'home' ? game.home_win_prob : 1 - game.home_win_prob;
      text.appendChild(
        el(
          'div',
          'team-sub',
          `Week ${game.week} · picked ${side === 'home' ? teamAbbr(game.home_team) : teamAbbr(game.away_team)} · model ${pct(prob)}`
        )
      );
      row.appendChild(text);
    } else {
      row.appendChild(el('div', 'team-name', gameId));
    }
    const remove = el('button', 'pick-btn', 'Remove');
    remove.type = 'button';
    remove.style.flex = '0 0 auto';
    remove.addEventListener('click', () => {
      delete state.picks[gameId];
      localStorage.setItem(PICKS_KEY, JSON.stringify(state.picks));
      renderGames();
      renderPicks();
    });
    row.appendChild(remove);
    container.appendChild(row);
  });
}

function filteredGames() {
  const week = $('#games-week-filter')?.value || 'all';
  const team = ($('#games-team-filter')?.value || '').trim().toLowerCase();
  const sort = $('#games-sort')?.value || 'kickoff';

  const list = state.games.filter((g) => {
    const weekOk = week === 'all' || String(g.week) === week;
    const teamOk =
      !team ||
      (g.home_team || '').toLowerCase().includes(team) ||
      (g.away_team || '').toLowerCase().includes(team);
    return weekOk && teamOk;
  });

  if (sort === 'confidence') {
    list.sort((a, b) => Math.abs((b.home_win_prob ?? 0.5) - 0.5) - Math.abs((a.home_win_prob ?? 0.5) - 0.5));
  } else if (sort === 'close') {
    list.sort((a, b) => Math.abs((a.home_win_prob ?? 0.5) - 0.5) - Math.abs((b.home_win_prob ?? 0.5) - 0.5));
  } else {
    list.sort((a, b) => {
      const key = (g) => `${g.gameday || ''}${g.gametime || ''}`;
      return key(a).localeCompare(key(b));
    });
  }
  return list;
}

function renderGameStats(list) {
  const row = $('#games-stats');
  if (!row) return;
  clear(row);
  const probs = list.map((g) => g.home_win_prob).filter((p) => p !== undefined && p !== null);
  const pick = list
    .map((g) => ({ g, edge: Math.abs((g.home_win_prob ?? 0.5) - 0.5) }))
    .sort((a, b) => b.edge - a.edge)[0];
  const totals = list.map((g) => g.total).filter((t) => t !== null && t !== undefined);
  const edges = list
    .map((g) => state.marketById.get(g.game_id)?.best_ev)
    .filter((e) => e !== null && e !== undefined);

  row.appendChild(statTile({ label: 'Games', value: list.length, sub: 'in current filter' }));
  row.appendChild(
    statTile({
      label: 'Strongest lean',
      value: pick && pick.g.home_win_prob !== null
        ? `${teamAbbr(pick.g.home_win_prob >= 0.5 ? pick.g.home_team : pick.g.away_team)} ${pct(pick.g.home_win_prob >= 0.5 ? pick.g.home_win_prob : 1 - pick.g.home_win_prob)}`
        : '—',
      sub: pick ? `${pick.g.away_team} @ ${pick.g.home_team}` : '',
      cls: 'good',
    })
  );
  row.appendChild(
    statTile({
      label: 'Average total',
      value: totals.length ? (totals.reduce((a, b) => a + b, 0) / totals.length).toFixed(1) : '—',
      sub: `${totals.length} games with a line`,
    })
  );
  row.appendChild(
    statTile({
      label: 'Positive EV',
      value: edges.length ? `${edges.filter((e) => e > 0).length}/${edges.length}` : '—',
      sub: edges.length ? 'games with a model edge' : 'odds load weekly',
      cls: edges.some((e) => e > 0) ? 'good' : '',
    })
  );
}

function renderGames() {
  const container = $('#games-list');
  if (!container) return;
  const list = filteredGames();
  window.__filteredGames = list;

  renderGameStats(list);
  renderPicks();

  const count = $('#games-count');
  if (count) count.textContent = `${list.length} game${list.length === 1 ? '' : 's'}`;

  const title = $('#games-title');
  const week = $('#games-week-filter')?.value || 'all';
  if (title) {
    title.textContent =
      week === 'all'
        ? 'All remaining games'
        : `Week ${week}${state.meta?.season ? ` · ${state.meta.season}` : ''}`;
  }

  clear(container);
  if (!list.length) {
    container.appendChild(
      emptyState({
        icon: '🏈',
        title: 'No games match those filters',
        message: 'Try another week or clear the team filter.',
        actionLabel: 'Clear filters',
        onAction: () => {
          const w = $('#games-week-filter');
          const t = $('#games-team-filter');
          if (w) w.value = 'all';
          if (t) t.value = '';
          renderGames();
        },
      })
    );
    return;
  }

  const sort = $('#games-sort')?.value || 'kickoff';
  if (sort !== 'kickoff') {
    list.forEach((g) => container.appendChild(gameCard(g)));
    return;
  }

  // Date grouping for the chronological view (ESPN-style scoreboard).
  let currentDate = null;
  list.forEach((g) => {
    const label = dateLabel(g.gameday);
    if (label !== currentDate) {
      currentDate = label;
      const head = el('div', 'date-head');
      head.style.gridColumn = '1 / -1';
      head.style.display = 'flex';
      head.style.alignItems = 'center';
      head.style.gap = '0.6rem';
      head.style.margin = '0.35rem 0 -0.1rem';
      head.appendChild(el('span', '', label));
      head.firstChild.style.fontSize = '0.75rem';
      head.firstChild.style.fontWeight = '700';
      head.firstChild.style.letterSpacing = '0.08em';
      head.firstChild.style.textTransform = 'uppercase';
      head.firstChild.style.color = 'var(--muted-2)';
      const rule = el('span');
      rule.style.flex = '1';
      rule.style.height = '1px';
      rule.style.background = 'var(--line)';
      head.appendChild(rule);
      container.appendChild(head);
    }
    container.appendChild(gameCard(g));
  });
}

/* ---------------- players ---------------- */

function playerStatColumns(player) {
  const isQB = player.position === 'QB';
  return {
    ydsLabel: isQB ? 'Pass yds' : 'Yds',
    yds: isQB ? player.projected_pass_yards : player.projected_yards,
    tdLabel: isQB ? 'Pass TD' : 'TD',
    td: isQB ? player.projected_pass_tds : player.projected_tds,
  };
}

function injuryPill(status) {
  if (!status) return el('span', 'muted-2', '—');
  const key = String(status).toLowerCase();
  const label = INJURY_LABEL[key] || key;
  const cls = ['out', 'doubtful', 'ir', 'na'].includes(key) ? 'neg' : 'amber';
  const pill = el('span', `tag ${cls}`, label.slice(0, 4).toUpperCase());
  pill.title = label;
  return pill;
}

function filteredPlayers() {
  const team = ($('#players-team-filter')?.value || '').trim().toLowerCase();
  const sort = $('#players-sort')?.value || 'proj';
  const list = state.players.filter((p) => {
    const posOk = state.playersPos === 'all' || p.position === state.playersPos;
    const teamOk = !team || (p.team || '').toLowerCase().includes(team);
    return posOk && teamOk;
  });
  if (sort === 'recent') {
    list.sort((a, b) => recentAvg(b) - recentAvg(a));
  } else if (sort === 'depth') {
    list.sort((a, b) => (a.depth_rank ?? 99) - (b.depth_rank ?? 99) || proj(b) - proj(a));
  } else {
    list.sort((a, b) => proj(b) - proj(a));
  }
  return list;
}

function proj(player) {
  return player.projected_fantasy_points ?? 0;
}

function recentAvg(player) {
  const values = player.recent_fantasy_points || [];
  if (!values.length) return 0;
  return values.slice(-3).reduce((a, b) => a + b, 0) / Math.min(3, values.length);
}

function renderPlayers() {
  const wrap = $('#players-table-wrap');
  if (!wrap) return;
  const list = filteredPlayers();
  clear(wrap);

  const count = $('#players-count');
  if (count) count.textContent = `${list.length} player${list.length === 1 ? '' : 's'}`;

  const sub = $('#players-sub');
  if (sub) {
    sub.textContent = state.players.length
      ? `Projected fantasy points and stat lines${state.meta?.week ? ` for week ${state.meta.week}` : ''} — sort by recent form or depth chart.`
      : 'Projected fantasy points and stat lines for this week’s matchups.';
  }

  const stats = $('#players-stats');
  if (stats) {
    clear(stats);
    const top = [...list].sort((a, b) => proj(b) - proj(a))[0];
    const injured = list.filter((p) => p.injury_status).length;
    stats.appendChild(statTile({ label: 'Players', value: list.length, sub: 'matching filter' }));
    stats.appendChild(
      statTile({
        label: 'Top projection',
        value: top ? `${proj(top).toFixed(1)}` : '—',
        sub: top ? `${top.name} · ${top.team}` : '',
        cls: 'good',
      })
    );
    stats.appendChild(
      statTile({
        label: 'Average projection',
        value: list.length
          ? (list.reduce((a, b) => a + proj(b), 0) / list.length).toFixed(1)
          : '—',
        sub: 'fantasy points',
      })
    );
    stats.appendChild(
      statTile({
        label: 'On the report',
        value: injured,
        sub: 'players with an injury status',
        cls: injured ? 'warn' : '',
      })
    );
  }

  if (!list.length) {
    wrap.appendChild(
      emptyState({
        icon: '🏈',
        title: 'No players match those filters',
        message: 'Try a different position or clear the team filter.',
        actionLabel: 'Clear filters',
        onAction: () => {
          const t = $('#players-team-filter');
          if (t) t.value = '';
          setPosition('all');
        },
      })
    );
    return;
  }

  const table = el('table', 'dtable');
  const thead = el('thead');
  const headRow = el('tr');
  headRow.appendChild(el('th', '', 'Player'));
  headRow.appendChild(el('th', 'hide-sm', 'Matchup'));
  const projTh = el('th', 'num', 'Proj');
  headRow.appendChild(projTh);
  headRow.appendChild(el('th', 'num hide-sm col-tgt', 'Tgt'));
  headRow.appendChild(el('th', 'num hide-sm col-rec', 'Rec'));
  headRow.appendChild(el('th', 'num hide-sm col-yds', 'Yds'));
  headRow.appendChild(el('th', 'num hide-sm col-td', 'TD'));
  headRow.appendChild(el('th', 'hide-sm', 'Last 6'));
  headRow.appendChild(el('th', 'hide-sm', 'Status'));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  const maxProj = Math.max(...list.map(proj), 1);
  list.forEach((player) => {
    const tr = el('tr');
    const playerTd = el('td');
    const pwrap = el('div', 'player');
    pwrap.appendChild(teamBadge(player.team, true));
    const info = el('div');
    info.style.minWidth = '0';
    info.appendChild(el('div', 'nm', player.name));
    info.appendChild(
      el(
        'div',
        'tm',
        `${teamAbbr(player.team)} · ${player.position}${player.depth_rank ? ` · #${player.depth_rank}` : ''}`
      )
    );
    if (player.matchup) {
      info.appendChild(
        el('div', 'tm only-sm', `${player.matchup} · ${timeLabel(player.gametime)}`.trim())
      );
    }
    pwrap.appendChild(info);
    playerTd.appendChild(pwrap);
    tr.appendChild(playerTd);

    const matchup = el('td', 'hide-sm');
    if (player.matchup) {
      matchup.appendChild(el('div', '', player.matchup));
      matchup.appendChild(el('div', 'muted-2', timeLabel(player.gametime)));
    } else {
      matchup.appendChild(el('span', 'muted-2', '—'));
    }
    tr.appendChild(matchup);

    const projTd = el('td', 'num');
    projTd.appendChild(el('div', 'proj', proj(player).toFixed(1)));
    const bar = el('div', 'proj-bar');
    const fill = el('i');
    fill.style.width = `${Math.max(4, (proj(player) / maxProj) * 100)}%`;
    bar.appendChild(fill);
    projTd.appendChild(bar);
    if (player.recent_fantasy_points && player.recent_fantasy_points.length) {
      projTd.appendChild(el('div', 'tm only-sm', `L3 ${recentAvg(player).toFixed(1)}`));
    }
    tr.appendChild(projTd);

    const cols = playerStatColumns(player);
    tr.appendChild(el('td', 'num hide-sm col-tgt', fmtStat(player.projected_targets, player.position === 'QB' || player.position === 'K')));
    tr.appendChild(el('td', 'num hide-sm col-rec', fmtStat(player.projected_receptions, player.position === 'QB' || player.position === 'K')));
    tr.appendChild(el('td', 'num hide-sm col-yds', fmtStat(cols.yds, false)));
    tr.appendChild(el('td', 'num hide-sm col-td', fmtStat(cols.td, false, 2)));

    const trendTd = el('td', 'hide-sm');
    trendTd.appendChild(sparkline(player.recent_fantasy_points));
    tr.appendChild(trendTd);

    const statusTd = el('td', 'hide-sm');
    statusTd.appendChild(injuryPill(player.injury_status));
    tr.appendChild(statusTd);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  // Trim columns that add no information for the selected position.
  if (state.playersPos === 'QB') {
    table.querySelectorAll('.col-tgt, .col-rec').forEach((node) => node.remove());
  } else if (state.playersPos === 'K') {
    table.querySelectorAll('.col-tgt, .col-rec, .col-yds, .col-td').forEach((node) => node.remove());
  }

  const scroll = el('div', 'table-scroll');
  scroll.appendChild(table);
  const box = el('div', 'table-wrap');
  box.appendChild(scroll);
  wrap.appendChild(box);
}

function fmtStat(value, skip, digits = 1) {
  if (skip) return '—';
  if (value === undefined || value === null) return '—';
  return Number(value).toFixed(digits);
}

/* ---------------- playoffs ---------------- */

function filteredPlayoffs() {
  const team = ($('#playoffs-team-filter')?.value || '').trim().toLowerCase();
  const sort = $('#playoffs-sort')?.value || 'sb';
  const list = state.playoffs.filter((t) => {
    const confOk = state.playoffsConf === 'all' || t.conference === state.playoffsConf;
    const teamOk = !team || (t.team || '').toLowerCase().includes(team);
    return confOk && teamOk;
  });
  const key = {
    sb: (t) => t.super_bowl_prob ?? 0,
    conf: (t) => t.conference_prob ?? 0,
    playoff: (t) => t.playoff_prob ?? 0,
    team: (t) => t.team || '',
  }[sort];
  if (sort === 'team') list.sort((a, b) => String(key(a)).localeCompare(String(key(b))));
  else list.sort((a, b) => key(b) - key(a));
  return list;
}

function renderPlayoffs() {
  const wrap = $('#playoffs-table-wrap');
  if (!wrap) return;
  clear(wrap);
  const list = filteredPlayoffs();

  const count = $('#playoffs-count');
  if (count) count.textContent = `${list.length} team${list.length === 1 ? '' : 's'}`;

  const stats = $('#playoffs-stats');
  if (stats) {
    clear(stats);
    const fav = [...state.playoffs].sort((a, b) => (b.super_bowl_prob ?? 0) - (a.super_bowl_prob ?? 0))[0];
    const locks = state.playoffs.filter((t) => (t.playoff_prob ?? 0) >= 0.99).length;
    const longshot = [...state.playoffs].sort((a, b) => (a.super_bowl_prob ?? 0) - (b.super_bowl_prob ?? 0))[0];
    stats.appendChild(statTile({ label: 'Teams', value: state.playoffs.length, sub: 'simulated' }));
    stats.appendChild(
      statTile({
        label: 'Title favourite',
        value: fav ? teamAbbr(fav.team) : '—',
        sub: fav ? `${pct(fav.super_bowl_prob, 1)} to win the Super Bowl` : '',
        cls: 'good',
      })
    );
    stats.appendChild(statTile({ label: 'Playoff locks', value: locks, sub: '≥ 99% to qualify' }));
    stats.appendChild(
      statTile({
        label: 'Longest shot',
        value: longshot ? teamAbbr(longshot.team) : '—',
        sub: longshot ? `${pct(longshot.super_bowl_prob, 1)} title chance` : '',
      })
    );
  }

  if (!list.length) {
    wrap.appendChild(
      emptyState({
        icon: '🏆',
        title: 'No teams match those filters',
        message: 'Switch conference or clear the team filter.',
        actionLabel: 'Clear filters',
        onAction: () => {
          const t = $('#playoffs-team-filter');
          if (t) t.value = '';
          setConference('all');
        },
      })
    );
    return;
  }

  const table = el('table', 'dtable po-table');
  const thead = el('thead');
  const headRow = el('tr');
  headRow.appendChild(el('th', '', '#'));
  headRow.appendChild(el('th', '', 'Team'));
  headRow.appendChild(el('th', 'hide-sm', 'Conf'));
  headRow.appendChild(el('th', '', 'Playoff'));
  headRow.appendChild(el('th', 'hide-sm', 'Conf. title'));
  headRow.appendChild(el('th', '', 'Super Bowl'));
  headRow.appendChild(el('th', 'num hide-sm', 'SB odds'));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  list.forEach((team, index) => {
    const tr = el('tr');
    tr.appendChild(el('td', 'muted-2', String(index + 1)));

    const teamTd = el('td');
    const identity = el('div', 'player');
    identity.appendChild(teamBadge(team.team, true));
    const info = el('div');
    info.appendChild(el('div', 'nm', teamAbbr(team.team)));
    info.appendChild(el('div', 'tm', teamFull(team.team)));
    identity.appendChild(info);
    teamTd.appendChild(identity);
    tr.appendChild(teamTd);

    tr.appendChild(el('td', 'muted hide-sm', team.conference || '—'));
    tapeCell(tr, team.playoff_prob, (team.playoff_prob ?? 0) >= 0.8 ? 'green' : '');
    tapeCell(tr, team.conference_prob, 'hide-sm');
    tapeCell(tr, team.super_bowl_prob, (team.super_bowl_prob ?? 0) >= 0.1 ? 'green' : 'dim');
    const titleOdds = (team.super_bowl_prob ?? 0) >= 0.01 ? american(team.super_bowl_prob) || '—' : '—';
    const oddsCellNode = el('td', 'num hide-sm muted', titleOdds);
    if (titleOdds === '—') oddsCellNode.title = 'Below 1% — odds not meaningful';
    tr.appendChild(oddsCellNode);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const scroll = el('div', 'table-scroll');
  scroll.appendChild(table);
  const box = el('div', 'table-wrap');
  box.appendChild(scroll);
  wrap.appendChild(box);
}

function tapeCell(tr, value, cls) {
  const td = el('td', cls.includes('hide-sm') ? 'hide-sm' : '');
  td.appendChild(meter(value ?? 0, cls.replace('hide-sm', '').trim()));
  tr.appendChild(td);
}

/* ---------------- market ---------------- */

function renderDistribution(games) {
  const container = $('#market-distribution');
  if (!container) return;
  clear(container);

  const bins = [
    { label: '≤−20', test: (e) => e <= -0.2 },
    { label: '−20', test: (e) => e > -0.2 && e <= -0.1 },
    { label: '−10', test: (e) => e > -0.1 && e <= -0.05 },
    { label: '−5', test: (e) => e > -0.05 && e < 0 },
    { label: '+5', test: (e) => e >= 0 && e < 0.05 },
    { label: '+10', test: (e) => e >= 0.05 && e < 0.1 },
    { label: '+20', test: (e) => e >= 0.1 && e < 0.2 },
    { label: '≥20', test: (e) => e >= 0.2 },
  ];
  const counts = bins.map(
    (b) => games.filter((g) => g.edge !== undefined && g.edge !== null && b.test(g.edge)).length
  );
  const max = Math.max(...counts, 1);

  bins.forEach((bin, index) => {
    const col = el('div', 'dist-col');
    const value = el('div', 'dist-label', String(counts[index]));
    value.style.color = counts[index] ? 'var(--muted)' : 'var(--muted-2)';
    col.appendChild(value);
    const bar = el('div', `dist-bar${index < 4 ? ' neg' : ''}`);
    bar.style.height = `${Math.max(3, (counts[index] / max) * 72)}px`;
    bar.title = `${counts[index]} game${counts[index] === 1 ? '' : 's'} in this bin`;
    if (!counts[index]) bar.style.opacity = '0.28';
    col.appendChild(bar);
    col.appendChild(el('div', 'dist-label', bin.label));
    container.appendChild(col);
  });
}

function marketSort(list) {
  const mode = $('#market-sort')?.value || 'ev';
  const sorted = [...list];
  if (mode === 'edge') sorted.sort((a, b) => Math.abs(b.edge ?? 0) - Math.abs(a.edge ?? 0));
  else if (mode === 'kickoff') sorted.sort((a, b) => String(a.commence_time || '').localeCompare(String(b.commence_time || '')));
  else sorted.sort((a, b) => (b.best_ev ?? -999) - (a.best_ev ?? -999));
  return sorted;
}

function bestMlForGame(game) {
  const evSide = game.best_ev_side;
  const home = game.best_home_ml;
  const away = game.best_away_ml;
  const preferHome = evSide ? evSide === 'home' : (game.model_home_win_prob ?? 0.5) >= 0.5;
  return preferHome
    ? { team: game.home_team, price: home?.price, book: home?.bookmaker }
    : { team: game.away_team, price: away?.price, book: away?.bookmaker };
}

function renderMarket() {
  const listEl = $('#market-list');
  if (!listEl || !state.market) return;
  clear(listEl);

  const summary = state.market.summary || {};
  const summaryEl = $('#market-summary');
  if (summaryEl) {
    clear(summaryEl);
    summaryEl.appendChild(
      statTile({ label: 'Games priced', value: summary.games_compared ?? 0, sub: 'model vs market' })
    );
    summaryEl.appendChild(
      statTile({
        label: 'Mean edge',
        value: summary.mean_edge !== undefined ? pct(summary.mean_edge, 1) : '—',
        sub: 'signed, home perspective',
      })
    );
    summaryEl.appendChild(
      statTile({
        label: 'Mean abs edge',
        value: summary.mean_abs_edge !== undefined ? pct(summary.mean_abs_edge, 1) : '—',
        sub: 'average disagreement',
      })
    );
    summaryEl.appendChild(
      statTile({
        label: 'Max abs edge',
        value: summary.max_abs_edge !== undefined ? pct(summary.max_abs_edge, 1) : '—',
        sub: 'largest single game',
        cls: 'warn',
      })
    );
  }

  const week = $('#market-week-filter')?.value || 'all';
  const team = ($('#market-team-filter')?.value || '').trim().toLowerCase();
  const filtered = (state.market.games || []).filter((g) => {
    const weekOk = week === 'all' || String(g.week) === week;
    const teamOk =
      !team ||
      (g.home_team || '').toLowerCase().includes(team) ||
      (g.away_team || '').toLowerCase().includes(team);
    return weekOk && teamOk;
  });
  const games = marketSort(filtered);

  const count = $('#market-count');
  if (count) count.textContent = `${games.length} game${games.length === 1 ? '' : 's'}`;

  renderDistribution(filtered);

  if (!games.length) {
    listEl.appendChild(
      emptyState({
        icon: '📈',
        title: 'No priced games match those filters',
        message: 'Odds are fetched once a week, so only the next couple of weeks are covered.',
        actionLabel: 'Clear filters',
        onAction: () => {
          const w = $('#market-week-filter');
          const t = $('#market-team-filter');
          if (w) w.value = 'all';
          if (t) t.value = '';
          renderMarket();
        },
      })
    );
    return;
  }

  games.forEach((game, index) => {
    listEl.appendChild(marketCard(game, index + 1));
  });
}

function marketCard(game, rank) {
  const card = el('article', 'market-card');
  const edgeValue = (game.edge ?? 0) * 100;
  const info = state.gamesById.get(game.game_id);

  const head = el('div', 'market-head');
  head.appendChild(el('span', `rank${rank <= 3 ? ' top' : ''}`, String(rank)));
  const names = el('div');
  names.style.flex = '1';
  names.style.minWidth = '0';
  names.appendChild(el('div', 'team-name', `${teamAbbr(game.away_team)} @ ${teamAbbr(game.home_team)}`));
  const kick = info ? kickoffLabel(info) : game.commence_time ? new Date(game.commence_time).toLocaleString() : '';
  names.appendChild(el('div', 'team-sub', `Week ${game.week} · ${kick}`));
  head.appendChild(names);
  head.appendChild(el('span', `tag ${edgeValue >= 0 ? 'edge' : 'neg'}`, `${signed(edgeValue, 1, '%')} ML`));
  card.appendChild(head);

  const duo = el('div', 'duo');
  duo.appendChild(box('Model', pct(game.model_home_win_prob, 1), ''));
  duo.appendChild(box('Market', pct(game.market_home_win_prob, 1), 'vig-free'));
  card.appendChild(duo);

  const ml = bestMlForGame(game);
  const grid = el('div', 'odds-grid');

  const mlCell = oddsCell(
    'Moneyline',
    ml.price ? americanFromDecimal(ml.price) : '—',
    `${teamAbbr(ml.team)}${ml.book ? ` · ${ml.book}` : ''}`,
    ml.price && ml.price < 2 ? 'neg-odds' : 'pos-odds'
  );
  grid.appendChild(mlCell);

  const spreadSide = (game.spread_edge ?? 0) >= 0 ? game.home_team : game.away_team;
  const spreadBest = (game.spread_edge ?? 0) >= 0 ? game.best_home_spread : game.best_away_spread;
  grid.appendChild(
    oddsCell(
      'Spread',
      spreadBest?.point !== undefined ? `${spreadBest.point > 0 ? '+' : ''}${spreadBest.point}` : '—',
      spreadBest?.price ? `${teamAbbr(spreadSide)} ${americanFromDecimal(spreadBest.price)} · ${spreadBest.bookmaker}` : '—',
      ''
    )
  );

  const totalSide = (game.total_edge ?? 0) >= 0 ? game.best_over : game.best_under;
  grid.appendChild(
    oddsCell(
      'Total',
      totalSide?.point !== undefined ? `${(game.total_edge ?? 0) >= 0 ? 'o' : 'u'}${totalSide.point}` : '—',
      totalSide?.price ? `${americanFromDecimal(totalSide.price)} · ${totalSide.bookmaker}` : '—',
      ''
    )
  );
  card.appendChild(grid);

  const footer = el('div', 'market-line');
  footer.appendChild(
    el(
      'span',
      'tag neutral',
      `Spread ${signed(game.spread_edge, 1, ' pts')} · Total ${signed(game.total_edge, 1, ' pts')}`
    )
  );
  if (game.best_ev !== null && game.best_ev !== undefined) {
    const evTag = el(
      'span',
      `tag ${game.best_ev >= 0 ? 'edge' : 'neg'}`,
      `Best EV ${formatEv(game.best_ev)}${
        game.best_ev_side
          ? ` · ${teamAbbr(game.best_ev_side === 'home' ? game.home_team : game.away_team)}`
          : ''
      }`
    );
    evTag.title = `Raw EV ${game.best_ev.toFixed(2)}% = model probability × best price − 1`;
    footer.appendChild(evTag);
  }
  card.appendChild(footer);

  return card;
}

function box(label, value, sub) {
  const node = el('div', 'box');
  node.appendChild(el('div', 'k', label));
  node.appendChild(el('div', 'v', value));
  if (sub) {
    const s = el('div', 'team-sub', sub);
    node.appendChild(s);
  }
  return node;
}

function oddsCell(label, value, sub, cls) {
  const node = el('div', 'odds-cell');
  node.appendChild(el('div', 'k', label));
  node.appendChild(el('div', `v ${cls}`.trim(), value));
  node.appendChild(el('div', 'b', sub || '\u00a0'));
  return node;
}

/* ---------------- filters / tabs ---------------- */

function setupFilters() {
  const rerenderGames = () => renderGames();
  $('#games-week-filter')?.addEventListener('change', rerenderGames);
  $('#games-team-filter')?.addEventListener('input', rerenderGames);
  $('#games-sort')?.addEventListener('change', rerenderGames);

  $('#players-team-filter')?.addEventListener('input', renderPlayers);
  $('#players-sort')?.addEventListener('change', renderPlayers);
  document.querySelectorAll('#players-pos-seg button').forEach((btn) => {
    btn.addEventListener('click', () => setPosition(btn.dataset.pos));
  });

  $('#playoffs-team-filter')?.addEventListener('input', renderPlayoffs);
  $('#playoffs-sort')?.addEventListener('change', renderPlayoffs);
  document.querySelectorAll('#playoffs-conf-seg button').forEach((btn) => {
    btn.addEventListener('click', () => setConference(btn.dataset.conf));
  });

  $('#market-week-filter')?.addEventListener('change', renderMarket);
  $('#market-team-filter')?.addEventListener('input', renderMarket);
  $('#market-sort')?.addEventListener('change', renderMarket);
}

function setPosition(pos) {
  state.playersPos = pos;
  document.querySelectorAll('#players-pos-seg button').forEach((btn) => {
    const active = btn.dataset.pos === pos;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
  renderPlayers();
}

function setConference(conf) {
  state.playoffsConf = conf;
  document.querySelectorAll('#playoffs-conf-seg button').forEach((btn) => {
    const active = btn.dataset.conf === conf;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
  renderPlayoffs();
}

function activateTab(name, focus = false) {
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');
  const bottom = document.querySelectorAll('.bottomnav button');
  tabs.forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  bottom.forEach((btn) => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-current', active ? 'page' : 'false');
  });
  panels.forEach((panel) => panel.classList.toggle('active', panel.id === name));
}

function setupTabs() {
  document.querySelectorAll('.tab, .bottomnav button').forEach((tab) => {
    tab.addEventListener('click', () => {
      const name = tab.dataset.tab;
      activateTab(name);
      history.replaceState(null, '', `#${name}`);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });

  const tablist = document.querySelector('.tabs');
  tablist?.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    const tabs = [...document.querySelectorAll('.tab')];
    const current = tabs.findIndex((t) => t.classList.contains('active'));
    const next = (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    activateTab(tabs[next].dataset.tab, true);
    history.replaceState(null, '', `#${tabs[next].dataset.tab}`);
    event.preventDefault();
  });

  const hash = window.location.hash.replace('#', '');
  if (['games', 'players', 'playoffs', 'market'].includes(hash)) {
    activateTab(hash);
    // The hash matches a section id, so the browser jumps to it on load.
    // Reset to the top so the sticky header does not cover the panel heading.
    requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: 'auto' }));
  }
}

/* ---------------- data loading ---------------- */

async function fetchJSON(name) {
  const res = await fetch(`${BASE}/${name}?t=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json();
}

async function fetchReport(name) {
  const res = await fetch(`${REPORTS_BASE}/${name}?t=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return res.json();
}

function populateSelect(select, values, label, format = (v) => `Week ${v}`) {
  if (!select || select.options.length > 1) return;
  values.forEach((value) => {
    const opt = document.createElement('option');
    opt.value = String(value);
    opt.textContent = format(value);
    select.appendChild(opt);
  });
  const fallback = values.length ? values[0] : null;
  if (fallback !== null) select.value = String(fallback);
}

function applyMeta(meta) {
  state.meta = meta || {};
  const metaData = state.meta;
  const season = metaData.season;
  const week =
    metaData.week ||
    (window.__filteredGames && window.__filteredGames[0]?.week) ||
    null;
  const seasonChip = $('#season-chip');
  if (seasonChip) {
    seasonChip.innerHTML = '';
    const strong = el('strong', '', season ? String(season) : '—');
    seasonChip.appendChild(strong);
    if (week) seasonChip.appendChild(document.createTextNode(` · Week ${week}`));
  }
  if (metaData.generated_at) {
    const chip = $('#freshness-text');
    if (chip) {
      chip.textContent = `Updated ${relativeTime(metaData.generated_at)}`;
      chip.title = new Date(metaData.generated_at).toLocaleString();
    }
    const footer = $('#footer-updated');
    if (footer) {
      footer.textContent = `Data updated ${new Date(metaData.generated_at).toLocaleString()} (${relativeTime(metaData.generated_at)})`;
    }
  }
}

function defaultWeek(games) {
  const weeks = [...new Set(games.map((g) => g.week).filter((w) => w !== undefined && w !== null))].sort(
    (a, b) => a - b
  );
  const today = new Date().toISOString().slice(0, 10);
  const upcoming = games.filter((g) => {
    const when = (g.gameday || g.commence_time || '').slice(0, 10);
    return when && when >= today && g.week !== undefined && g.week !== null;
  });
  if (upcoming.length) return Math.min(...upcoming.map((g) => g.week));
  return weeks.length ? weeks[weeks.length - 1] : null;
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
  setupFilters();

  try {
    const [games, players, playoffs, market, meta] = await Promise.all([
      loadGames(),
      fetchJSON('players_week.json'),
      fetchJSON('playoffs.json'),
      fetchReport('market_comparison.json').catch(() => null),
      fetchJSON('meta.json').catch(() => null),
    ]);

    state.games = games || [];
    state.players = players || [];
    state.playoffs = playoffs || [];
    state.market = market;
    state.marketById = new Map((market?.games || []).map((g) => [g.game_id, g]));
    state.gamesById = new Map(state.games.map((g) => [g.game_id, g]));
    applyMeta(meta);

    // Populate filters once, defaulting to the current week.
    const gameWeeks = [...new Set(state.games.map((g) => g.week))].sort((a, b) => a - b);
    const marketWeeks = [...new Set((state.market?.games || []).map((g) => g.week))].sort((a, b) => a - b);

    const gameWeekSelect = $('#games-week-filter');
    populateSelect(gameWeekSelect, gameWeeks, 'Week');
    const currentWeek = defaultWeek(state.games);
    if (gameWeekSelect && currentWeek !== null && gameWeeks.includes(currentWeek)) {
      gameWeekSelect.value = String(currentWeek);
    }

    const marketWeekSelect = $('#market-week-filter');
    populateSelect(marketWeekSelect, marketWeeks, 'Week');
    const currentMarketWeek = defaultWeek(state.market?.games || []);
    if (marketWeekSelect && currentMarketWeek !== null && marketWeeks.includes(currentMarketWeek)) {
      marketWeekSelect.value = String(currentMarketWeek);
    }

    if (!state.meta.week && currentWeek !== null) state.meta.week = currentWeek;

    // Default position filter to QB when available.
    const positions = new Set(state.players.map((p) => p.position));
    setPosition(positions.has('QB') ? 'QB' : 'all');

    renderGames();
    renderPlayers();
    renderPlayoffs();
    if (market) renderMarket();

    // Refresh header metadata now that the week and game list are known.
    applyMeta(state.meta);
  } catch (err) {
    console.error(err);
    const container = $('#games-list');
    if (container) {
      container.innerHTML = '';
      container.appendChild(
        emptyState({
          icon: '⚠️',
          title: 'Could not load predictions',
          message: 'The data pipeline may still be running. Refresh in a few minutes.',
          actionLabel: 'Retry',
          onAction: () => window.location.reload(),
        })
      );
    }
  }
}

document.addEventListener('DOMContentLoaded', loadAll);
