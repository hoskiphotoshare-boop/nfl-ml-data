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
  futures: null,
  explanationById: new Map(),
  scoreExplanationById: new Map(),
  futuresByTeam: new Map(),
  meta: null,
  marketById: new Map(),
  gamesById: new Map(),
  gamesView: 'schedule',
  playersPos: 'QB',
  playersSort: { key: 'proj', dir: 'desc' },
  historyWindow: 3,
  selectedPlayerId: null,
  playoffsConf: 'all',
  playoffsSort: { key: 'super_bowl', dir: 'desc' },
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
  // Keep tiny probabilities readable instead of rounding them to 0%.
  const places = digits === 0 && value > 0 && value < 0.05 ? 1 : digits;
  return `${(value * 100).toFixed(places)}%`;
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

function decSuffix(prob) {
  if (prob === undefined || prob === null || prob < 0.005 || prob >= 1) return '';
  const dec = decimalOdds(prob);
  return dec ? ` (${dec})` : '';
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
  wrap.title = 'Last five results: W win, L loss, T tie (oldest → newest).';
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

function statTile({ label, value, sub, cls = '', title = '' }) {
  const tile = el('div', `stat ${cls}`.trim());
  if (title) tile.title = title;
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

function recentStatValues(player, key) {
  const stats = player.recent_stats;
  if (stats && Array.isArray(stats[key])) return stats[key].slice(-6);
  if (key === 'fantasy_points') return (player.recent_fantasy_points || []).slice(-6);
  return [];
}

function historyRow(player, key, digits = 1) {
  const values = recentStatValues(player, key);
  const row = el('div', 'hist');
  if (!values.length) {
    row.appendChild(el('span', 'muted-2', '—'));
    return row;
  }
  const shown = values.slice(-(state.historyWindow || 3));
  row.title = `Last ${shown.length} games, oldest to newest`;
  shown.forEach((value, index) => {
    const text = Number(value)
      .toFixed(digits)
      .replace(/\.0+$/, '');
    row.appendChild(el('span', index === shown.length - 1 ? 'last' : '', text));
  });
  return row;
}

function meter(value, cls = '') {
  const wrap = el('div', `meter ${cls}`.trim());
  const track = el('div', 'track');
  const fill = el('i');
  fill.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
  track.appendChild(fill);
  wrap.appendChild(track);
  const val = el('span', 'val', pct(value));
  const dec = decimalOdds(value);
  if (dec && value >= 0.005) val.appendChild(el('span', 'dec', ` (${dec})`));
  wrap.appendChild(val);
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

/* ---------------- table sorting ---------------- */

function compareSortValues(a, b, dir) {
  const aMissing = a === null || a === undefined || Number.isNaN(a);
  const bMissing = b === null || b === undefined || Number.isNaN(b);
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  let result;
  if (typeof a === 'string' || typeof b === 'string') {
    result = String(a).localeCompare(String(b), undefined, { numeric: true });
  } else {
    result = a - b;
  }
  return dir === 'asc' ? result : -result;
}

function sortRecords(list, getValue, sort) {
  return [...list].sort((a, b) =>
    compareSortValues(getValue(a), getValue(b), sort.dir)
  );
}

function nextSort(sort, key, ascendingKeys = []) {
  if (sort.key === key) {
    sort.dir = sort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    sort.key = key;
    sort.dir = ascendingKeys.includes(key) ? 'asc' : 'desc';
  }
  return sort;
}

function sortHeader(label, key, sort, onSort, className = '', title = '') {
  const th = el('th', className);
  const active = sort.key === key;
  const button = el('button');
  button.type = 'button';
  if (active) button.classList.add('sorted');
  button.appendChild(document.createTextNode(label));
  button.appendChild(
    el('span', 'caret', active ? (sort.dir === 'asc' ? '▲' : '▼') : '⇅')
  );
  button.title = title
    ? `${title} Click to sort.`
    : `Sort by ${label.toLowerCase()}.`;
  button.setAttribute(
    'aria-label',
    `${label}: sort ${active ? (sort.dir === 'asc' ? 'descending' : 'ascending') : 'ascending'}`
  );
  button.addEventListener('click', () => onSort(key));
  th.appendChild(button);
  th.setAttribute('aria-sort', active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none');
  return th;
}

function renderSortState(sort, selector, mapping) {
  const select = $(selector);
  if (!select) return;
  const value = mapping[sort.key];
  if (value && [...select.options].some((option) => option.value === value)) {
    select.value = value;
  }
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

const MARKET_LABEL = { ml: 'Moneyline', spread: 'Spread', total: 'Total' };
const MARKET_SHORT = { ml: 'ML', spread: 'SPR', total: 'TOT' };

const FEATURE_HELP = {
  avg_points: 'Average points scored per game before this week.',
  avg_allowed: 'Average points allowed per game before this week.',
  off_epa: 'Offensive EPA per play (expected points added). Higher is better.',
  def_epa: 'Defensive EPA per play allowed. Lower is better.',
  success_rate: 'Share of plays that gain positive expected points.',
  pass_epa: 'Passing EPA per play.',
  rush_epa: 'Rushing EPA per play.',
  passing_epa: 'Passing EPA accumulated before this week.',
  rushing_epa: 'Rushing EPA accumulated before this week.',
  rest_days: 'Days of rest since the previous game.',
  rest_diff: 'Rest-day difference, home minus away.',
  short_week: '1 if the team is on a short week (e.g. Thursday game).',
  div_game: '1 if both teams are in the same division.',
  prior_games: 'Games already played this season.',
  win_pct: 'Win percentage before this week.',
  diff_pg: 'Point differential per game before this week.',
  win_pct_diff: 'Win-percentage difference, home minus away.',
  diff_pg_diff: 'Point-differential-per-game difference, home minus away.',
  qb_prior_epa: 'Trailing EPA per attempt for the starting quarterback.',
  qb_starts: 'Career starts by the listed starting quarterback.',
  qb_changed: '1 if the starting quarterback changed from last week.',
  qb_missing: '1 if the usual starting quarterback is unavailable.',
  inj_impact:
    'Weighted injury impact: report status severity × position importance, summed over players on the report.',
  inj_impact_diff: 'Injury-impact difference, home minus away.',
  inj_missing: '1 if no injury report was available.',
};

function driverHelp(driver) {
  const base = String(driver.feature || '').replace(/^(home|away)_/, '');
  return FEATURE_HELP[base] || '';
}

const GLOSSARY = {
  common: [
    ['pp', 'Percentage points. The gap between two probabilities: 60% vs 55% is +5 pp.'],
    [
      'Decimal odds',
      'Total return per 1 unit staked, including the stake. 2.50 means a 1-unit bet returns 2.50.',
    ],
    ['Fair odds', 'The model’s own decimal price: 1 ÷ probability.'],
    ['#', 'A rank or depth-chart position, depending on the column.'],
    ['ET', 'Eastern Time. Kickoff times are shown in ET.'],
    ['vs / @', '“vs” means the team is at home; “@” means it is on the road.'],
    ['pts', 'Points — used for projected margins, totals and lean distance.'],
    ['°F / mph', 'Degrees Fahrenheit and miles per hour for kickoff weather.'],
    ['—', 'No data available for that item.'],
    [
      '+ / −',
      'For odds or probabilities, + means the model rates that side above the market; − means below.',
    ],
    [
      'Team codes',
      'Official NFL abbreviations: KC = Chiefs, LA = Rams, LAC = Chargers, LV = Raiders, SF = 49ers, TB = Buccaneers, WAS = Commanders.',
    ],
  ],
  games: [
    ['ML (moneyline)', 'Pick the outright winner. No point spread applies.'],
    [
      'SPR (spread)',
      'Point handicap. “KC -3” means Kansas City must win by more than 3 for that side to cash.',
    ],
    ['TOT (total)', 'Combined points scored by both teams, with an over/under line.'],
    ['O/U', 'Over or under the posted total.'],
    ['Pick’em', 'A spread at (or very near) zero — no team is favoured.'],
    [
      'Form dots',
      'The last five results: W win, L loss, T tie, newest on the right.',
    ],
    [
      'Mkt',
      'Best available market decimal price for that team’s moneyline.',
    ],
    [
      'Rank (#)',
      'Position in the current sort. In the edges view, #1 has the largest best edge.',
    ],
    [
      'Win bar',
      'Model win probability split between the two teams; the wider side is more likely to win.',
    ],
    ['Weather', 'Forecast temperature (°F) and wind (mph) at kickoff for outdoor games.'],
    [
      'Model spread',
      'The model’s projected point margin. Positive favours the home team; negative favours the away team.',
    ],
    ['Model total', 'The model’s projected combined points for both teams.'],
    [
      'Lean vs 50%',
      'How far the model’s win probability is from a coin flip, in percentage points. 0 = pick’em; 30 pts = an 80% vs 20% split.',
    ],
    [
      'Edge',
      'Model probability minus the vig-free market probability, in percentage points. Positive means the model rates that side higher than the market.',
    ],
    [
      'Vig / overround',
      'The bookmaker margin built into prices. We remove it before comparing model and market, so probabilities add to 100%.',
    ],
    ['Priced game', 'A game with at least one market line from a sportsbook.'],
    [
      'EV',
      'Expected value per 1 unit staked at the best available price: model probability × price − 1.',
    ],
    ['Best price', 'The highest decimal price available across the books we track.'],
    [
      'Div',
      'Division game. These average slightly fewer points and a smaller home-field advantage, and the models use that flag.',
    ],
    [
      'EPA',
      'Expected points added. A play-quality measure: positive EPA is better for the offense, negative for the defense.',
    ],
    [
      'NGS',
      'Next Gen Stats: tracking-based metrics such as separation, cushion and rushing efficiency.',
    ],
    [
      'SHAP (driver)',
      'How much a feature moved this game’s prediction, in log-odds for the winner model or points for spread/total. Positive pushes the pick up, negative pulls it down.',
    ],
    [
      'Rest days',
      'Days since the team’s previous game. Short weeks (e.g. Thursday games) are flagged separately.',
    ],
  ],
  players: [
    [
      'Proj',
      'Projected fantasy points for this week under standard scoring (4-point passing TD, no PPR).',
    ],
    [
      'Range',
      'The 80% projection interval: floor (10th percentile) to ceiling (90th percentile). Roughly 4 in 5 outcomes should land inside it.',
    ],
    [
      'Last 3 / Last 6',
      'The small numbers under each stat show the last three games by default, oldest to newest. The bold value is the most recent game; “+ Last 6” expands every stat to the last six.',
    ],
    ['Tgt', 'Targets: passes thrown in the player’s direction.'],
    ['Rec', 'Receptions.'],
    ['INT', 'Interceptions thrown by the quarterback.'],
    ['Rush yds', 'Rushing yards.'],
    ['Rush TD', 'Rushing touchdowns.'],
    ['FG', 'Field goals made.'],
    ['FG att', 'Field-goal attempts.'],
    ['XP', 'Extra points made.'],
    [
      'Proj bar',
      'Relative size of the projection against the highest projection in the current filter.',
    ],
    ['Recent form', 'Sorts by average fantasy points over the last three games.'],
    ['Depth chart', 'Sorts by listed depth-chart rank, #1 first.'],
    ['Yds', 'Yards. For quarterbacks this is passing yards; for skill players it is receiving/rushing yards.'],
    ['TD', 'Touchdowns. For quarterbacks this is passing touchdowns.'],
    ['QB / RB / WR / TE / K', 'Quarterback, running back, wide receiver, tight end, kicker.'],
    ['# (depth)', 'Depth-chart rank at the player’s position. #1 is the starter.'],
    ['Matchup', 'Opponent and kickoff time for this week’s game.'],
    ['Status', 'Injury-report designation for this week.'],
    ['Q / D / O / IR', 'Questionable · Doubtful · Out · Injured reserve.'],
    ['Standard scoring', '4 points per passing TD, 6 per rushing/receiving TD, no points per reception.'],
  ],
  playoffs: [
    ['Playoff', 'Probability of reaching the postseason in the simulated seasons.'],
    ['Conf. title', 'Probability of winning the AFC or NFC championship.'],
    ['Super Bowl', 'Probability of winning the Super Bowl.'],
    [
      'Market',
      'Vig-free market Super Bowl probability, normalized so all teams add to 100%, with the best available decimal price.',
    ],
    [
      'Edge',
      'Model probability minus vig-free market probability, in percentage points. Positive means the model is higher than the market.',
    ],
    ['AFC / NFC', 'The two conferences.'],
    [
      'Overround',
      'The bookmaker margin: implied probabilities add up to more than 100%. We normalize them to remove it.',
    ],
    [
      'Probability meter',
      'The bar and percentage show the simulated probability; the decimal in parentheses is fair odds.',
    ],
    ['Lock', 'A team with at least a 99% simulated chance to reach the playoffs.'],
    [
      'Monte Carlo',
      'The remaining schedule and bracket are played out thousands of times; each outcome is a simulated season.',
    ],
    [
      'SRS',
      'Simple rating system: an opponent-adjusted point-differential rating. The simulator blends it 50/50 with the match model.',
    ],
    [
      'Simulation noise',
      'Each simulated season gets a random team-strength adjustment, so injuries, form and coaching uncertainty are reflected.',
    ],
  ],
};

function renderLegend(id, groups) {
  const node = document.getElementById(id);
  if (!node) return;
  clear(node);
  const entries = [...GLOSSARY.common, ...groups.flatMap((group) => GLOSSARY[group] || [])];
  entries.forEach(([term, definition]) => {
    const item = el('div', 'legend-item');
    item.appendChild(el('dt', 'legend-term', term));
    item.appendChild(el('dd', 'legend-def', definition));
    node.appendChild(item);
  });
}

function bestMarket(market) {
  if (!market || !market.markets) return null;
  let best = null;
  Object.entries(market.markets).forEach(([key, entry]) => {
    if (!entry || entry.edge === undefined || entry.edge === null) return;
    if (!best || entry.edge > best.market.edge) best = { key, market: entry };
  });
  return best;
}

function marketPick(entry) {
  if (!entry) return '—';
  if (entry.side === 'over' || entry.side === 'under') {
    return `${entry.side === 'over' ? 'Over' : 'Under'} ${entry.line}`;
  }
  const line =
    entry.line !== null && entry.line !== undefined
      ? ` ${entry.line > 0 ? '+' : ''}${entry.line}`
      : '';
  return `${teamAbbr(entry.team)}${line}`;
}

function edgePoints(edge) {
  if (edge === undefined || edge === null || Number.isNaN(edge)) return '—';
  return `${edge >= 0 ? '+' : ''}${(edge * 100).toFixed(1)} pp`;
}

function shortMarketTag(entry) {
  if (entry.side === 'over' || entry.side === 'under') {
    return `${entry.side === 'over' ? 'O' : 'U'}${entry.line}`;
  }
  // The line lives in the edge grid; keep the header chip short.
  return teamAbbr(entry.team);
}

function edgeGrid(market, best) {
  const grid = el('div', 'edge-grid');
  ['ml', 'spread', 'total'].forEach((key) => {
    const entry = market.markets ? market.markets[key] : null;
    if (!entry) {
      const empty = el('div', 'edge-cell empty');
      empty.appendChild(el('div', 'k', MARKET_LABEL[key]));
      empty.appendChild(el('div', 'pick', '—'));
      grid.appendChild(empty);
      return;
    }
    const isBest = best && best.key === key;
    const cell = el('div', `edge-cell${isBest ? ' best' : ''}`);
    cell.appendChild(el('div', 'k', MARKET_LABEL[key]));
    cell.appendChild(el('div', 'pick', marketPick(entry)));
    const valueRow = el('div', 'edge-val-row');
    valueRow.appendChild(
      el(
        'span',
        `edge-val ${entry.edge >= 0.02 ? 'pos' : entry.edge > 0 ? 'thin' : 'muted'}`,
        edgePoints(entry.edge)
      )
    );
    if (entry.price) {
      valueRow.appendChild(el('span', 'price', entry.price.toFixed(2)));
    }
    cell.appendChild(valueRow);
    cell.title = `${MARKET_LABEL[key]} · model ${pct(entry.model_prob, 1)} vs vig-free market ${pct(
      entry.market_prob,
      1
    )}. Edge ${edgePoints(entry.edge)} = model minus market, in percentage points (pp).${
      entry.price ? `\nBest decimal price ${entry.price.toFixed(2)} at ${entry.bookmaker}.` : ''
    }${
      entry.ev !== null && entry.ev !== undefined
        ? `\nExpected value ${formatEv(entry.ev)} per 1 unit staked.`
        : ''
    }`;
    grid.appendChild(cell);
  });
  return grid;
}

function gameCard(game, rank) {
  const card = el('article', 'game-card');
  const market = state.marketById.get(game.game_id);
  const best = bestMarket(market);
  const { home, away } = probsFor(game);
  const homeFav = home !== null && home >= 0.5;

  const top = el('div', 'game-top');
  if (rank) top.appendChild(el('span', `rank${rank <= 3 ? ' top' : ''}`, String(rank)));
  top.appendChild(el('span', 'when', kickoffLabel(game)));
  const venue = game.stadium || (game.location && game.location !== 'Home' ? game.location : '');
  if (venue) {
    const v = el('span', 'venue', `· ${venue}`);
    v.title = [game.roof, game.surface].filter(Boolean).join(' · ');
    top.appendChild(v);
  }
  const weather = [
    game.temp !== null && game.temp !== undefined ? `${game.temp}°F` : null,
    game.wind !== null && game.wind !== undefined ? `${game.wind} mph` : null,
  ]
    .filter(Boolean)
    .join(' · ');
  if (weather) {
    const w = el('span', 'venue', `· ${weather}`);
    w.title = 'Forecast weather';
    top.appendChild(w);
  }
  top.appendChild(el('span', 'spacer'));
  if (game.div_game) {
    const divTag = el('span', 'tag div', 'Div');
    divTag.title =
      'Division game — lower-scoring on average and a smaller home-field advantage.';
    top.appendChild(divTag);
  }
  if (best) {
    const bestTag = el(
      'span',
      `tag edge-tag ${market.outlier ? 'amber' : best.market.edge >= 0.02 ? 'edge' : 'neutral'}`,
      `${MARKET_SHORT[best.key]} ${shortMarketTag(best.market)} ${edgePoints(best.market.edge)}`
    );
    bestTag.title = `${MARKET_LABEL[best.key]} · model ${pct(
      best.market.model_prob,
      1
    )} vs vig-free market ${pct(best.market.market_prob, 1)}. Edge = model − market, in percentage points.`;
    top.appendChild(bestTag);
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
    const bestMl = row.team === game.home_team ? market?.best_home_ml : market?.best_away_ml;
    if (bestMl && bestMl.price) {
      const oddsLine = el('div', 'team-sub', `mkt ${bestMl.price.toFixed(2)}`);
      oddsLine.title = `Best available market decimal price for ${teamAbbr(
        row.team
      )} moneyline (${bestMl.bookmaker || 'market'}).`;
      names.appendChild(oddsLine);
    }
    line.appendChild(names);
    const probNode = el('span', 'prob');
    probNode.appendChild(document.createTextNode(pct(row.prob)));
    const probDec = decimalOdds(row.prob);
    if (probDec && row.prob >= 0.005) probNode.appendChild(el('span', 'dec', ` (${probDec})`));
    line.appendChild(probNode);
    card.appendChild(line);
  });

  if (home !== null) {
    const bar = el('div', 'winbar');
    bar.setAttribute(
      'role',
      'img'
    );
    bar.setAttribute(
      'aria-label',
      `${teamAbbr(game.away_team)} ${pct(away)} · ${teamAbbr(game.home_team)} ${pct(home)}`
    );
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
    {
      label: 'Lean vs 50%',
      value: home === null ? '—' : `${(Math.abs(home - 0.5) * 100).toFixed(1)} pts`,
      title:
        'How far the model’s win probability is from a coin flip, in percentage points. 0 = pick’em; 30 pts = a 80% vs 20% split. Bigger lean = more confident.',
    },
    {
      label: 'Model spread',
      value: spreadLine(game),
      title:
        'The model’s projected point margin. A positive spread means the home team is favoured; negative means the home team is the underdog.',
    },
    {
      label: 'Model total',
      value: game.total !== null && game.total !== undefined ? game.total.toFixed(1) : '—',
      title: 'The model’s projected combined points for both teams.',
    },
  ].forEach(({ label, value, title }) => {
    const cell = el('div', 'meta-cell');
    if (title) cell.title = title;
    cell.appendChild(el('div', 'k', label));
    cell.appendChild(el('div', 'v', value));
    meta.appendChild(cell);
  });
  card.appendChild(meta);

  if (market && market.markets) card.appendChild(edgeGrid(market, best));
  if (market && market.outlier) {
    const marketKey = market.outlier_market;
    let drivers = [];
    let unit = '';
    // The displayed spread is derived from the win probability, so the match
    // model (not the margin regressor) explains spread disagreements. Only the
    // total uses the total regressor.
    if (marketKey === 'total') {
      const scoreGame = state.scoreExplanationById?.get(game.game_id);
      drivers = scoreGame?.total?.drivers?.slice(0, 3) || [];
      if (drivers.length) unit = ' pts';
    }
    if (!drivers.length) {
      drivers = state.explanationById?.get(game.game_id)?.drivers?.slice(0, 3) || [];
      unit = '';
    }
    if (drivers.length) {
      const block = el('div', 'driver-block');
      block.appendChild(el('div', 'k', 'Why the model differs'));
      const tags = el('div', 'driver-tags');
      drivers.forEach((driver) => {
        const tag = el(
          'span',
          `tag ${driver.shap >= 0 ? 'edge' : 'neg'}`,
          `${driver.label} ${driver.shap >= 0 ? '+' : ''}${driver.shap.toFixed(2)}${unit}`
        );
        tag.title = `${driver.label}\nSHAP ${driver.shap >= 0 ? '+' : ''}${driver.shap.toFixed(
          2
        )} ${unit ? 'points' : 'log-odds'} — ${
          driver.shap >= 0 ? 'pushes the projection up' : 'pulls the projection down'
        }.\nFeature value ${driver.value}.${
          driverHelp(driver) ? `\n${driverHelp(driver)}` : ''
        }`;
        tags.appendChild(tag);
      });
      block.appendChild(tags);
      card.appendChild(block);
    }
  }

  const buttons = el('div', 'pick-buttons');
  [
    { side: 'away', team: game.away_team },
    { side: 'home', team: game.home_team },
  ].forEach(({ side, team }) => {
    const btn = el('button', 'pick-btn', `Pick ${teamAbbr(team)}`);
    btn.type = 'button';
    btn.title = `${teamAbbr(team)} to win outright`;
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
          `Week ${game.week} · picked ${side === 'home' ? teamAbbr(game.home_team) : teamAbbr(game.away_team)} · model ${pct(prob)}${decSuffix(prob)}`
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

function gameEdge(game) {
  return bestMarket(state.marketById.get(game.game_id))?.market.edge ?? -1;
}

function showFirstOddsWeek() {
  const weeks = [...new Set((state.market?.games || []).map((g) => g.week))]
    .filter((w) => w !== undefined && w !== null)
    .sort((a, b) => a - b);
  const weekSelect = $('#games-week-filter');
  const teamInput = $('#games-team-filter');
  if (teamInput) teamInput.value = '';
  if (weekSelect && weeks.length) weekSelect.value = String(weeks[0]);
  renderGames();
}

function filteredGames() {
  const week = $('#games-week-filter')?.value || 'all';
  const team = ($('#games-team-filter')?.value || '').trim().toLowerCase();
  const sort = $('#games-sort')?.value || 'kickoff';

  let list = state.games.filter((g) => {
    const weekOk = week === 'all' || String(g.week) === week;
    const teamOk =
      !team ||
      (g.home_team || '').toLowerCase().includes(team) ||
      (g.away_team || '').toLowerCase().includes(team);
    return weekOk && teamOk;
  });

  if (state.gamesView === 'edges') {
    list = list.filter((g) => bestMarket(state.marketById.get(g.game_id)));
    let compare;
    if (sort === 'ev') {
      compare = (a, b) =>
        (state.marketById.get(b.game_id)?.best_market_ev ?? -999) -
        (state.marketById.get(a.game_id)?.best_market_ev ?? -999);
    } else if (sort === 'kickoff') {
      compare = (a, b) =>
        `${a.gameday || ''}${a.gametime || ''}`.localeCompare(`${b.gameday || ''}${b.gametime || ''}`);
    } else {
      compare = (a, b) => gameEdge(b) - gameEdge(a);
    }
    list.sort(compare);
    return list;
  }

  if (sort === 'confidence') {
    list.sort(
      (a, b) =>
        Math.abs((b.home_win_prob ?? 0.5) - 0.5) - Math.abs((a.home_win_prob ?? 0.5) - 0.5)
    );
  } else if (sort === 'close') {
    list.sort(
      (a, b) =>
        Math.abs((a.home_win_prob ?? 0.5) - 0.5) - Math.abs((b.home_win_prob ?? 0.5) - 0.5)
    );
  } else {
    list.sort((a, b) => `${a.gameday || ''}${a.gametime || ''}`.localeCompare(`${b.gameday || ''}${b.gametime || ''}`));
  }
  return list;
}

function renderGameStats(list) {
  const row = $('#games-stats');
  if (!row) return;
  clear(row);

  const ranked = list
    .map((game) => ({ game, market: state.marketById.get(game.game_id) }))
    .filter((item) => item.market)
    .map((item) => ({ ...item, best: bestMarket(item.market) }))
    .filter((item) => item.best);
  const top = [...ranked].sort((a, b) => b.best.market.edge - a.best.market.edge)[0];

  if (state.gamesView === 'edges') {
    const edges = ranked.map((item) => item.best.market.edge);
    const sorted = [...edges].sort((a, b) => a - b);
    const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
    row.appendChild(
      statTile({
        label: 'Priced games',
        value: list.length,
        sub: 'with a market line',
        title: 'Games in this filter that have at least one market line from a sportsbook.',
      })
    );
    row.appendChild(
      statTile({
        label: 'Best edge',
        value: top ? edgePoints(top.best.market.edge) : '—',
        sub: top ? `${marketPick(top.best.market)} ${MARKET_LABEL[top.best.key].toLowerCase()}` : '',
        cls: top ? 'good' : '',
        title:
          'Largest gap between the model probability and the vig-free market probability across all priced games.',
      })
    );
    row.appendChild(
      statTile({
        label: 'Strong edges',
        value: `${edges.filter((e) => e >= 0.05).length}/${edges.length}`,
        sub: '5+ probability points',
        cls: edges.some((e) => e >= 0.05) ? 'good' : '',
        title: 'Priced games whose best edge is at least 5 percentage points.',
      })
    );
    row.appendChild(
      statTile({
        label: 'Median edge',
        value: median !== null ? edgePoints(median) : '—',
        sub: 'best market per game',
        title: 'Median of each game’s best edge, in percentage points.',
      })
    );
    return;
  }

  const pick = list
    .map((g) => ({ game: g, edge: Math.abs((g.home_win_prob ?? 0.5) - 0.5) }))
    .sort((a, b) => b.edge - a.edge)[0];
  const totals = list.map((g) => g.total).filter((t) => t !== null && t !== undefined);

  row.appendChild(
    statTile({
      label: 'Games',
      value: list.length,
      sub: 'in current filter',
      title: 'Games in the current week and team filter.',
    })
  );
  row.appendChild(
    statTile({
      label: 'Strongest lean',
      value:
        pick && pick.game.home_win_prob !== null && pick.game.home_win_prob !== undefined
          ? `${teamAbbr(pick.game.home_win_prob >= 0.5 ? pick.game.home_team : pick.game.away_team)} ${pct(
              pick.game.home_win_prob >= 0.5 ? pick.game.home_win_prob : 1 - pick.game.home_win_prob
            )}${decSuffix(
              pick.game.home_win_prob >= 0.5 ? pick.game.home_win_prob : 1 - pick.game.home_win_prob
            )}`
          : '—',
      sub: pick ? `${pick.game.away_team} @ ${pick.game.home_team}` : '',
      cls: 'good',
      title:
        'The game where the model’s win probability is furthest from a coin flip (50%), with fair decimal odds.',
    })
  );
  row.appendChild(
    statTile({
      label: 'Average total',
      value: totals.length ? (totals.reduce((a, b) => a + b, 0) / totals.length).toFixed(1) : '—',
      sub: `${totals.length} games with a line`,
      title: 'Average model-projected combined points across games with a total.',
    })
  );
  row.appendChild(
    statTile({
      label: 'Best edge',
      value: top ? edgePoints(top.best.market.edge) : '—',
      sub: top
        ? `${marketPick(top.best.market)} · ${MARKET_LABEL[top.best.key].toLowerCase()}`
        : 'odds load daily',
      cls: top ? 'good' : '',
      title:
        'Largest gap between model probability and vig-free market probability; positive means the model rates that side higher than the market.',
    })
  );
}

function renderOddsFreshness() {
  const node = $('#odds-freshness');
  if (!node) return;
  const market = state.market;
  if (!market || !market.odds_fetched_at) {
    node.textContent = '';
    return;
  }
  const events = market.odds_events ?? (market.games || []).length;
  node.textContent = `Odds updated ${relativeTime(market.odds_fetched_at)} · ${events} games priced`;
  node.title = new Date(market.odds_fetched_at).toLocaleString();
}

function renderModelFreshness() {
  const meta = state.meta || {};
  const stamps = [
    ['#games-updated', 'Predictions updated', meta.games ?? state.games.length, 'games'],
    ['#players-freshness', 'Projections updated', meta.players ?? state.players.length, 'players'],
  ];
  stamps.forEach(([selector, label, count, unit]) => {
    const node = $(selector);
    if (!node) return;
    if (!meta.generated_at) {
      node.textContent = '';
      return;
    }
    const week = meta.week ? ` · Week ${meta.week}` : '';
    node.textContent = `${label} ${relativeTime(meta.generated_at)}${week} · ${count} ${unit}`;
    node.title = new Date(meta.generated_at).toLocaleString();
  });
}

function renderGames() {
  const container = $('#games-list');
  if (!container) return;
  const list = filteredGames();
  const view = state.gamesView;

  renderGameStats(list);
  renderPicks();
  renderOddsFreshness();

  const extras = $('#edges-extras');
  if (extras) extras.style.display = view === 'edges' ? 'block' : 'none';
  if (view === 'edges') renderDistribution(list);

  const count = $('#games-count');
  if (count) {
    count.textContent = `${list.length} ${view === 'edges' ? 'priced game' : 'game'}${
      list.length === 1 ? '' : 's'
    }`;
  }

  const title = $('#games-title');
  const week = $('#games-week-filter')?.value || 'all';
  if (title) {
    if (view === 'edges') {
      title.textContent =
        week === 'all' ? 'Model edges · all priced games' : `Model edges · Week ${week}`;
    } else {
      const season = state.meta?.season || state.games?.[0]?.season;
      title.textContent =
        week === 'all' ? 'All remaining games' : `Week ${week}${season ? ` · ${season}` : ''}`;
    }
  }

  clear(container);
  if (!list.length) {
    container.appendChild(
      view === 'edges'
        ? emptyState({
            icon: '📉',
            title: 'No odds for this week yet',
            message:
              'Books post lines through the week. Try a week that already has prices, or switch back to the schedule.',
            actionLabel: 'Show priced weeks',
            onAction: showFirstOddsWeek,
          })
        : emptyState({
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
  if (view === 'edges' || sort !== 'kickoff') {
    list.forEach((game, index) =>
      container.appendChild(gameCard(game, view === 'edges' ? index + 1 : null))
    );
    return;
  }

  // Date grouping for the chronological schedule view.
  let currentDate = null;
  list.forEach((game) => {
    const label = dateLabel(game.gameday);
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
    container.appendChild(gameCard(game));
  });
}

const STAT_HISTORY_TIP =
  'Small numbers below are recent games, oldest to newest; use + Last 6 for all six.';

const SKILL_COLUMNS = [
  { key: 'targets', valueKey: 'projected_targets', label: 'Tgt', digits: 1, hist: 0, tip: 'Projected targets.' },
  { key: 'receptions', valueKey: 'projected_receptions', label: 'Rec', digits: 1, hist: 0, tip: 'Projected receptions.' },
  { key: 'yards', valueKey: 'projected_yards', label: 'Yds', digits: 1, hist: 0, tip: 'Projected yards (passing for quarterbacks).' },
  { key: 'touchdowns', valueKey: 'projected_tds', label: 'TD', digits: 2, hist: 0, tip: 'Projected touchdowns (passing for quarterbacks).' },
];

const QB_COLUMNS = [
  { key: 'passing_yards', valueKey: 'projected_pass_yards', label: 'Pass yds', digits: 1, hist: 0, tip: 'Projected passing yards.' },
  { key: 'passing_tds', valueKey: 'projected_pass_tds', label: 'Pass TD', digits: 2, hist: 0, tip: 'Projected passing touchdowns.' },
  { key: 'passing_interceptions', valueKey: 'projected_interceptions', label: 'INT', digits: 2, hist: 0, tip: 'Projected interceptions thrown.' },
  { key: 'rushing_yards', valueKey: 'projected_rush_yards', label: 'Rush yds', digits: 1, hist: 0, tip: 'Projected rushing yards.' },
  { key: 'rushing_tds', valueKey: 'projected_rush_tds', label: 'Rush TD', digits: 2, hist: 0, tip: 'Projected rushing touchdowns.' },
];

const KICKER_COLUMNS = [
  { key: 'fg_made', valueKey: 'projected_fg', label: 'FG', digits: 1, hist: 0, tip: 'Projected field goals made.' },
  { key: 'fg_att', valueKey: 'projected_fg_att', label: 'FG att', digits: 1, hist: 0, tip: 'Projected field-goal attempts.' },
  { key: 'pat_made', valueKey: 'projected_xp', label: 'XP', digits: 1, hist: 0, tip: 'Projected extra points made.' },
];

function playerColumns(mode) {
  if (mode === 'QB') return QB_COLUMNS;
  if (mode === 'K') return KICKER_COLUMNS;
  return SKILL_COLUMNS;
}

function resolveStat(player, column, mode) {
  const pos = player.position;
  if (mode !== 'all') {
    return { value: player[column.valueKey], histKey: column.key, skip: false };
  }
  if (pos === 'QB') {
    if (column.key === 'targets' || column.key === 'receptions') return { skip: true };
    if (column.key === 'yards') {
      return { value: player.projected_pass_yards, histKey: 'passing_yards', skip: false };
    }
    if (column.key === 'touchdowns') {
      return { value: player.projected_pass_tds, histKey: 'passing_tds', skip: false };
    }
    return { value: player[column.valueKey], histKey: column.key, skip: false };
  }
  if (pos === 'K') return { skip: true };
  return { value: player[column.valueKey], histKey: column.key, skip: false };
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

const PLAYER_ASC_KEYS = ['player', 'matchup', 'status'];
const INJURY_ORDER = { out: 0, ir: 0, doubtful: 1, questionable: 2, probable: 3 };

function injurySortRank(status) {
  return status ? INJURY_ORDER[String(status).toLowerCase()] ?? 4 : 9;
}

function playerSortValue(player, key) {
  if (key === 'player') return player.name || '';
  if (key === 'matchup') return player.matchup || player.opponent || '';
  if (key === 'proj') return proj(player);
  if (key === 'recent') return recentAvg(player);
  if (key === 'depth') return player.depth_rank ?? null;
  if (key === 'status') return injurySortRank(player.injury_status);
  const column = playerColumns(state.playersPos).find((item) => item.key === key);
  if (!column) return null;
  const stat = resolveStat(player, column, state.playersPos);
  return stat.skip ? null : stat.value ?? null;
}

function normalizePlayerSort() {
  const valid = new Set(['player', 'matchup', 'proj', 'recent', 'depth', 'status']);
  playerColumns(state.playersPos).forEach((column) => valid.add(column.key));
  if (!valid.has(state.playersSort.key)) {
    state.playersSort = { key: 'proj', dir: 'desc' };
  }
}

function setPlayerSort(key) {
  nextSort(state.playersSort, key, PLAYER_ASC_KEYS);
  normalizePlayerSort();
  renderSortState(state.playersSort, '#players-sort', {
    proj: 'proj',
    recent: 'recent',
    depth: 'depth',
  });
  renderPlayers();
}

function filteredPlayers() {
  const team = ($('#players-team-filter')?.value || '').trim().toLowerCase();
  const list = state.players.filter((p) => {
    const posOk = state.playersPos === 'all' || p.position === state.playersPos;
    const teamOk = !team || (p.team || '').toLowerCase().includes(team);
    return posOk && teamOk;
  });
  normalizePlayerSort();
  return sortRecords(list, (player) => playerSortValue(player, state.playersSort.key), state.playersSort);
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
  const previousScroll = wrap.querySelector('.table-scroll')?.scrollLeft || 0;
  const list = filteredPlayers();
  clear(wrap);

  const count = $('#players-count');
  if (count) count.textContent = `${list.length} player${list.length === 1 ? '' : 's'}`;

  const sub = $('#players-sub');
  if (sub) {
    const historyNote =
      state.historyWindow === 6
        ? 'Small numbers under each stat are the last six games, oldest to newest.'
        : 'Small numbers under each stat are the last three games; use + Last 6 for all six.';
    sub.textContent = state.players.length
      ? `Projected fantasy points and stat lines${state.meta?.week ? ` for week ${state.meta.week}` : ''}. ${historyNote}`
      : 'Projected fantasy points and stat lines for this week’s matchups.';
  }

  const historyToggle = $('#players-history-toggle');
  if (historyToggle) {
    const expanded = state.historyWindow === 6;
    historyToggle.textContent = expanded ? '− Last 3' : '+ Last 6';
    historyToggle.title = expanded
      ? 'Show only the last three games'
      : 'Show the last six games';
    historyToggle.setAttribute('aria-pressed', String(expanded));
  }

  const stats = $('#players-stats');
  if (stats) {
    clear(stats);
    const top = [...list].sort((a, b) => proj(b) - proj(a))[0];
    const injured = list.filter((p) => p.injury_status).length;
    stats.appendChild(
      statTile({
        label: 'Players',
        value: list.length,
        sub: 'matching filter',
        title: 'Players in the current position and team filter.',
      })
    );
    stats.appendChild(
      statTile({
        label: 'Top projection',
        value: top ? `${proj(top).toFixed(1)}` : '—',
        sub: top ? `${top.name} · ${top.team}` : '',
        cls: 'good',
        title: 'Highest projected fantasy points in the current filter.',
      })
    );
    stats.appendChild(
      statTile({
        label: 'Average projection',
        value: list.length
          ? (list.reduce((a, b) => a + proj(b), 0) / list.length).toFixed(1)
          : '—',
        sub: 'fantasy points',
        title: 'Average projected fantasy points across the current filter.',
      })
    );
    stats.appendChild(
      statTile({
        label: 'On the report',
        value: injured,
        sub: 'players with an injury status',
        cls: injured ? 'warn' : '',
        title: 'Players carrying any injury-report status (Q, D, O or IR).',
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
  headRow.appendChild(
    sortHeader('Player', 'player', state.playersSort, setPlayerSort, 'sticky-col')
  );
  headRow.appendChild(
    sortHeader(
      'Matchup',
      'matchup',
      state.playersSort,
      setPlayerSort,
      'hide-sm',
      'Opponent and kickoff time for this week’s game.'
    )
  );
  headRow.appendChild(
    sortHeader(
      'Proj',
      'proj',
      state.playersSort,
      setPlayerSort,
      'num',
      'Projected fantasy points (standard scoring, 4-pt passing TD). “Range” is the 80% projection interval; the small numbers are the last six games.'
    )
  );
  const columns = playerColumns(state.playersPos);
  columns.forEach((column) => {
    headRow.appendChild(
      sortHeader(
        column.label,
        column.key,
        state.playersSort,
        setPlayerSort,
        `num col-${column.key}`,
        `${column.tip} ${STAT_HISTORY_TIP}`
      )
    );
  });
  headRow.appendChild(
    sortHeader(
      'Status',
      'status',
      state.playersSort,
      setPlayerSort,
      '',
      'Injury report status: Q questionable, D doubtful, O out, IR injured reserve.'
    )
  );
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  const maxProj = Math.max(...list.map(proj), 1);
  list.forEach((player) => {
    const playerId = String(player.player_id);
    const isOpen = state.selectedPlayerId === playerId;
    const tr = el('tr', isOpen ? 'row-open' : '');
    tr.tabIndex = 0;
    tr.setAttribute('aria-expanded', String(isOpen));
    tr.setAttribute(
      'aria-label',
      `${isOpen ? 'Hide' : 'Show'} last six games for ${player.name}`
    );
    tr.title = isOpen ? 'Hide last six games' : 'Show last six games';
    const toggleRow = () => {
      state.selectedPlayerId = isOpen ? null : playerId;
      renderPlayers();
    };
    tr.addEventListener('click', (event) => {
      if (event.target.closest('a, button, summary, details')) return;
      toggleRow();
    });
    tr.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (event.target.closest('a, button, summary, details')) return;
      event.preventDefault();
      toggleRow();
    });
    const playerTd = el('td', 'sticky-col');
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
    pwrap.appendChild(el('span', 'row-chevron', isOpen ? '▾' : '▸'));
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
    if (player.floor_fantasy_points !== undefined && player.ceiling_fantasy_points !== undefined) {
      const range = el('details', 'stat-range');
      range.open = !window.matchMedia('(max-width: 760px)').matches;
      const summary = el('summary', '', 'Range');
      summary.title =
        '80% projection interval: floor (10th percentile) to ceiling (90th percentile).';
      range.appendChild(summary);
      range.appendChild(
        el(
          'span',
          'range-value',
          `${player.floor_fantasy_points}–${player.ceiling_fantasy_points}`
        )
      );
      projTd.appendChild(range);
    }
    projTd.appendChild(historyRow(player, 'fantasy_points', 1));
    tr.appendChild(projTd);

    columns.forEach((column) => {
      const stat = resolveStat(player, column, state.playersPos);
      const td = el('td', `num col-${column.key}`);
      td.appendChild(el('div', 'stat-val', fmtStat(stat.value, stat.skip, column.digits)));
      if (!stat.skip) td.appendChild(historyRow(player, stat.histKey, column.hist));
      tr.appendChild(td);
    });

    const statusTd = el('td');
    statusTd.appendChild(injuryPill(player.injury_status));
    tr.appendChild(statusTd);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  wrap.appendChild(
    el('p', 'scroll-hint only-sm', 'Swipe the table sideways for all stats and status →')
  );
  const scroll = el('div', 'table-scroll');
  scroll.appendChild(table);
  const box = el('div', 'table-wrap');
  box.appendChild(scroll);
  wrap.appendChild(box);
  scroll.scrollLeft = previousScroll;
}

function fmtStat(value, skip, digits = 1) {
  if (skip) return '—';
  if (value === undefined || value === null) return '—';
  return Number(value).toFixed(digits);
}

/* ---------------- playoffs ---------------- */

function futuresFor(abbr) {
  return (state.futuresByTeam && state.futuresByTeam.get(abbr)) || null;
}

function renderFuturesFreshness() {
  const node = $('#futures-freshness');
  if (!node) return;
  const futures = state.futures;
  if (!futures || !futures.updated_at) {
    node.textContent = 'Super Bowl futures are not available yet.';
    return;
  }
  const summary = futures.summary || {};
  const overround = futures.summary?.overround;
  node.textContent = `Futures updated ${relativeTime(futures.updated_at)} · ${summary.teams_matched ?? 0} teams priced${
    overround !== null && overround !== undefined ? ` · market overround ${(overround * 100).toFixed(1)}%` : ''
  }`;
  node.title = new Date(futures.updated_at).toLocaleString();
}

const PLAYOFF_ASC_KEYS = ['team', 'conf'];

function playoffSortValue(team, key) {
  if (key === 'rank') return team.super_bowl_prob ?? null;
  if (key === 'team') return team.team || '';
  if (key === 'conf') return team.conference || '';
  if (key === 'edge') return futuresFor(team.team)?.edge ?? null;
  if (key === 'super_bowl') return team.super_bowl_prob ?? null;
  if (key === 'market') return futuresFor(team.team)?.market_sb_prob ?? null;
  if (key === 'conf_title') return team.conference_prob ?? null;
  if (key === 'playoff') return team.playoff_prob ?? null;
  return null;
}

function setPlayoffSort(key) {
  nextSort(state.playoffsSort, key, PLAYOFF_ASC_KEYS);
  renderSortState(state.playoffsSort, '#playoffs-sort', {
    rank: 'sb',
    super_bowl: 'sb',
    conf_title: 'conf',
    playoff: 'playoff',
    edge: 'edge',
    team: 'team',
    market: 'market',
  });
  renderPlayoffs();
}

function filteredPlayoffs() {
  const team = ($('#playoffs-team-filter')?.value || '').trim().toLowerCase();
  const list = state.playoffs.filter((t) => {
    const confOk = state.playoffsConf === 'all' || t.conference === state.playoffsConf;
    const teamOk = !team || (t.team || '').toLowerCase().includes(team);
    return confOk && teamOk;
  });
  return sortRecords(
    list,
    (entry) => playoffSortValue(entry, state.playoffsSort.key),
    state.playoffsSort
  );
}

function renderPlayoffs() {
  const wrap = $('#playoffs-table-wrap');
  if (!wrap) return;
  const previousScroll = wrap.querySelector('.table-scroll')?.scrollLeft || 0;
  clear(wrap);
  const list = filteredPlayoffs();
  renderFuturesFreshness();

  const count = $('#playoffs-count');
  if (count) count.textContent = `${list.length} team${list.length === 1 ? '' : 's'}`;

  const stats = $('#playoffs-stats');
  if (stats) {
    clear(stats);
    const fav = [...state.playoffs].sort((a, b) => (b.super_bowl_prob ?? 0) - (a.super_bowl_prob ?? 0))[0];
    const withFutures = state.playoffs
      .map((team) => ({ team, futures: futuresFor(team.team) }))
      .filter((item) => item.futures && item.futures.edge !== null && item.futures.edge !== undefined);
    const bestEdge = [...withFutures].sort((a, b) => b.futures.edge - a.futures.edge)[0];
    const marketFav = [...withFutures].sort(
      (a, b) => (b.futures.market_sb_prob ?? 0) - (a.futures.market_sb_prob ?? 0)
    )[0];

    stats.appendChild(
      statTile({
        label: 'Teams',
        value: state.playoffs.length,
        sub: 'simulated',
        title: 'Teams included in the Monte Carlo simulation.',
      })
    );
    stats.appendChild(
      statTile({
        label: 'Title favourite',
        value: fav ? teamAbbr(fav.team) : '—',
        sub: fav
          ? `${pct(fav.super_bowl_prob, 1)}${decSuffix(fav.super_bowl_prob)} model · ${pct(
              futuresFor(fav.team)?.market_sb_prob,
              1
            )}${decSuffix(futuresFor(fav.team)?.market_sb_prob)} market`
          : '',
        cls: 'good',
        title:
          'Team with the highest simulated Super Bowl probability. Model and market are shown as probabilities with fair decimal odds.',
      })
    );
    if (bestEdge) {
      stats.appendChild(
        statTile({
          label: 'Best futures edge',
          value: edgePoints(bestEdge.futures.edge),
          sub: `${teamAbbr(bestEdge.team.team)} · model ${pct(bestEdge.team.super_bowl_prob, 1)}${decSuffix(
            bestEdge.team.super_bowl_prob
          )} vs market ${pct(bestEdge.futures.market_sb_prob, 1)}${decSuffix(bestEdge.futures.market_sb_prob)}`,
          cls: bestEdge.futures.edge >= 0.02 ? 'good' : '',
          title:
            'Largest model-minus-market gap on the Super Bowl winner market, in percentage points after removing the bookmaker overround.',
        })
      );
      stats.appendChild(
        statTile({
          label: 'Market favourite',
          value: marketFav ? teamAbbr(marketFav.team.team) : '—',
          sub: marketFav
            ? `${pct(marketFav.futures.market_sb_prob, 1)}${decSuffix(
                marketFav.futures.market_sb_prob
              )} · ${marketFav.futures.best_price ? marketFav.futures.best_price.toFixed(2) : '—'}`
            : '',
          title:
            'Team with the highest vig-free market Super Bowl probability, with the best available decimal price.',
        })
      );
    } else {
      stats.appendChild(
        statTile({
          label: 'Playoff locks',
          value: state.playoffs.filter((t) => (t.playoff_prob ?? 0) >= 0.99).length,
          sub: '≥ 99% to qualify',
          title: 'Teams the simulation sends to the playoffs in at least 99% of runs.',
        })
      );
      stats.appendChild(
        statTile({
          label: 'Futures',
          value: '—',
          sub: 'waiting for market odds',
          title: 'Super Bowl futures prices have not loaded yet.',
        })
      );
    }
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
  headRow.appendChild(
    sortHeader('#', 'rank', state.playoffsSort, setPlayoffSort, '', 'Rank by model Super Bowl probability.')
  );
  headRow.appendChild(
    sortHeader('Team', 'team', state.playoffsSort, setPlayoffSort, 'sticky-col')
  );
  headRow.appendChild(
    sortHeader('Conf', 'conf', state.playoffsSort, setPlayoffSort, 'hide-sm', 'AFC or NFC.')
  );
  headRow.appendChild(
    sortHeader(
      'Edge',
      'edge',
      state.playoffsSort,
      setPlayoffSort,
      'num',
      'Model probability minus vig-free market probability, in percentage points. Positive means the model is higher than the market.'
    )
  );
  headRow.appendChild(
    sortHeader(
      'Super Bowl',
      'super_bowl',
      state.playoffsSort,
      setPlayoffSort,
      '',
      'Probability of winning the Super Bowl.'
    )
  );
  headRow.appendChild(
    sortHeader(
      'Market',
      'market',
      state.playoffsSort,
      setPlayoffSort,
      '',
      'Vig-free market Super Bowl probability, normalized across teams, with the best available decimal price.'
    )
  );
  headRow.appendChild(
    sortHeader(
      'Conf. title',
      'conf_title',
      state.playoffsSort,
      setPlayoffSort,
      '',
      'Probability of winning the conference championship.'
    )
  );
  headRow.appendChild(
    sortHeader(
      'Playoff',
      'playoff',
      state.playoffsSort,
      setPlayoffSort,
      '',
      'Probability of reaching the playoffs in the simulated seasons.'
    )
  );
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el('tbody');
  list.forEach((team, index) => {
    const futures = futuresFor(team.team);
    const edge = futures ? futures.edge : null;
    const tr = el('tr');
    tr.appendChild(el('td', 'muted-2', String(index + 1)));

    const teamTd = el('td', 'sticky-col');
    const identity = el('a', 'player team-link');
    identity.href = `/teams/${team.team}.html`;
    identity.title = `Open ${teamFull(team.team)} team page`;
    identity.appendChild(teamBadge(team.team, true));
    const info = el('div');
    info.appendChild(el('div', 'nm', teamAbbr(team.team)));
    info.appendChild(el('div', 'tm', teamFull(team.team)));
    if (futures && edge !== null && edge !== undefined) {
      info.appendChild(
        el(
          'div',
          `tm only-sm ${edge >= 0.02 ? 'pos-odds' : edge < 0 ? 'neg-odds' : ''}`.trim(),
          `${edgePoints(edge)} futures`
        )
      );
    }
    identity.appendChild(info);
    teamTd.appendChild(identity);
    tr.appendChild(teamTd);

    tr.appendChild(el('td', 'muted hide-sm', team.conference || '—'));

    const edgeTd = el('td', 'num');
    if (edge !== null && edge !== undefined) {
      edgeTd.appendChild(
        el(
          'span',
          `tag edge-tag ${edge >= 0.02 ? 'edge' : edge > 0 ? 'neutral' : 'neg'}`,
          edgePoints(edge)
        )
      );
    } else {
      edgeTd.appendChild(el('span', 'muted-2', '—'));
    }
    tr.appendChild(edgeTd);

    tapeCell(
      tr,
      team.super_bowl_prob,
      (team.super_bowl_prob ?? 0) >= 0.1 ? 'green' : 'dim',
      'Chance to win the Super Bowl.'
    );

    const marketTd = el('td');
    if (futures && futures.market_sb_prob !== null && futures.market_sb_prob !== undefined) {
      marketTd.appendChild(
        el('div', 'proj', `${pct(futures.market_sb_prob, 1)}${decSuffix(futures.market_sb_prob)}`)
      );
      marketTd.appendChild(
        el(
          'div',
          'tm',
          `${futures.best_price ? futures.best_price.toFixed(2) : '—'}${
            futures.best_bookmaker ? ` · ${futures.best_bookmaker}` : ''
          }`
        )
      );
      marketTd.title = `Model ${pct(team.super_bowl_prob, 1)} vs market ${pct(futures.market_sb_prob, 1)}`;
    } else {
      marketTd.appendChild(el('span', 'muted-2', '—'));
    }
    tr.appendChild(marketTd);

    tapeCell(
      tr,
      team.conference_prob,
      '',
      'Chance to win the conference championship.'
    );
    tapeCell(
      tr,
      team.playoff_prob,
      (team.playoff_prob ?? 0) >= 0.8 ? 'green' : '',
      'Chance to reach the playoffs in the simulated seasons.'
    );
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  wrap.appendChild(
    el('p', 'scroll-hint only-sm', 'Swipe the table sideways for Super Bowl, market, conf. title and playoff →')
  );
  const scroll = el('div', 'table-scroll');
  scroll.appendChild(table);
  const box = el('div', 'table-wrap');
  box.appendChild(scroll);
  wrap.appendChild(box);
  scroll.scrollLeft = previousScroll;
}

function tapeCell(tr, value, cls, title = '') {
  const td = el('td', cls.includes('hide-sm') ? 'hide-sm' : '');
  if (title) td.title = title;
  td.appendChild(meter(value ?? 0, cls.replace('hide-sm', '').trim()));
  tr.appendChild(td);
}

/* ---------------- market ---------------- */

function renderDistribution(games) {
  const container = $('#games-distribution');
  if (!container) return;
  clear(container);

  const edges = games
    .map((game) => bestMarket(state.marketById.get(game.game_id))?.market.edge)
    .filter((value) => typeof value === 'number');

  const bins = [
    { label: '0–2', min: 0, max: 0.02 },
    { label: '2–5', min: 0.02, max: 0.05 },
    { label: '5–10', min: 0.05, max: 0.1 },
    { label: '10–15', min: 0.1, max: 0.15 },
    { label: '15–20', min: 0.15, max: 0.2 },
    { label: '20–30', min: 0.2, max: 0.3 },
    { label: '30+', min: 0.3, max: Infinity },
  ];
  const counts = bins.map(
    (bin) => edges.filter((edge) => edge >= bin.min && edge < bin.max).length
  );
  const max = Math.max(...counts, 1);

  bins.forEach((bin, index) => {
    const col = el('div', 'dist-col');
    const value = el('div', 'dist-label', String(counts[index]));
    value.style.color = counts[index] ? 'var(--muted)' : 'var(--muted-2)';
    col.appendChild(value);
    const bar = el('div', 'dist-bar');
    bar.style.height = `${Math.max(3, (counts[index] / max) * 72)}px`;
    bar.title = `${counts[index]} game${counts[index] === 1 ? '' : 's'} in this bin`;
    if (!counts[index]) bar.style.opacity = '0.28';
    col.appendChild(bar);
    col.appendChild(el('div', 'dist-label', bin.label));
    container.appendChild(col);
  });
}

const SORT_OPTIONS = {
  schedule: [
    ['kickoff', 'Kickoff order'],
    ['confidence', 'Most confident'],
    ['close', 'Closest games'],
  ],
  edges: [
    ['edge', 'Biggest edge'],
    ['ev', 'Best price (EV)'],
    ['kickoff', 'Kickoff order'],
  ],
};

function rebuildGamesSort() {
  const select = $('#games-sort');
  if (!select) return;
  const previous = select.value;
  const options = SORT_OPTIONS[state.gamesView] || SORT_OPTIONS.schedule;
  select.innerHTML = '';
  options.forEach(([value, label]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });
  select.value = options.some(([value]) => value === previous) ? previous : options[0][0];
}

function setGamesView(view) {
  state.gamesView = view;
  document.querySelectorAll('#games-view-seg button').forEach((btn) => {
    const active = btn.dataset.view === view;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
  rebuildGamesSort();
  const sortSelect = $('#games-sort');
  if (sortSelect) sortSelect.value = (SORT_OPTIONS[view] || SORT_OPTIONS.schedule)[0][0];

  if (view === 'edges') {
    const oddsWeeks = [...new Set((state.market?.games || []).map((g) => g.week))]
      .filter((w) => w !== undefined && w !== null)
      .map(String);
    const select = $('#games-week-filter');
    const current = select?.value;
    if (select && current && current !== 'all' && !oddsWeeks.includes(current) && oddsWeeks.length) {
      select.value = oddsWeeks.sort((a, b) => Number(a) - Number(b))[0];
    }
  }

  renderGames();
}

function setupFilters() {
  const rerenderGames = () => renderGames();
  $('#games-week-filter')?.addEventListener('change', rerenderGames);
  $('#games-team-filter')?.addEventListener('input', rerenderGames);
  $('#games-sort')?.addEventListener('change', rerenderGames);
  document.querySelectorAll('#games-view-seg button').forEach((btn) => {
    btn.addEventListener('click', () => setGamesView(btn.dataset.view));
  });

  $('#players-team-filter')?.addEventListener('input', renderPlayers);
  $('#players-history-toggle')?.addEventListener('click', () => {
    state.historyWindow = state.historyWindow === 6 ? 3 : 6;
    renderPlayers();
  });
  $('#players-sort')?.addEventListener('change', (event) => {
    const key = event.target.value;
    state.playersSort = { key, dir: key === 'depth' ? 'asc' : 'desc' };
    renderPlayers();
  });
  document.querySelectorAll('#players-pos-seg button').forEach((btn) => {
    btn.addEventListener('click', () => setPosition(btn.dataset.pos));
  });

  $('#playoffs-team-filter')?.addEventListener('input', renderPlayoffs);
  $('#playoffs-sort')?.addEventListener('change', (event) => {
    const key =
      {
        sb: 'super_bowl',
        conf: 'conf_title',
        playoff: 'playoff',
        edge: 'edge',
        team: 'team',
        market: 'market',
      }[event.target.value] || 'super_bowl';
    state.playoffsSort = { key, dir: key === 'team' ? 'asc' : 'desc' };
    renderPlayoffs();
  });
  document.querySelectorAll('#playoffs-conf-seg button').forEach((btn) => {
    btn.addEventListener('click', () => setConference(btn.dataset.conf));
  });
}

function setPosition(pos) {
  state.playersPos = pos;
  state.selectedPlayerId = null;
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
  if (hash === 'market') {
    activateTab('games');
    setGamesView('edges');
  } else if (['games', 'players', 'playoffs'].includes(hash)) {
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
  const season = metaData.season || state.games?.[0]?.season || null;
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
  renderModelFreshness();
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
    const [games, players, playoffs, market, futures, explanations, meta] = await Promise.all([
      loadGames(),
      fetchJSON('players_week.json'),
      fetchJSON('playoffs.json'),
      fetchReport('market_comparison.json').catch(() => null),
      fetchReport('futures_comparison.json').catch(() => null),
      fetchReport('model_explanations.json').catch(() => null),
      fetchJSON('meta.json').catch(() => null),
    ]);

    state.games = games || [];
    state.players = players || [];
    state.playoffs = playoffs || [];
    state.market = market;
    state.futures = futures;
    state.futuresByTeam = new Map((futures?.teams || []).map((team) => [team.team, team]));
    state.explanationById = new Map(
      (explanations?.match?.games || []).map((game) => [game.game_id, game])
    );
    state.scoreExplanationById = new Map(
      (explanations?.score?.games || []).map((game) => [game.game_id, game])
    );
    state.marketById = new Map((market?.games || []).map((g) => [g.game_id, g]));
    state.gamesById = new Map(state.games.map((g) => [g.game_id, g]));
    applyMeta(meta);

    // Retry once if the meta call was dropped (keeps the freshness chip honest).
    if (!meta) {
      fetchJSON('meta.json')
        .then((retry) => {
          if (retry) applyMeta(retry);
        })
        .catch(() => {});
    }

    // Populate filters once, defaulting to the current week.
    const gameWeeks = [...new Set(state.games.map((g) => g.week))].sort((a, b) => a - b);
    const gameWeekSelect = $('#games-week-filter');
    populateSelect(gameWeekSelect, gameWeeks, 'Week');
    const currentWeek = defaultWeek(state.games);
    if (gameWeekSelect && currentWeek !== null && gameWeeks.includes(currentWeek)) {
      gameWeekSelect.value = String(currentWeek);
    }

    if (!state.meta.week && currentWeek !== null) state.meta.week = currentWeek;

    // Default position filter to QB when available.
    const positions = new Set(state.players.map((p) => p.position));
    setPosition(positions.has('QB') ? 'QB' : 'all');

    rebuildGamesSort();
    renderGames();
    renderPlayers();
    renderPlayoffs();
    renderLegend('games-legend', ['games']);
    renderLegend('players-legend', ['players']);
    renderLegend('playoffs-legend', ['playoffs']);

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
