# Feature Experiments

Walk-forward backtest window: **2022–2026** (train on all prior seasons, test on the target season).

## 2026-11-24 — Rest-day features

- **Features:** `home_rest_days`, `away_rest_days`, `rest_diff`, `home_short_week`, `away_short_week`
- **Before:** accuracy 0.5963, log loss 0.6651, Brier 0.2363, AUC 0.6244
- **After:** accuracy 0.5978, log loss 0.6642, Brier 0.2359, AUC 0.6261
- **Decision:** ✅ **Accept** — small but consistent improvement across all metrics.

## 2026-11-24 — Injury-report features

- **Features:** `home_inj_out`, `home_inj_doubtful`, `home_inj_questionable`, `home_inj_total`,
  `away_inj_out`, `away_inj_doubtful`, `away_inj_questionable`, `away_inj_total`,
  `inj_out_diff`, `inj_total_diff`
- **Before:** accuracy 0.5978, log loss 0.6642, Brier 0.2359, AUC 0.6261
- **After:** accuracy 0.5501, log loss 0.6900, Brier 0.2483, AUC 0.5406
- **Decision:** ❌ **Reject** — degraded every metric; the model fell back to home-team baseline.
- **Note:** injury data ingestion is retained for future player-level experiments, but the
  match model ignores these features.

## 2026-11-24 — Rolling team-stats features

- **Features:** `home_passing_epa`, `away_passing_epa`, `home_rushing_epa`, `away_rushing_epa`
- **Important:** computed as **pre-game rolling averages** from `team_stats`, not same-game stats.
- **Before:** accuracy 0.5986, log loss 0.6640, Brier 0.2358, AUC 0.6264
- **After:** accuracy 0.6048, log loss 0.6615, Brier 0.2345, AUC 0.6356
- **Decision:** ✅ **Accept** — consistent improvement across all metrics.

## 2026-11-24 — Snap-count features

- **Features:** team-week snap aggregates:
  `home/away_snap_players`, `snap_off_total`, `snap_def_total`,
  `snap_off_mean_pct`, `snap_off_players_50`, `snap_def_mean_pct`,
  `snap_def_players_50` (all pre-game rolling).
- **Before:** accuracy 0.6048, log loss 0.6615, Brier 0.2345, AUC 0.6356
- **After:** accuracy 0.6071, log loss 0.6632, Brier 0.2354, AUC 0.6299
- **Decision:** ❌ **Reject** — accuracy improved marginally, but log loss, Brier, and AUC all got worse.

## 2026-11-24 — Next-gen stats (rolling team aggregates)

- **Features:** team-week rolling averages from NGS passing, rushing, and receiving:
  - passing: time-to-throw, completed/intended air yards, aggressiveness, CPOE, passer rating
  - rushing: efficiency, 8+ defender rate, time-to-LOS, RYOE/attempt, RYOE%
  - receiving: cushion, separation, air-yard share, catch %, YAC above expectation
- **Before:** accuracy 0.6048, log loss 0.6615, Brier 0.2345, AUC 0.6356
- **After:** accuracy 0.6379, log loss 0.6316, Brier 0.2204, AUC 0.6908
- **Decision:** ✅ **Accept** — large, consistent improvement across all metrics.

## 2026-11-24 — PFR advanced stats

- **Features:** team-week rolling averages from PFR advanced stats:
  - pass: drop %, bad-throw %, pressure %
  - rush: yards before/after contact averages
  - rec: drop %, receiver rating
  - def: completion %, yards allowed/target, passer rating allowed, missed-tackle %,
    pressures, sacks
- **Before (NGS model):** accuracy ~0.643, log loss ~0.633, Brier ~0.221, AUC ~0.690
- **After (with PFR):** accuracy 0.638, log loss 0.637, Brier 0.223, AUC 0.684
- **Decision:** ❌ **Reject** — every metric got worse; PFR adds noise on top of NGS.

## 2026-11-24 — Roster continuity

- **Features:** `home_roster_continuity`, `away_roster_continuity`, `roster_continuity_diff`
  (week-over-week fraction of players retained).
- **Before:** accuracy 0.6418, log loss 0.6300, Brier 0.2199, AUC 0.6924
- **After:** accuracy 0.6379, log loss 0.6310, Brier 0.2200, AUC 0.6936
- **Decision:** ❌ **Reject for now** — essentially neutral; AUC improves by +0.0012
  but accuracy and log loss get slightly worse. Revisit with a longer continuity window
  or trade-weighted roster changes.

## 2026-11-24 — Participation features

- **Features:** pre-game rolling team averages from play participation:
  pass rushers faced, pressure rate, time to throw, air yards, defenders in box,
  man/zone coverage rates.
- **Before (2022–2026):** accuracy 0.6418, log loss 0.6300, Brier 0.2199, AUC 0.6924
- **After:** accuracy 0.6387, log loss 0.6403, Brier 0.2235, AUC 0.6899
- **Decision:** ❌ **Reject** — worse across all metrics; participation adds noise
  on top of NGS and team stats.

## 2026-11-24 — FTN charting

- **Features:** pre-game rolling team averages from FTN charting:
  motion, play action, screen, RPO, no huddle, QB out of pocket,
  blitzers, pass rushers, defenders in box, offense backfield.
- **Before (2022–2026):** accuracy 0.6410, log loss 0.6307, Brier 0.2201, AUC 0.6929
- **After:** accuracy 0.6418, log loss 0.6314, Brier 0.2205, AUC 0.6915
- **Decision:** ❌ **Reject** — negligible accuracy gain; log loss, Brier, and AUC all got worse.
