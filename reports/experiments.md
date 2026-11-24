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
