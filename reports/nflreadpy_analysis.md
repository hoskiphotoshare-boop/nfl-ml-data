# nflreadpy Data Inventory & Model Relevance

This document audits every `nflreadpy` loader available in **0.1.5** and records
whether it is used, ingested, or still to be tested in the NFL pipeline.

**Legend**

- **Used** – already in the feature/model pipeline.
- **Ingested** – downloaded to `data/` but not yet used in a model.
- **Candidate** – likely useful; next to test.
- **Low** – unlikely to help match/player/playoff models.
- **Metadata** – ID mappings / reference data.

| Loader | Dataset | Model relevance | Status | Notes |
|--------|---------|-----------------|--------|-------|
| `load_schedules` | Schedules / results | **Match** | **Used** | Rolling team averages, rest days, labels |
| `load_pbp` | Play-by-play | **Match / player** | **Used** | Pre-game rolling EPA, success, pass/rush EPA |
| `load_player_stats` | Player game stats | **Player** | **Used** | Targets, receptions, yards, TDs, fantasy points |
| `load_team_stats` | Team game stats | **Match** | **Ingested** | Not yet joined into match features |
| `load_nextgen_stats` | NGS passing/rushing/receiving | **Player / match** | **Candidate** | Air yards, separation, time-to-throw |
| `load_snap_counts` | Snap counts | **Player / match** | **Candidate** | Snap share and role changes |
| `load_depth_charts` | Depth charts | **Player / match** | **Ingested** | Role changes, starter continuity |
| `load_rosters_weekly` | Weekly rosters | **Player / match** | **Ingested** | Team changes, roster continuity |
| `load_rosters` | Final rosters | **Player** | **Candidate** | Season-level roster metadata |
| `load_injuries` | Injury reports | **Player / match** | **Tested – rejected** | Raw counts hurt match model |
| `load_trades` | Trades | **Player / match** | **Ingested** | Trade timing and roster disruption |
| `load_players` | Player metadata | **Metadata** | **Ingested** | Cross-platform IDs, names |
| `load_teams` | Team metadata | **Metadata** | **Used** (playoff sim) | Conference mapping |
| `load_pfr_advstats` | PFR advanced stats | **Player / match** | **Candidate** | Pressure, drops, broken tackles |
| `load_participation` | Participation | **Player / match** | **Candidate** | Routes run, personnel packages |
| `load_ftn_charting` | FTN charting | **Match / player** | **Candidate** | Coverage, pressure, play design |
| `load_ff_opportunity` | Expected fantasy points | **Player** | **Candidate** | Expected yards/TDs, opportunity |
| `load_ff_rankings` | Fantasy rankings | **Player** | **Low** | External rankings, not causal |
| `load_ff_playerids` | FF player IDs | **Metadata** | **Metadata** | ID mapping |
| `load_combine` | Combine results | **Player** | **Low** | Rookie athleticism; weak weekly signal |
| `load_draft_picks` | Draft picks | **Player / team** | **Low** | Team draft capital; not weekly |
| `load_contracts` | Contracts | **Player / team** | **Low** | Contract status; weak weekly signal |
| `load_officials` | Referee crews | **Match** | **Candidate (low)** | Penalty tendencies by crew |

## Priority order for the next experiments

1. **`load_team_stats`** – aggregate team offensive/defensive metrics by game.
2. **`load_snap_counts`** – team snap continuity and player role changes.
3. **`load_nextgen_stats`** – team-level passing/rushing efficiency.
4. **`load_pfr_advstats`** – pressure/drop/broken-tackle rates.
5. **`load_participation`** – personnel packages and route participation.
6. **`load_ftn_charting`** – coverage/pressure charting.
7. **`load_ff_opportunity`** – expected fantasy points for player models.
8. **Roster continuity features** from `trades` + `rosters_weekly` + `depth_charts`.
9. **`load_officials`** – referee penalty tendencies (low priority).

## What “test and backtest” means here

For each feature group:

1. Build the features in `src/features.py`.
2. Add them to a new ablation row in `scripts/feature_ablation.py`.
3. Run the walk-forward backtest (2022–2026).
4. Compare pooled accuracy / log loss / Brier / AUC.
5. Accept the group only if it improves log loss and Brier without hurting AUC.
6. Record the decision in `reports/experiments.md`.

This mirrors the HockeyPicks feature gate: **add → backtest → accept/reject**.
