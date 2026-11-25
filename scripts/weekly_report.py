#!/usr/bin/env python3
"""Print a Markdown weekly model report for GitHub issues."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def load(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"# Weekly NFL Model Report\n")
    print(f"_Generated: {now}_\n")

    meta = load("predictions/meta.json", {})
    if meta:
        print("## Data freshness\n")
        print(f"- Games: **{meta.get('games', '—')}**")
        print(f"- Season: **{meta.get('season', '—')}**")
        print(f"- Generated: **{meta.get('generated_at', '—')}**\n")

    backtest = load("reports/backtest.json", {})
    if backtest:
        pooled = backtest.get("pooled", {})
        print("## Match model backtest\n")
        print("| Metric | Value |")
        print("|--------|-------|")
        for key in ["accuracy", "log_loss", "brier", "auc"]:
            if key in pooled:
                print(f"| {key} | {pooled[key]:.4f} |")
        print()

    player = load("reports/player_backtest.json", {})
    if player:
        print("## Player model backtest\n")
        print("| Position | MAE | RMSE | Rows |")
        print("|----------|-----|------|------|")
        for pos, m in player.items():
            if m:
                print(f"| {pos} | {m.get('mae', '—')} | {m.get('rmse', '—')} | {m.get('rows', '—')} |")
        print()

    market = load("reports/market_comparison.json", {})
    if market:
        summary = market.get("summary", {})
        print("## Market comparison\n")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Games compared | {summary.get('games_compared', '—')} |")
        print(f"| Mean edge | {summary.get('mean_edge', '—')} |")
        print(f"| Mean abs edge | {summary.get('mean_abs_edge', '—')} |")
        print(f"| Max abs edge | {summary.get('max_abs_edge', '—')} |")
        print()

    ablation = load("reports/feature_ablation.json", {})
    if ablation:
        print("## Feature ablation\n")
        print("| Feature set | Accuracy | Log loss | AUC |")
        print("|-------------|----------|----------|-----|")
        for name, data in ablation.items():
            pooled = data.get("pooled", {})
            if pooled:
                print(
                    f"| {name} | {pooled.get('accuracy', 0):.4f} | "
                    f"{pooled.get('log_loss', 0):.4f} | {pooled.get('auc', 0):.4f} |"
                )
        print()

    print("---")
    print("_Model-implied metrics; not betting advice._")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"# Weekly NFL Model Report\n\nReport generation failed: {exc}")
