from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_model_scorecard as scorecard  # noqa: E402


def test_num_and_fmt_handle_nulls():
    assert scorecard.num(None) == 0.0
    assert scorecard.num("not-a-number") == 0.0
    assert scorecard.num("2.5") == 2.5
    assert scorecard.num(3) == 3.0
    assert scorecard.fmt(None) == "0.000"
    assert scorecard.fmt(1.23456) == "1.235"


def test_scorecard_survives_null_report_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "OUT", str(tmp_path / "scorecard.json"))
    monkeypatch.setattr(
        scorecard,
        "match_numbers",
        lambda: {
            "season": 2026,
            "acc": None,
            "ll": None,
            "auc": None,
            "pooled_acc": None,
            "pooled_ll": None,
            "pooled_auc": None,
            "baseline_acc": None,
            "baseline_ll": None,
            "games": None,
        },
    )
    monkeypatch.setattr(scorecard, "player_numbers", lambda: {})
    monkeypatch.setattr(
        scorecard,
        "score_numbers",
        lambda: {
            "total_mae": None,
            "margin_mae": None,
            "total_naive": None,
            "margin_naive": None,
        },
    )
    payloads = {
        "reports/player_backtest.json": {
            "QB": {"mae": None, "coverage_80": None, "coverage_80_adj": None},
            "K": {"mae": None, "coverage_80": None, "coverage_80_adj": None},
        },
        "reports/playoff_backtest.json": {
            "pooled": {"playoff": None, "conf": None, "sb": None}
        },
        "reports/futures_comparison.json": {"summary": {"mean_abs_edge": None}},
        "reports/model_health.json": {"market": {"mean_abs_edge": None, "outliers": None}},
    }
    monkeypatch.setattr(
        scorecard, "load", lambda path, default: payloads.get(path, default)
    )

    scorecard.main()

    data = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert data["rows"]
    assert "None" not in json.dumps(data)
