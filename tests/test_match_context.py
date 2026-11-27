import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import polars as pl


REQUIRED = [
    "home_prior_games", "away_prior_games", "home_win_pct", "away_win_pct",
    "home_diff_pg", "away_diff_pg", "win_pct_diff", "diff_pg_diff",
    "home_blend_diff", "away_blend_diff", "blend_diff_diff",
    "home_blend_win", "away_blend_win", "blend_win_diff",
    "home_qb_prior_epa", "away_qb_prior_epa", "home_qb_starts", "away_qb_starts",
    "home_qb_changed", "away_qb_changed", "home_qb_missing", "away_qb_missing",
    "home_inj_impact", "away_inj_impact", "inj_impact_diff",
    "home_inj_missing", "away_inj_missing",
]


def test_match_features_has_context():
    if not os.path.exists("features/match_features.parquet"):
        import pytest
        pytest.skip("features not built")
    df = pl.read_parquet("features/match_features.parquet")
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"missing context cols: {missing}"


def test_match_model_has_context_and_count():
    if not os.path.exists("models/match_model.joblib"):
        import pytest
        pytest.skip("model not trained")
    bundle = joblib.load("models/match_model.joblib")
    feats = bundle["features"]
    missing = [c for c in REQUIRED if c not in feats]
    assert not missing, f"model missing context cols: {missing}"
    assert len(feats) >= 45, f"expected >=45 features, got {len(feats)}"
    assert bundle.get("n_features", len(feats)) == len(feats)
