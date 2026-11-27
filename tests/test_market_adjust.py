import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from market_adjust import movement_weight, blend_toward, derive_spread_from_prob


def test_weight_no_movement():
    assert movement_weight(None, 0.03) == 0.0
    assert movement_weight(0.01, 0.03) == 0.0


def test_weight_thresholds():
    # 1x threshold -> 15%
    assert abs(movement_weight(0.03, 0.03) - 0.15) < 1e-9
    # 2x threshold -> 30%
    assert abs(movement_weight(0.06, 0.03) - 0.30) < 1e-9
    # 3x capped at 30%
    assert abs(movement_weight(0.09, 0.03) - 0.30) < 1e-9


def test_total_weight_threshold():
    assert movement_weight(1.0, 1.4) == 0.0
    assert abs(movement_weight(1.4, 1.4) - 0.15) < 1e-9
    assert abs(movement_weight(2.8, 1.4) - 0.30) < 1e-9


def test_blend():
    assert blend_toward(10.0, 20.0, 0.15) == (0.85 * 10 + 0.15 * 20)


def test_derive_spread_consistent():
    s = derive_spread_from_prob(0.5, 13.5)
    assert abs(s) < 1e-9
