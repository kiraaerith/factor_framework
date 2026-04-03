"""Unit tests for LabelGenerator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import numpy as np
from factors.fundamental.composite.label_generator import (
    LabelGenerator,
    _compute_forward_returns,
    _cross_sectional_rank,
    _cross_sectional_zscore,
    _neutralize_industry,
)

np.random.seed(42)


def test_forward_returns_basic():
    """Forward returns: basic correctness."""
    N, T = 3, 10
    close = np.ones((N, T)) * 100.0
    close[:, 5] = 110.0  # day 5 price = 110

    fwd = _compute_forward_returns(close, days=5)

    # day 0 -> day 5: 110/100 - 1 = 0.1
    assert np.isclose(fwd[0, 0], 0.1), f"Expected 0.1, got {fwd[0, 0]}"
    # last 5 days should be NaN
    assert np.all(np.isnan(fwd[:, -5:])), "Last 5 days should be NaN"
    print("[PASS] test_forward_returns_basic")


def test_forward_returns_nan_propagation():
    """Forward returns: NaN in close should propagate."""
    N, T = 2, 10
    close = np.arange(1, T + 1, dtype=float).reshape(1, T).repeat(N, axis=0)
    close[0, 3] = np.nan  # NaN at day 3

    fwd = _compute_forward_returns(close, days=2)
    # day 1 -> day 3: close[3] is NaN -> fwd[1] should be NaN
    assert np.isnan(fwd[0, 1]), "NaN close should propagate to forward return"
    # day 3 -> day 5: close[3] is NaN -> fwd[3] should be NaN
    assert np.isnan(fwd[0, 3]), "NaN close should propagate to forward return"
    print("[PASS] test_forward_returns_nan_propagation")


def test_rank_basic():
    """Rank: values should map to [0, 1] percentiles."""
    arr = np.array([
        [10.0, 30.0],
        [20.0, 20.0],
        [30.0, 10.0],
        [40.0, np.nan],
    ])
    result = _cross_sectional_rank(arr)

    # Column 0: 10<20<30<40 -> ranks 0, 1/3, 2/3, 1
    assert np.isclose(result[0, 0], 0.0)
    assert np.isclose(result[1, 0], 1.0 / 3.0)
    assert np.isclose(result[2, 0], 2.0 / 3.0)
    assert np.isclose(result[3, 0], 1.0)

    # Column 1: 30>20>10, NaN stays NaN -> ranks: 10->0, 20->0.5, 30->1, NaN
    assert np.isclose(result[2, 1], 0.0)
    assert np.isclose(result[1, 1], 0.5)
    assert np.isclose(result[0, 1], 1.0)
    assert np.isnan(result[3, 1])

    print("[PASS] test_rank_basic")


def test_rank_preserves_nan():
    """Rank: NaN positions should stay NaN."""
    N, T = 50, 10
    arr = np.random.randn(N, T)
    arr[5, 3] = np.nan
    arr[10, 7] = np.nan

    result = _cross_sectional_rank(arr)
    assert np.isnan(result[5, 3])
    assert np.isnan(result[10, 7])
    assert np.isfinite(result[0, 0])
    print("[PASS] test_rank_preserves_nan")


def test_rank_range():
    """Rank: all values should be in [0, 1]."""
    N, T = 100, 30
    arr = np.random.randn(N, T)
    result = _cross_sectional_rank(arr)

    valid = ~np.isnan(result)
    assert np.all(result[valid] >= 0.0)
    assert np.all(result[valid] <= 1.0)
    print("[PASS] test_rank_range")


def test_zscore_label():
    """Z-score on labels: mean~0, std~1 per day."""
    N, T = 200, 20
    arr = np.random.randn(N, T) * 5 + 100

    result = _cross_sectional_zscore(arr)
    col_means = np.nanmean(result, axis=0)
    col_stds = np.nanstd(result, axis=0, ddof=1)

    assert np.allclose(col_means, 0, atol=1e-10)
    assert np.allclose(col_stds, 1, atol=1e-10)
    print("[PASS] test_zscore_label")


def test_industry_neutral_label():
    """Industry neutral on labels: within-industry mean~0."""
    N, T = 100, 15
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])
    industry_map = {s: f"ind_{i % 4}" for i, s in enumerate(symbols)}

    arr = np.random.randn(N, T)
    for i in range(N):
        arr[i, :] += (i % 4) * 5

    result = _neutralize_industry(arr, symbols, industry_map)

    for ind_id in range(4):
        mask = np.array([i % 4 == ind_id for i in range(N)])
        group_mean = np.nanmean(result[mask, :], axis=0)
        assert np.allclose(group_mean, 0, atol=1e-10), \
            f"Industry {ind_id} mean not zero"

    print("[PASS] test_industry_neutral_label")


def test_generator_raw():
    """LabelGenerator with raw: should equal forward returns."""
    N, T = 50, 30
    close = np.cumsum(np.random.randn(N, T) * 0.01 + 1.001, axis=1) * 10
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    gen = LabelGenerator(forward_days=5, steps=['raw'])
    labels = gen.generate(close, symbols)

    expected = _compute_forward_returns(close, 5)
    assert np.allclose(labels, expected, equal_nan=True)
    print("[PASS] test_generator_raw")


def test_generator_empty_steps():
    """LabelGenerator with empty steps: should equal forward returns."""
    N, T = 50, 30
    close = np.cumsum(np.random.randn(N, T) * 0.01 + 1.001, axis=1) * 10
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    gen = LabelGenerator(forward_days=5, steps=[])
    labels = gen.generate(close, symbols)

    expected = _compute_forward_returns(close, 5)
    assert np.allclose(labels, expected, equal_nan=True)
    print("[PASS] test_generator_empty_steps")


def test_generator_rank():
    """LabelGenerator with rank: output in [0,1]."""
    N, T = 100, 20
    close = np.cumsum(np.random.randn(N, T) * 0.01 + 1.001, axis=1) * 10
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    gen = LabelGenerator(forward_days=5, steps=['rank'])
    labels = gen.generate(close, symbols)

    valid = ~np.isnan(labels)
    assert np.all(labels[valid] >= 0.0)
    assert np.all(labels[valid] <= 1.0)
    print("[PASS] test_generator_rank")


def test_generator_compose():
    """LabelGenerator with multiple steps: industry_neutral then rank."""
    N, T = 100, 20
    close = np.cumsum(np.random.randn(N, T) * 0.01 + 1.001, axis=1) * 10
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])
    industry_map = {s: f"ind_{i % 5}" for i, s in enumerate(symbols)}

    gen = LabelGenerator(forward_days=5, steps=['industry_neutral', 'rank'])
    gen.set_industry_map(industry_map)
    labels = gen.generate(close, symbols)

    assert labels.shape == (N, T)
    valid = ~np.isnan(labels)
    assert np.all(labels[valid] >= 0.0)
    assert np.all(labels[valid] <= 1.0)
    print("[PASS] test_generator_compose")


def test_generator_from_returns():
    """generate_from_returns: should skip forward return computation."""
    N, T = 50, 20
    fwd_returns = np.random.randn(N, T)
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    gen = LabelGenerator(forward_days=5, steps=['zscore'])
    labels = gen.generate_from_returns(fwd_returns, symbols)

    col_means = np.nanmean(labels, axis=0)
    assert np.allclose(col_means, 0, atol=1e-10)
    print("[PASS] test_generator_from_returns")


def test_generator_invalid_step():
    """LabelGenerator should reject unknown steps."""
    try:
        LabelGenerator(steps=['bad_step'])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'bad_step' in str(e)
        print("[PASS] test_generator_invalid_step")


def test_generator_last_days_nan():
    """Forward returns: last forward_days columns must be NaN."""
    N, T, days = 10, 50, 10
    close = np.cumsum(np.random.randn(N, T) * 0.01 + 1.001, axis=1) * 10
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    for steps in [['raw'], ['zscore'], ['rank']]:
        gen = LabelGenerator(forward_days=days, steps=steps)
        labels = gen.generate(close, symbols)
        assert np.all(np.isnan(labels[:, -days:])), \
            f"steps={steps}: last {days} cols should be NaN"

    print("[PASS] test_generator_last_days_nan")


if __name__ == '__main__':
    test_forward_returns_basic()
    test_forward_returns_nan_propagation()
    test_rank_basic()
    test_rank_preserves_nan()
    test_rank_range()
    test_zscore_label()
    test_industry_neutral_label()
    test_generator_raw()
    test_generator_empty_steps()
    test_generator_rank()
    test_generator_compose()
    test_generator_from_returns()
    test_generator_invalid_step()
    test_generator_last_days_nan()
    print("\n=== ALL TESTS PASSED ===")
