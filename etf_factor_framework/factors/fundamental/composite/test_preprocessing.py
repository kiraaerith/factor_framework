r"""
FactorPreprocessor unit tests with synthetic data.

Run:
    cd D:\code_project_v2\etf_cross_ml-master\etf_factor_framework
    C:\Users\cheny\miniconda3\envs\etf_cross_ml\python.exe factors\fundamental\composite\test_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import numpy as np
from factors.fundamental.composite.preprocessing import (
    FactorPreprocessor,
    _cross_sectional_zscore,
    _neutralize_industry,
    _neutralize_size,
    _orthogonalize,
)

np.random.seed(42)


def test_zscore():
    """Z-score: each column should have mean~0, std~1."""
    N, T = 100, 50
    arr = np.random.randn(N, T) * 3 + 10  # non-zero mean, non-unit std
    result = _cross_sectional_zscore(arr)

    col_means = np.nanmean(result, axis=0)
    col_stds = np.nanstd(result, axis=0, ddof=1)

    assert np.allclose(col_means, 0, atol=1e-10), f"mean not zero: {col_means[:5]}"
    assert np.allclose(col_stds, 1, atol=1e-10), f"std not one: {col_stds[:5]}"
    print("[PASS] test_zscore")


def test_zscore_with_nan():
    """Z-score should preserve NaN positions."""
    N, T = 50, 20
    arr = np.random.randn(N, T)
    arr[0, 0] = np.nan
    arr[10, 5] = np.nan

    result = _cross_sectional_zscore(arr)
    assert np.isnan(result[0, 0]), "NaN not preserved at [0,0]"
    assert np.isnan(result[10, 5]), "NaN not preserved at [10,5]"
    assert np.isfinite(result[1, 0]), "Non-NaN became NaN"
    print("[PASS] test_zscore_with_nan")


def test_zscore_few_valid():
    """Z-score: n_valid < 3 should keep raw values."""
    N, T = 5, 3
    arr = np.full((N, T), np.nan)
    arr[0, 0] = 1.0
    arr[1, 0] = 2.0  # only 2 valid in column 0

    result = _cross_sectional_zscore(arr)
    assert result[0, 0] == 1.0, "Should keep raw when n_valid < 3"
    assert result[1, 0] == 2.0, "Should keep raw when n_valid < 3"
    print("[PASS] test_zscore_few_valid")


def test_industry_neutral():
    """Industry neutral: within each industry, mean should be ~0."""
    N, T = 100, 30
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])
    industry_map = {s: f"ind_{i % 5}" for i, s in enumerate(symbols)}

    # Give each industry a different mean
    arr = np.random.randn(N, T)
    for i in range(N):
        arr[i, :] += (i % 5) * 10  # industry 0: +0, industry 1: +10, ...

    result = _neutralize_industry(arr, symbols, industry_map)

    # Check each industry has mean ~0
    for ind_id in range(5):
        mask = np.array([i % 5 == ind_id for i in range(N)])
        group_mean = np.nanmean(result[mask, :], axis=0)
        assert np.allclose(group_mean, 0, atol=1e-10), \
            f"Industry {ind_id} mean not zero: {group_mean[:3]}"

    print("[PASS] test_industry_neutral")


def test_size_neutral():
    """Size neutral: factor should be uncorrelated with log(mc) after."""
    N, T = 200, 20
    mc = np.random.uniform(10, 1000, (N, T))  # market cap
    # Factor correlated with market cap
    arr = 0.5 * np.log(mc) + np.random.randn(N, T) * 0.1

    result = _neutralize_size(arr, mc)

    # Check correlation with log(mc) is ~0 for each day
    for t in range(T):
        valid = np.isfinite(result[:, t]) & np.isfinite(mc[:, t])
        corr = np.corrcoef(result[valid, t], np.log(mc[valid, t]))[0, 1]
        assert abs(corr) < 0.05, f"Day {t}: corr={corr:.4f} still significant"

    print("[PASS] test_size_neutral")


def test_orthogonalize():
    """Orthogonalize: factor correlations should be significantly reduced."""
    N, T, F = 500, 30, 4
    # Create highly correlated factors
    base = np.random.randn(N, T)
    features = np.zeros((N, T, F))
    features[:, :, 0] = base + np.random.randn(N, T) * 0.1
    features[:, :, 1] = base * 0.8 + np.random.randn(N, T) * 0.2
    features[:, :, 2] = np.random.randn(N, T)
    features[:, :, 3] = base * 0.3 + np.random.randn(N, T) * 0.5

    # Measure before
    before_max_corr = []
    for t in range(T):
        corr = np.corrcoef(features[:, t, :].T)
        off_diag = np.abs(corr[np.triu_indices(F, k=1)])
        before_max_corr.append(off_diag.max())

    result = _orthogonalize(features)

    # Measure after
    after_max_corr = []
    for t in range(T):
        X = result[:, t, :]
        valid = np.isfinite(X).all(axis=1)
        if valid.sum() < F + 1:
            continue
        corr = np.corrcoef(X[valid].T)
        off_diag = np.abs(corr[np.triu_indices(F, k=1)])
        after_max_corr.append(off_diag.max())

    avg_before = np.mean(before_max_corr)
    avg_after = np.mean(after_max_corr)
    reduction = 1 - avg_after / avg_before

    print(f"  avg max|corr| before={avg_before:.3f}, after={avg_after:.3f}, "
          f"reduction={reduction:.1%}")
    assert reduction > 0.5, \
        f"Orthogonalization should reduce correlation by >50%, got {reduction:.1%}"

    print("[PASS] test_orthogonalize")


def test_pipeline_compose():
    """Pipeline: multiple steps compose correctly."""
    N, T, F = 100, 20, 3
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])
    industry_map = {s: f"ind_{i % 3}" for i, s in enumerate(symbols)}
    mc = np.random.uniform(10, 1000, (N, T))
    mc_symbols = symbols.copy()

    features = np.random.randn(N, T, F) * 5 + 20

    pipeline = FactorPreprocessor(
        steps=['industry_neutral', 'size_neutral', 'zscore', 'orthogonalize'],
        size_factor_indices={1},  # factor[1] is SIZE-like, skip size_neutral
    )
    pipeline.set_industry_map(industry_map)
    pipeline.set_market_cap(mc, mc_symbols)

    result = pipeline.transform(features, symbols)

    assert result.shape == (N, T, F), f"Shape mismatch: {result.shape}"
    assert np.isfinite(result).sum() > 0, "All NaN output"

    print("[PASS] test_pipeline_compose")


def test_size_factor_skip():
    """Pipeline: SIZE factor should skip size_neutral."""
    N, T, F = 50, 10, 2
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])
    mc = np.random.uniform(10, 1000, (N, T))

    # Factor 0: normal, Factor 1: SIZE (should skip size_neutral)
    features = np.random.randn(N, T, F)
    features[:, :, 1] = np.log(mc) + np.random.randn(N, T) * 0.1  # SIZE-like

    pipeline = FactorPreprocessor(
        steps=['size_neutral'],
        size_factor_indices={1},
    )
    pipeline.set_market_cap(mc, symbols)

    result = pipeline.transform(features, symbols)

    # Factor 0: should be neutralized (less correlated with mc)
    # Factor 1: should be unchanged (skipped)
    assert np.allclose(result[:, :, 1], features[:, :, 1]), \
        "SIZE factor should not be modified by size_neutral"

    print("[PASS] test_size_factor_skip")


def test_empty_steps():
    """Pipeline with no steps should return a copy."""
    N, T, F = 10, 5, 2
    features = np.random.randn(N, T, F)
    symbols = np.array([f"SHSE.{600000 + i}" for i in range(N)])

    pipeline = FactorPreprocessor(steps=[])
    result = pipeline.transform(features, symbols)

    assert np.allclose(result, features), "Empty pipeline should return copy"
    assert result is not features, "Should return a copy, not the same object"
    print("[PASS] test_empty_steps")


def test_invalid_step():
    """Pipeline should reject unknown steps."""
    try:
        FactorPreprocessor(steps=['unknown_step'])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'unknown_step' in str(e)
        print("[PASS] test_invalid_step")


if __name__ == '__main__':
    test_zscore()
    test_zscore_with_nan()
    test_zscore_few_valid()
    test_industry_neutral()
    test_size_neutral()
    test_orthogonalize()
    test_pipeline_compose()
    test_size_factor_skip()
    test_empty_steps()
    test_invalid_step()
    print("\n=== ALL TESTS PASSED ===")
