"""
Engineering tests for ROA factor.
Run from etf_factor_framework directory:
    python factors/fundamental/profitability/test_roa_engineering.py
"""
import os
import sys
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.profitability.roa import ROA

LIXINGER_DB = r"E:\code_project_v2\china_stock_data\lixinger.db"

TEST_START = "2022-01-01"
TEST_END   = "2024-12-31"
TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

print("=" * 60)
print("ROA Engineering Tests")
print("=" * 60)

print(f"\n[Setup] Loading FundamentalData ({TEST_START} ~ {TEST_END})")
fd = FundamentalData(
    start_date=TEST_START,
    end_date=TEST_END,
    stock_codes=TEST_CODES,
)
calculator = ROA()
result = calculator.calculate(fd)

# ----------------------------------------------------------------
# Test 1: Shape / dtype
# ----------------------------------------------------------------
print("\n[Test 1] Shape / dtype")
assert result.values.ndim == 2, "values must be 2-D"
assert result.values.shape == (len(result.symbols), len(result.dates)), \
    f"shape mismatch: {result.values.shape} vs ({len(result.symbols)}, {len(result.dates)})"
assert result.values.dtype == np.float64, \
    f"expected float64, got {result.values.dtype}"
print(f"  shape={result.values.shape}, dtype={result.values.dtype}")
print("[PASS] Test 1: Shape / dtype")

# ----------------------------------------------------------------
# Test 2: Idempotency
# ----------------------------------------------------------------
print("\n[Test 2] Idempotency")
result2 = calculator.calculate(fd)
both_nan = np.isnan(result.values) & np.isnan(result2.values)
assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"
print("[PASS] Test 2: Idempotency")

# ----------------------------------------------------------------
# Test 3: NaN boundary
# ----------------------------------------------------------------
print("\n[Test 3] NaN boundary")
valid = result.values[~np.isnan(result.values)]
assert not np.isinf(valid).any(), "Factor contains inf values"
nan_ratio = np.isnan(result.values).mean()
assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
print(f"  nan_ratio={nan_ratio:.1%}")
print(f"[PASS] Test 3: NaN boundary (nan_ratio={nan_ratio:.1%})")

# ----------------------------------------------------------------
# Test 4: Golden Test
# q_m_roa_t is a direct field from DB - compare factor value vs DB value
# For 600519, find a quarter where report_date is known, verify the factor
# uses the DB value directly after the report_date.
# ----------------------------------------------------------------
print("\n[Test 4] Golden Test")
conn = sqlite3.connect(LIXINGER_DB)

# Get column name - try stock_code and code variants
cur = conn.execute("PRAGMA table_info(financial_statements)")
cols = [row[1] for row in cur.fetchall()]
print(f"  Table columns (first 10): {cols[:10]}")

# Determine code column name
code_col = "stock_code" if "stock_code" in cols else "code"

df_gold = pd.read_sql(
    f"SELECT {code_col}, date, report_date, q_m_roa_t FROM financial_statements "
    f"WHERE {code_col} = '600519' AND date LIKE '2023-09-30%'",
    conn
)
conn.close()

print(f"  Raw DB row: {df_gold.to_dict('records')}")
assert len(df_gold) > 0, "No row found for 600519 2023-09-30"
expected_val = float(df_gold['q_m_roa_t'].iloc[0])
release_date_str = pd.Timestamp(df_gold['report_date'].iloc[0]).strftime('%Y-%m-%d')
print(f"  Expected q_m_roa_t = {expected_val}, released on {release_date_str}")

# Build a small fd range after release date so the new quarter is used
fd_gold = FundamentalData(
    start_date="2023-01-01",
    end_date="2023-12-31",
    stock_codes=["600519"],
)
result_gold = calculator.calculate(fd_gold)

# Find symbol index (account for SHSE. prefix)
sym_matches = [i for i, s in enumerate(result_gold.symbols) if '600519' in str(s)]
assert len(sym_matches) > 0, f"600519 not found in symbols: {result_gold.symbols}"
sym_idx = sym_matches[0]

release_date = pd.Timestamp(release_date_str)
dates_pd = pd.DatetimeIndex(result_gold.dates)
# Find the date index on or after release_date
after_idx = dates_pd.searchsorted(release_date, side='left')
if after_idx >= len(dates_pd):
    after_idx = len(dates_pd) - 1

actual = result_gold.values[sym_idx, after_idx]
print(f"  Factor value at {dates_pd[after_idx].date()}: {actual}")
assert abs(actual - expected_val) < 1e-6, \
    f"Golden test FAILED: got {actual:.8f}, expected {expected_val:.8f}"
print(f"[PASS] Test 4: Golden Test (actual={actual:.6f}, expected={expected_val:.6f})")

# ----------------------------------------------------------------
# Test 5: Time alignment (quarterly factor)
# q_m_roa_t is a quarterly TTM field.
# Verify the factor respects report_date (no look-ahead bias):
#   - Before a report is released, the old value should still apply
#   - On/after release date, the new value should take effect
# ----------------------------------------------------------------
print("\n[Test 5] Time alignment")
conn = sqlite3.connect(LIXINGER_DB)
df_align = pd.read_sql(
    f"SELECT {code_col}, date, report_date, q_m_roa_t FROM financial_statements "
    f"WHERE {code_col} = '600519' AND date IN "
    f"(SELECT date FROM financial_statements WHERE {code_col} = '600519' "
    f" ORDER BY date DESC LIMIT 12) "
    f"ORDER BY date DESC",
    conn
)
conn.close()

print(f"  Recent quarters for 600519:")
for _, row in df_align.iterrows():
    print(f"    date={str(row['date'])[:10]}, report_date={str(row['report_date'])[:10]}, "
          f"q_m_roa_t={row['q_m_roa_t']}")

# Find Q3-2023 and Q2-2023
q3_rows = df_align[df_align['date'].str.startswith('2023-09-30')]
q2_rows = df_align[df_align['date'].str.startswith('2023-06-30')]

if len(q3_rows) > 0 and len(q2_rows) > 0:
    release_q3 = pd.Timestamp(str(q3_rows.iloc[0]['report_date'])[:10])
    expected_before = float(q2_rows.iloc[0]['q_m_roa_t'])
    expected_after  = float(q3_rows.iloc[0]['q_m_roa_t'])
    print(f"  Q2 2023 value (before Q3 release): {expected_before}")
    print(f"  Q3 2023 value (after Q3 release):  {expected_after}")
    print(f"  Q3 release date: {release_q3.date()}")

    fd_align = FundamentalData(
        start_date="2023-01-01",
        end_date="2023-12-31",
        stock_codes=["600519"],
    )
    result_align = calculator.calculate(fd_align)
    dates_align = pd.DatetimeIndex(result_align.dates)
    sym_matches2 = [i for i, s in enumerate(result_align.symbols) if '600519' in str(s)]
    sym_align_idx = sym_matches2[0]

    # Day before release
    before_pos = dates_align.searchsorted(release_q3, side='left')
    before_idx = max(0, before_pos - 1)
    val_before = result_align.values[sym_align_idx, before_idx]
    print(f"  Factor value on {dates_align[before_idx].date()} (before release): {val_before}")

    # Day of release (or first trading day after)
    after_pos = dates_align.searchsorted(release_q3, side='left')
    if after_pos >= len(dates_align):
        after_pos = len(dates_align) - 1
    val_after = result_align.values[sym_align_idx, after_pos]
    print(f"  Factor value on {dates_align[after_pos].date()} (after release):  {val_after}")

    assert abs(val_before - expected_before) < 1e-6, \
        f"Before-release value mismatch: got {val_before:.6f}, expected {expected_before:.6f}"
    assert abs(val_after - expected_after) < 1e-6, \
        f"After-release value mismatch: got {val_after:.6f}, expected {expected_after:.6f}"
    print(f"[PASS] Test 5: Time alignment (before={val_before}, after={val_after})")
else:
    print("  WARNING: Could not find Q2/Q3 2023 for 600519, skipping exact alignment check")
    print("  Checking generic: factor values change over time (not all identical)")
    sym_matches3 = [i for i, s in enumerate(result.symbols) if '600519' in str(s)]
    if sym_matches3:
        vals_600519 = result.values[sym_matches3[0], :]
        non_nan = vals_600519[~np.isnan(vals_600519)]
        assert len(np.unique(non_nan)) > 1, "All non-NaN values are identical, possible alignment bug"
    print("[PASS] Test 5: Time alignment (values change over time)")

print("\n" + "=" * 60)
print("All 5 engineering tests PASSED")
print("=" * 60)
