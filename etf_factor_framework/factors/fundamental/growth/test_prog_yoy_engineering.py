"""
Engineering tests for ProG_YoY factor.
Run: python factors/fundamental/growth/test_prog_yoy_engineering.py
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
from factors.fundamental.growth.prog_yoy import ProG_YoY

LIXINGER_DB = r"E:\code_project_v2\china_stock_data\lixinger.db"

TEST_START = "2022-01-01"
TEST_END   = "2024-12-31"
TEST_CODES = ["600519", "000858", "601318", "600036"]

print("=" * 60)
print("ProG_YoY Engineering Tests")
print("=" * 60)

print(f"\n[Setup] Loading FundamentalData ({TEST_START} ~ {TEST_END})")
fd = FundamentalData(
    start_date=TEST_START,
    end_date=TEST_END,
    stock_codes=TEST_CODES,
)
calculator = ProG_YoY()
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
assert nan_ratio < 0.9, f"NaN ratio too high: {nan_ratio:.1%}"
print(f"  nan_ratio={nan_ratio:.1%}")
print(f"[PASS] Test 3: NaN boundary (nan_ratio={nan_ratio:.1%})")

# ----------------------------------------------------------------
# Test 4: Golden Test
# For 600519, quarter ending 2023-09-30, released 2023-10-21
# expected q_ps_np_c_y2y = 0.1503 (from DB query)
# ----------------------------------------------------------------
print("\n[Test 4] Golden Test")
conn = sqlite3.connect(LIXINGER_DB)
df_gold = pd.read_sql(
    "SELECT date, report_date, q_ps_np_c_y2y FROM financial_statements "
    "WHERE stock_code = '600519' AND date LIKE '2023-09-30%'",
    conn
)
conn.close()

print(f"  Raw DB row: {df_gold.to_dict('records')}")
assert len(df_gold) > 0, "No row found for 600519 2023-09-30"
expected_val = float(df_gold['q_ps_np_c_y2y'].iloc[0])
release_date_str = pd.Timestamp(df_gold['report_date'].iloc[0]).strftime('%Y-%m-%d')
print(f"  Expected q_ps_np_c_y2y = {expected_val}, released on {release_date_str}")

# Build a small fd range after release date so the new quarter is used
fd_gold = FundamentalData(
    start_date="2023-01-01",
    end_date="2023-12-31",
    stock_codes=["600519"],
)
result_gold = calculator.calculate(fd_gold)

sym_idx = np.where(result_gold.symbols == "SHSE.600519")[0][0]
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
# Release date of 2023-09-30 report: 2023-10-21
# Before release: should use Q2 2023-06-30 data (0.2018)
# On/after release: should use Q3 2023-09-30 data (0.1503)
# ----------------------------------------------------------------
print("\n[Test 5] Time alignment")
conn = sqlite3.connect(LIXINGER_DB)
df_align = pd.read_sql(
    "SELECT date, report_date, q_ps_np_c_y2y FROM financial_statements "
    "WHERE stock_code = '600519' AND date IN "
    "(SELECT date FROM financial_statements WHERE stock_code = '600519' "
    " ORDER BY date DESC LIMIT 12) "
    "ORDER BY date DESC",
    conn
)
conn.close()

print(f"  Recent quarters for 600519:")
for _, row in df_align.iterrows():
    print(f"    date={row['date'][:10]}, report_date={row['report_date'][:10]}, "
          f"q_ps_np_c_y2y={row['q_ps_np_c_y2y']:.4f}")

# Q3 2023: released 2023-10-21
# Q2 2023: date=2023-06-30, released 2023-08-03
release_q3 = pd.Timestamp("2023-10-21")
expected_before = float(df_align[df_align['date'].str.startswith('2023-06-30')]['q_ps_np_c_y2y'].iloc[0])
expected_after  = float(df_align[df_align['date'].str.startswith('2023-09-30')]['q_ps_np_c_y2y'].iloc[0])
print(f"  Q2 2023 value (before release of Q3): {expected_before:.4f}")
print(f"  Q3 2023 value (after release of Q3):  {expected_after:.4f}")

fd_align = FundamentalData(
    start_date="2023-01-01",
    end_date="2023-12-31",
    stock_codes=["600519"],
)
result_align = calculator.calculate(fd_align)
dates_align = pd.DatetimeIndex(result_align.dates)
sym_align_idx = np.where(result_align.symbols == "SHSE.600519")[0][0]

# Day before release
before_idx = dates_align.searchsorted(release_q3, side='left') - 1
val_before = result_align.values[sym_align_idx, before_idx]
print(f"  Factor value on {dates_align[before_idx].date()} (before release): {val_before:.4f}")

# Day of release (or first trading day after)
after_idx2 = dates_align.searchsorted(release_q3, side='left')
val_after = result_align.values[sym_align_idx, after_idx2]
print(f"  Factor value on {dates_align[after_idx2].date()} (after release):  {val_after:.4f}")

assert abs(val_before - expected_before) < 1e-6, \
    f"Before-release value mismatch: got {val_before:.6f}, expected {expected_before:.6f}"
assert abs(val_after - expected_after) < 1e-6, \
    f"After-release value mismatch: got {val_after:.6f}, expected {expected_after:.6f}"
assert val_before != val_after or np.isnan(val_before), \
    f"Time alignment FAILED: factor value did not change at release date"
print(f"[PASS] Test 5: Time alignment (before={val_before:.4f}, after={val_after:.4f})")

print("\n" + "=" * 60)
print("All 5 engineering tests PASSED")
print("=" * 60)
