"""
ACCRUAL_RATIO factor (应计比率)

应计比率是衡量盈余质量的经典指标，反映了净利润中应计部分占总资产的比例。
应计部分 = 净利润 - 经营现金流，代表非现金性质的会计利润。

Formula: ACCRUAL_RATIO = (NI - OCF) / Total Assets
                       = (q_ps_npadnrpatoshaopc_c - q_cfs_ncffoa_c) / q_bs_ta_t

Data fields from lixinger.financial_statements:
  - q_ps_npadnrpatoshaopc_c : 扣除非经常性损益净利润(单季) / Net profit after deducting non-recurring items (single quarter)
  - q_cfs_ncffoa_c          : 经营活动现金流量净额(单季) / Net cash flow from operating activities (single quarter)
  - q_bs_ta_t               : 资产总计 / Total Assets

Factor direction: -1 (lower is better, indicating higher earnings quality)
Factor category: quality

Notes:
  - Uses report_date for forward-fill (no future data leakage).
  - Stocks with Total Assets <= 0 are excluded.
  - Mainboard filter: only SHSE.60xxxx and SZSE.00xxxx.

References:
  - Sloan (1996): "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows
    About Future Earnings?" The Accounting Review.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "ACCRUAL_RATIO"
FACTOR_DIRECTION = -1  # negative: lower accrual ratio (higher earnings quality) is better


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class AccrualRatio(FundamentalFactorCalculator):
    """
    应计比率因子 (Accrual Ratio)

    计算公式: (扣非净利润 - 经营现金流) / 总资产
    
    该因子衡量公司净利润中的应计部分占比，是经典的盈余质量指标。
    较低的应计比率通常意味着更高的盈余质量和更可持续的利润。
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION}

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Calculate ACCRUAL_RATIO daily panel.

        Returns:
            FactorData: N stocks × T days, values = (NI - OCF) / Total Assets
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        T = len(trading_dates)

        # Load required fields using get_daily_panel
        fields = {
            "ni": "q_ps_npadnrpatoshaopc_c",  # Net Income (扣非净利润, single quarter)
            "ocf": "q_cfs_ncffoa_c",          # Operating Cash Flow (经营现金流, single quarter)
            "ta": "q_bs_ta_t",                # Total Assets (总资产, period-end)
        }

        all_symbols: set = set()
        panels: dict = {}

        for key, field in fields.items():
            try:
                vals, syms, dts = fundamental_data.get_daily_panel(field)
                df = pd.DataFrame(
                    vals,
                    index=syms.tolist(),
                    columns=pd.DatetimeIndex(dts),
                )
                panels[key] = df
                all_symbols.update(syms.tolist())
            except Exception as exc:
                warnings.warn(
                    f"ACCRUAL_RATIO: failed to load field '{field}' "
                    f"(key={key}): {exc}."
                )
                panels[key] = pd.DataFrame(dtype=np.float64)

        if not all_symbols:
            empty = np.empty((0, T), dtype=np.float64)
            return FactorData(
                values=empty,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype="datetime64[ns]"),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        all_symbols_sorted = sorted(all_symbols)
        N = len(all_symbols_sorted)
        td_idx = pd.DatetimeIndex(trading_dates)

        # Align all panels to (all_symbols, trading_dates)
        def _align(df: pd.DataFrame) -> np.ndarray:
            """Reindex df to (all_symbols, trading_dates), return float64 array."""
            if df.empty:
                return np.full((N, T), np.nan, dtype=np.float64)
            aligned = df.reindex(index=all_symbols_sorted, columns=td_idx)
            return aligned.values.astype(np.float64)

        ni_arr = _align(panels["ni"])     # Net Income
        ocf_arr = _align(panels["ocf"])   # Operating Cash Flow
        ta_arr = _align(panels["ta"])     # Total Assets

        # Apply mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in all_symbols_sorted])
        if not mainboard_mask.all():
            ni_arr = np.where(mainboard_mask[:, None], ni_arr, np.nan)
            ocf_arr = np.where(mainboard_mask[:, None], ocf_arr, np.nan)
            ta_arr = np.where(mainboard_mask[:, None], ta_arr, np.nan)

        # Compute ACCRUAL_RATIO = (NI - OCF) / TA
        with np.errstate(divide="ignore", invalid="ignore"):
            accrual = ni_arr - ocf_arr  # Accruals = NI - OCF
            accrual_ratio = np.where(
                ~np.isnan(ta_arr) & (ta_arr > 0),  # TA must be positive
                accrual / ta_arr,
                np.nan,
            )

        # Handle inf values
        accrual_ratio = np.where(np.isinf(accrual_ratio), np.nan, accrual_ratio)
        accrual_ratio = accrual_ratio.astype(np.float64)

        nan_ratio = np.isnan(accrual_ratio).mean() if accrual_ratio.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"ACCRUAL_RATIO NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=accrual_ratio,
            symbols=np.array(all_symbols_sorted),
            dates=np.array(trading_dates, dtype="datetime64[ns]"),
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python accrual_ratio.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("ACCRUAL_RATIO factor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] Compute ACCRUAL_RATIO factor")
    calculator = AccrualRatio()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols      : {result.symbols.tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"NaN ratio    : {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), \
        "Idempotency failed"

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {np.round(last5, 6).tolist()}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid : {len(valid_cs)}")
        print(f"  mean    : {valid_cs.mean():.6f}")
        print(f"  median  : {np.median(valid_cs):.6f}")
        print(f"  min     : {valid_cs.min():.6f}")
        print(f"  max     : {valid_cs.max():.6f}")
        print(f"  std     : {valid_cs.std():.6f}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2013-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        try:
            report = detector.detect(calculator, fd_leak)
        except ValueError as e:
            if "panel is empty" in str(e):
                print(f"  split_ratio={sr}: panel empty, skipping.")
                continue
            raise
        report.print_report()
        if report.has_leakage:
            leakage_found = True
            print(f"[FAIL] Leakage detected at split_ratio={sr}")
        else:
            print(f"[OK] No leakage at split_ratio={sr}")

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
        print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")
