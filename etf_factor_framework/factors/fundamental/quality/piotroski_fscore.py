"""
Piotroski F-Score Factor

基于 Piotroski (2000) 论文的 9 个二元信号综合评分，满分 9 分，越高越好。

信号分组：
  盈利维度 (P1-P4):
    P1: ROA > 0              → q_m_roa_t > 0
    P2: OCF > 0              → q_cfs_ncffoa_c > 0
    P3: ROA YoY 提升         → q_m_roa_t > lag(q_m_roa_t)
    P4: OCF > 净利润         → q_m_ncffoa_np_r_t > 1

  杠杆/流动性维度 (P5-P7):
    P5: 长期借款比率 YoY 下降  → q_bs_ltl_t/q_bs_ta_t < lag
        注：q_bs_ltl_t = 长期借款（非长期负债合计），为长期有息债务的代理指标，
            与 Piotroski 原文的 long-term debt 含义一致。
    P6: 流动比率 YoY 提升     → q_m_c_r_t > lag(q_m_c_r_t)
    P7: 无股本稀释（代理）    → 隐含股数 = q_ps_np_c / q_ps_beps_c，YoY 未增加
        注：lixinger financial_statements 无直接股本数量字段。
            用（净利润 / 基本EPS）作为流通股数的代理，YoY 未增加则视为无稀释。
            当净利润或EPS为负时该信号置 NaN（不纳入得分计算）。

  效率维度 (P8-P9):
    P8: 毛利率 YoY 提升      → q_ps_gp_m_t > lag(q_ps_gp_m_t)
    P9: 资产周转率 YoY 提升  → q_m_ta_to_t > lag(q_m_ta_to_t)

因子方向：+1（F-Score 越高越好）
因子类别：质量 (Quality)

YoY 滞后窗口：默认 250 个交易日（约 1 年）。
最少有效信号数：当可用信号 < min_valid_signals 时，该股当日因子值为 NaN。
"""

import os
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

FACTOR_NAME = "PIOTROSKI_FSCORE"
FACTOR_DIRECTION = 1
MC_MIN_BILLION = 50.0        # 市值下限（亿元），低于此值的股票当日因子置 NaN


_DEFAULT_LAG_DAYS = 250      # ~1 trading year for YoY comparison
_DEFAULT_MIN_SIGNALS = 5     # require at least 5 of 9 signals to have valid data


# Field mapping for each Piotroski component
_FIELDS = {
    "roa":    "q_m_roa_t",           # P1, P3
    "ocf":    "q_cfs_ncffoa_c",      # P2
    "ocf_np": "q_m_ncffoa_np_r_t",   # P4 (OCF/NI ratio, TTM)
    "ltl":    "q_bs_ltl_t",          # P5 (长期借款, proxy for long-term debt)
    "ta":     "q_bs_ta_t",           # P5 (total assets)
    "cr":     "q_m_c_r_t",           # P6 (current ratio)
    "np":     "q_ps_np_c",           # P7 proxy (single-quarter net profit)
    "beps":   "q_ps_beps_c",         # P7 proxy (basic EPS)
    "gpm":    "q_ps_gp_m_t",         # P8 (gross profit margin)
    "ato":    "q_m_ta_to_t",         # P9 (asset turnover)
}


class PiotroskiFScore(FundamentalFactorCalculator):
    """
    Piotroski F-Score 因子

    参数
    ----
    lag_days : int
        YoY 比较的滞后交易日数，默认 250（约 1 年）。
    min_valid_signals : int
        要求至少有多少个信号有有效数据，低于此值的股票当日 F-Score 为 NaN，
        默认 5（9 个信号中多数可用）。
    """

    def __init__(self, lag_days: int = _DEFAULT_LAG_DAYS,
                 min_valid_signals: int = _DEFAULT_MIN_SIGNALS):
        self._lag_days = lag_days
        self._min_signals = min_valid_signals

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "direction": FACTOR_DIRECTION,
            "lag_days": self._lag_days,
            "min_valid_signals": self._min_signals,
        }

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 Piotroski F-Score 日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 0~9 的整数或 NaN（数据不足时）。
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 1: Load all component panels as DataFrames
        # ------------------------------------------------------------------
        all_symbols: set = set()
        raw_panels: dict = {}

        for key, field in _FIELDS.items():
            try:
                vals, syms, dts = fundamental_data.get_daily_panel(field)
                df = pd.DataFrame(
                    vals,
                    index=syms.tolist(),
                    columns=pd.DatetimeIndex(dts),
                )
                raw_panels[key] = df
                all_symbols.update(syms.tolist())
            except Exception as exc:
                warnings.warn(
                    f"PiotroskiFScore: failed to load field '{field}' "
                    f"(key={key}): {exc}. This signal will be NaN."
                )
                raw_panels[key] = pd.DataFrame(dtype=np.float64)

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
        if not all_symbols_sorted:
            empty = np.empty((0, T), dtype=np.float64)
            return FactorData(
                values=empty,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype="datetime64[ns]"),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )
        N = len(all_symbols_sorted)
        td_idx = pd.DatetimeIndex(trading_dates)

        # ------------------------------------------------------------------
        # Step 2: Align all panels to (all_symbols, trading_dates)
        # ------------------------------------------------------------------
        def _align(df: pd.DataFrame) -> np.ndarray:
            """Reindex df to (all_symbols, trading_dates), return float64 array."""
            if df.empty:
                return np.full((N, T), np.nan, dtype=np.float64)
            aligned = df.reindex(index=all_symbols_sorted, columns=td_idx)
            return aligned.values.astype(np.float64)

        roa_arr    = _align(raw_panels["roa"])    # (N, T)
        ocf_arr    = _align(raw_panels["ocf"])
        ocf_np_arr = _align(raw_panels["ocf_np"])
        ltl_arr    = _align(raw_panels["ltl"])
        ta_arr     = _align(raw_panels["ta"])
        cr_arr     = _align(raw_panels["cr"])
        np_arr     = _align(raw_panels["np"])
        beps_arr   = _align(raw_panels["beps"])
        gpm_arr    = _align(raw_panels["gpm"])
        ato_arr    = _align(raw_panels["ato"])

        lag = self._lag_days

        # ------------------------------------------------------------------
        # Step 3: Helper — YoY direction signals
        # ------------------------------------------------------------------
        def _yoy_increased(arr: np.ndarray) -> np.ndarray:
            """Returns 1 where arr[t] > arr[t-lag], NaN where data missing."""
            result = np.full((N, T), np.nan, dtype=np.float64)
            if T > lag:
                curr = arr[:, lag:]
                prev = arr[:, :-lag]
                valid = ~np.isnan(curr) & ~np.isnan(prev)
                result[:, lag:] = np.where(valid, (curr > prev).astype(np.float64), np.nan)
            return result

        def _yoy_decreased(arr: np.ndarray) -> np.ndarray:
            """Returns 1 where arr[t] < arr[t-lag], NaN where data missing."""
            result = np.full((N, T), np.nan, dtype=np.float64)
            if T > lag:
                curr = arr[:, lag:]
                prev = arr[:, :-lag]
                valid = ~np.isnan(curr) & ~np.isnan(prev)
                result[:, lag:] = np.where(valid, (curr < prev).astype(np.float64), np.nan)
            return result

        # ------------------------------------------------------------------
        # Step 4: Compute 9 binary signals
        # ------------------------------------------------------------------

        # P1: ROA > 0
        p1 = np.where(~np.isnan(roa_arr), (roa_arr > 0).astype(np.float64), np.nan)

        # P2: OCF > 0 (single-quarter operating cash flow)
        p2 = np.where(~np.isnan(ocf_arr), (ocf_arr > 0).astype(np.float64), np.nan)

        # P3: ROA YoY increased
        p3 = _yoy_increased(roa_arr)

        # P4: OCF > Net Income (OCF/NI ratio > 1)
        p4 = np.where(~np.isnan(ocf_np_arr), (ocf_np_arr > 1).astype(np.float64), np.nan)

        # P5: Long-term borrowings ratio (q_bs_ltl_t / q_bs_ta_t) YoY decreased
        # q_bs_ltl_t = 长期借款 (long-term bank borrowings), not total long-term liabilities.
        # Used as a proxy for long-term debt, consistent with Piotroski (2000).
        with np.errstate(divide="ignore", invalid="ignore"):
            lev_arr = np.where(
                ~np.isnan(ltl_arr) & ~np.isnan(ta_arr) & (ta_arr > 0),
                ltl_arr / ta_arr,
                np.nan,
            )
        p5 = _yoy_decreased(lev_arr)

        # P6: Current ratio YoY increased
        p6 = _yoy_increased(cr_arr)

        # P7: No share dilution (proxy via implied share count = NP / basic EPS)
        # Signal is valid only when both NP > 0 and EPS > 0 (loss cases excluded).
        # P7 = 1 if implied share count did NOT increase YoY.
        with np.errstate(divide="ignore", invalid="ignore"):
            share_proxy = np.where(
                ~np.isnan(np_arr) & ~np.isnan(beps_arr)
                & (np_arr > 0) & (beps_arr > 0),
                np_arr / beps_arr,
                np.nan,
            )
        p7 = _yoy_decreased(share_proxy)  # shares didn't grow = no dilution

        # P8: Gross margin YoY increased
        p8 = _yoy_increased(gpm_arr)

        # P9: Asset turnover YoY increased
        p9 = _yoy_increased(ato_arr)

        # ------------------------------------------------------------------
        # Step 5: Aggregate — nansum, require min valid signals
        # ------------------------------------------------------------------
        signals = np.stack([p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=0)  # (9, N, T)
        valid_count = (~np.isnan(signals)).sum(axis=0)   # (N, T)
        signal_sum  = np.nansum(signals, axis=0)         # (N, T)

        f_score = np.where(
            valid_count >= self._min_signals,
            signal_sum,
            np.nan,
        ).astype(np.float64)

        # ------------------------------------------------------------------
        # Filter: daily market-cap floor — mask out stocks with mc unknown
        # or mc < MC_MIN_BILLION.  mc from lixinger is in 亿元.
        # ------------------------------------------------------------------
        mc_vals, mc_syms, _ = fundamental_data.get_market_cap_panel()
        mc_sym_idx = {s: i for i, s in enumerate(mc_syms.tolist())}
        mc_aligned = np.full((N, T), np.nan, dtype=np.float64)
        for i, sym in enumerate(all_symbols_sorted):
            j = mc_sym_idx.get(sym)
            if j is not None:
                mc_aligned[i, :] = mc_vals[j, :]

        # Where mc is unknown OR below threshold → set factor to NaN
        # 必须同时过滤 NaN（无市值数据的股票），否则它们在 size 中性化时
        # 无法参与 OLS 回归，原始值直接透传，会以异常高分霸占 top-K。
        invalid_mc = np.isnan(mc_aligned) | (mc_aligned < MC_MIN_BILLION)
        f_score[invalid_mc] = np.nan

        n_mc_masked = int(invalid_mc.sum())
        if n_mc_masked > 0:
            n_no_mc = int(np.isnan(mc_aligned).sum())
            n_below = n_mc_masked - n_no_mc
            print(f"  - 市值过滤: 无市值数据 {n_no_mc} 个股-日, <{MC_MIN_BILLION:.0f}亿 {n_below} 个股-日")

        nan_ratio = np.isnan(f_score).mean() if f_score.size > 0 else 1.0
        if nan_ratio > 0.9:
            warnings.warn(
                f"PiotroskiFScore NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if required fields exist in the lixinger database."
            )

        return FactorData(
            values=f_score,
            symbols=np.array(all_symbols_sorted),
            dates=np.array(trading_dates, dtype="datetime64[ns]"),
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (python piotroski_fscore.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("PiotroskiFScore factor smoke test")
    print("=" * 60)

    TEST_START = "2019-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)

    print("\n[Step 2] Compute PiotroskiFScore factor")
    calculator = PiotroskiFScore()
    result = calculator.calculate(fd)

    print(f"\nFactor shape  : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range    : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio     : {nan_ratio:.1%}")
    assert nan_ratio < 0.95, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf"
    assert valid.min() >= 0 and valid.max() <= 9, f"Score out of [0,9]: [{valid.min()}, {valid.max()}]"

    last_cs = result.values[:, -1]
    last_cs = last_cs[~np.isnan(last_cs)]
    print(f"\nLast cross-section stats ({len(last_cs)} stocks):")
    print(f"  mean : {last_cs.mean():.2f}")
    print(f"  std  : {last_cs.std():.2f}")
    print(f"  min  : {last_cs.min():.0f}")
    print(f"  max  : {last_cs.max():.0f}")
    score_dist = {int(s): int((last_cs == s).sum()) for s in range(10)}
    print(f"  dist : {score_dist}")

    print("\n[Step 3] Idempotency check")
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"
    print("  [OK] Idempotent")

    print("\n[Step 4] Leakage detection")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31")
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
            print(f"[FAIL] Leakage at split_ratio={sr}")
        else:
            print(f"[OK] No leakage at split_ratio={sr}")

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
        print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")
