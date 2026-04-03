"""
RP_EP factor (Regression-Purified Earnings-to-Price)

Formula: OLS residual from cross-section regression each day:
  ln(1 + |QuartNP|) * sign(QuartNP) = alpha + beta * ln(MV) + eps
  factor = eps  (captures single-quarter earnings premium relative to market size)

Data fields:
  - q_ps_npatoshopc_c : single-quarter net profit attributable to parent shareholders
                        lixinger.financial_statements (quarterly, forward-filled to daily)
  - mc               : total market cap                lixinger.fundamental (daily)

Factor direction: positive (higher residual = undervalued earnings given market size)
Factor category: value - regression-purified
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

FACTOR_NAME = "RP_EP"
FACTOR_DIRECTION = 1  # positive: higher residual is better (undervalued earnings)
MIN_VALID_STOCKS = 100  # minimum cross-section size for reliable OLS


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class RP_EP(FundamentalFactorCalculator):
    """
    Regression-Purified Earnings-to-Price (RP_EP).

    Each trading day, runs a cross-sectional OLS:
      QuartNP_ln_i = alpha + beta * MV_ln_i + eps_i

    where:
      QuartNP_ln = sign(QuartNP) * log(1 + |QuartNP|)   [signed log of single-quarter NP]
      MV_ln      = log(mc)                                [log total market cap]
      eps        = OLS residual = the factor value

    Forward-fills quarterly q_ps_npatoshopc_c values from report_date (no future leakage).
    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    Cross-sections with fewer than MIN_VALID_STOCKS are set to NaN.
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
        Calculate RP_EP daily panel via cross-section OLS residuals.

        Returns:
            FactorData: N stocks x T days, values = OLS residuals (float64)
        """
        # Ensure raw data is loaded
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        raw = fundamental_data._raw_data

        if raw is None or raw.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ------------------------------------------------------------------
        # Step 1: Extract financial column and apply mainboard filter
        # ------------------------------------------------------------------
        needed = ["symbol", "stock_code", "date", "report_date", "q_ps_npatoshopc_c"]
        df = raw[[c for c in needed if c in raw.columns]].copy()

        mainboard_mask = np.array([_is_mainboard(s) for s in df["symbol"]])
        df = df[mainboard_mask].copy()

        if df.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ------------------------------------------------------------------
        # Step 2: Keep records with valid QuartNP and within end_date
        # ------------------------------------------------------------------
        df = df[
            df["q_ps_npatoshopc_c"].notna() &
            (df["report_date"] <= fundamental_data.end_date)
        ].copy()

        if df.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ------------------------------------------------------------------
        # Step 3: Forward-fill QuartNP from report_date to daily panel
        # ------------------------------------------------------------------
        pivot_np = df.pivot_table(
            index="symbol",
            columns="report_date",
            values="q_ps_npatoshopc_c",
            aggfunc="last",
        )
        all_dates_np = pivot_np.columns.union(trading_dates).sort_values()
        pivot_np = pivot_np.reindex(columns=all_dates_np)
        panel_np = pivot_np.ffill(axis=1).reindex(columns=trading_dates)

        # ------------------------------------------------------------------
        # Step 4: Load daily market cap panel and apply mainboard filter
        # ------------------------------------------------------------------
        mc_values, mc_symbols, mc_dates = fundamental_data.get_market_cap_panel()
        mc_symbols = np.array(mc_symbols)
        mc_dates = pd.DatetimeIndex(mc_dates)

        mb_mask = np.array([_is_mainboard(s) for s in mc_symbols])
        mc_values = mc_values[mb_mask]
        mc_symbols_mb = mc_symbols[mb_mask]

        mc_panel = pd.DataFrame(mc_values, index=mc_symbols_mb,
                                columns=mc_dates).reindex(columns=trading_dates)

        # ------------------------------------------------------------------
        # Step 5: Align both panels to a unified symbol index
        # ------------------------------------------------------------------
        all_symbols = sorted(set(panel_np.index) | set(mc_panel.index))
        panel_np = panel_np.reindex(index=all_symbols)   # (N, T) QuartNP
        mc_panel = mc_panel.reindex(index=all_symbols)   # (N, T) market cap

        np_arr = panel_np.values.astype(np.float64)   # (N, T)
        mc_arr = mc_panel.values.astype(np.float64)   # (N, T)
        N = len(all_symbols)
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 6: Cross-section OLS each trading day → residuals
        # ------------------------------------------------------------------
        factor_values = np.full((N, T), np.nan, dtype=np.float64)

        for t in range(T):
            quart_np = np_arr[:, t]
            mv = mc_arr[:, t]

            # Valid: both known AND mc > 0
            valid = ~np.isnan(quart_np) & ~np.isnan(mv) & (mv > 0)
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID_STOCKS:
                continue

            qnp_v = quart_np[valid]
            mv_v = mv[valid]

            # Signed log of single-quarter net profit (handles losses)
            qnp_ln = np.sign(qnp_v) * np.log1p(np.abs(qnp_v))
            mv_ln = np.log(mv_v)

            # OLS: qnp_ln = alpha + beta * mv_ln + eps
            X = np.column_stack([np.ones(n_valid), mv_ln])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, qnp_ln, rcond=None)
                residuals = qnp_ln - X @ coeffs
            except np.linalg.LinAlgError:
                continue

            factor_values[valid, t] = residuals

        symbols_arr = np.array(all_symbols)
        dates_arr = np.array(trading_dates, dtype='datetime64[ns]')

        nan_ratio = np.isnan(factor_values).mean() if factor_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"RP_EP NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=factor_values,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python rp_ep.py)
# Uses full market with short date range to ensure cross-section OLS works.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RP_EP factor smoke test")
    print("=" * 60)

    # Use full market (no stock_codes) so OLS has enough cross-section stocks.
    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute RP_EP factor")
    calculator = RP_EP()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
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
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid    : {len(valid_cs)}")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")

    print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calculator, fd_leak)
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
