"""
RP_BP factor (Regression-Purified Book-to-Price)

Formula: OLS residual from cross-section regression each day:
  ln(|NetAsset| + 1) * sign(NetAsset) = alpha + beta * ln(MV) + eps
  factor = eps  (captures net asset premium relative to market size)

Data fields:
  - q_bs_toe_t    : total owner's equity (period-end)     lixinger.financial_statements
  - q_bs_etmsh_t  : minority shareholder equity (period-end), filled 0 if missing
  - mc            : total market cap (yuan)                lixinger.fundamental

  NetAsset = q_bs_toe_t - q_bs_etmsh_t
  NetAsset_ln = sign(NetAsset) * log(1 + |NetAsset|)
  MV_ln = log(mc)   [mc must be > 0]

Factor direction: positive (higher residual = undervalued book value given market size)
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

FACTOR_NAME = "RP_BP"
FACTOR_DIRECTION = 1  # positive: higher residual is better (undervalued)
MIN_VALID_STOCKS = 100  # minimum cross-section size for reliable OLS


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class RP_BP(FundamentalFactorCalculator):
    """
    Regression-Purified Book-to-Price (RP_BP).

    Each trading day, runs a cross-sectional OLS:
      NetAsset_ln_i = alpha + beta * MV_ln_i + eps_i

    where:
      NetAsset_ln = sign(NetAsset) * log(1 + |NetAsset|)   [signed log of net asset]
      MV_ln       = log(mc)                                 [log total market cap]
      eps         = OLS residual = the factor value

    Forward-fills quarterly NetAsset values from report_date (no future leakage).
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

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Calculate RP_BP daily panel via cross-section OLS residuals.

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
        # Step 1: Extract balance sheet columns and apply mainboard filter
        # ------------------------------------------------------------------
        needed = ["symbol", "stock_code", "date", "report_date",
                  "q_bs_toe_t", "q_bs_etmsh_t"]
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
        # Step 2: Compute NetAsset = total equity - minority equity
        # ------------------------------------------------------------------
        df["q_bs_etmsh_t"] = df["q_bs_etmsh_t"].fillna(0.0)
        df["NetAsset"] = df["q_bs_toe_t"] - df["q_bs_etmsh_t"]

        # Only keep records with valid toe and published by end_date
        df = df[
            df["q_bs_toe_t"].notna() &
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
        # Step 3: Forward-fill NetAsset from report_date to daily panel
        # ------------------------------------------------------------------
        pivot_na = df.pivot_table(
            index="symbol",
            columns="report_date",
            values="NetAsset",
            aggfunc="last",
        )
        all_dates_na = pivot_na.columns.union(trading_dates).sort_values()
        pivot_na = pivot_na.reindex(columns=all_dates_na)
        panel_na = pivot_na.ffill(axis=1).reindex(columns=trading_dates)

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
        all_symbols = sorted(set(panel_na.index) | set(mc_panel.index))
        panel_na = panel_na.reindex(index=all_symbols)    # (N, T) NetAsset
        mc_panel = mc_panel.reindex(index=all_symbols)    # (N, T) market cap

        na_arr = panel_na.values.astype(np.float64)   # (N, T)
        mc_arr = mc_panel.values.astype(np.float64)   # (N, T)
        N = len(all_symbols)
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 6: Cross-section OLS each trading day → residuals
        # ------------------------------------------------------------------
        factor_values = np.full((N, T), np.nan, dtype=np.float64)

        for t in range(T):
            net_asset = na_arr[:, t]
            mv = mc_arr[:, t]

            # Valid: both known AND mc > 0
            valid = ~np.isnan(net_asset) & ~np.isnan(mv) & (mv > 0)
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID_STOCKS:
                continue

            na_v = net_asset[valid]
            mv_v = mv[valid]

            # Signed log of net asset (handles negative net asset)
            na_ln = np.sign(na_v) * np.log1p(np.abs(na_v))
            mv_ln = np.log(mv_v)

            # OLS: na_ln = alpha + beta * mv_ln + eps
            X = np.column_stack([np.ones(n_valid), mv_ln])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, na_ln, rcond=None)
                residuals = na_ln - X @ coeffs
            except np.linalg.LinAlgError:
                continue

            factor_values[valid, t] = residuals

        symbols_arr = np.array(all_symbols)
        dates_arr = np.array(trading_dates, dtype='datetime64[ns]')

        nan_ratio = np.isnan(factor_values).mean() if factor_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"RP_BP NaN ratio is high: {nan_ratio:.1%}, please check data"
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
# Smoke test (run: python rp_bp.py)
# Uses full market with short date range to ensure cross-section OLS works.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RP_BP factor smoke test")
    print("=" * 60)

    # Use full market (no stock_codes) with 2-year window so the OLS
    # has enough cross-section stocks (>= MIN_VALID_STOCKS = 100).
    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute RP_BP factor")
    calculator = RP_BP()
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
