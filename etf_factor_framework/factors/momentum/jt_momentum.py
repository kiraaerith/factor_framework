"""
Jegadeesh & Titman (1993) Cross-sectional Momentum Factor.

Formation period J months → skip 1 month → holding period K months (SMA approximation).

Each day t:
  1. raw_mom[t] = close[t - skip] / close[t - skip - J] - 1
     (cumulative return over J months, ending 1 month ago)
  2. factor[t] = SMA_K(raw_mom)[t]
     (K-month SMA approximates overlapping portfolio)

Parameters (in trading days, 1 month ≈ 21 days):
  formation_days : J months in trading days
  holding_days   : K months in trading days (SMA window)
  skip_days      : skip period in trading days (default 21 = 1 month)

Factor direction: +1 (higher momentum = better)
Data source: tushare daily_hfq (adjusted close)
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator
from factors.fundamental.pricevol_data import PriceVolData
from data.stock_data_loader import StockDataLoader

logger = logging.getLogger(__name__)


class JTMomentum(FundamentalFactorCalculator):
    """Parameterized JT Momentum factor.

    Subclasses set FORMATION_DAYS, HOLDING_DAYS, SKIP_DAYS.
    """

    FACTOR_NAME: str = "JT_MOM"
    FORMATION_DAYS: int = 126   # J months in trading days
    HOLDING_DAYS: int = 126     # K months in trading days (SMA window)
    SKIP_DAYS: int = 21         # skip 1 month

    @property
    def name(self) -> str:
        return self.FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return self.FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "formation_days": self.FORMATION_DAYS,
            "holding_days": self.HOLDING_DAYS,
            "skip_days": self.SKIP_DAYS,
            "direction": 1,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        # Total lookback needed: skip + formation + holding (for SMA)
        total_lookback = self.SKIP_DAYS + self.FORMATION_DAYS + self.HOLDING_DAYS
        extra_days = total_lookback + 30  # buffer for trading day gaps

        logger.info(f"{self.FACTOR_NAME}: loading OHLCV with {extra_days} extra lookback days")

        if pricevol_data is not None:
            ohlcv = pricevol_data.get_ohlcv(lookback_extra_days=extra_days)
        else:
            # Fallback: direct load (leakage detection will NOT work in this path)
            start_date = fundamental_data.start_date
            end_date = fundamental_data.end_date
            logger.warning(
                f"{self.FACTOR_NAME}: pricevol_data not provided, "
                f"falling back to direct StockDataLoader (leakage detection disabled)"
            )
            loader = StockDataLoader()
            ohlcv = loader.load_ohlcv(
                start_date=str(start_date.date()),
                end_date=str(end_date.date()),
                use_adjusted=True,
                lookback_extra_days=extra_days,
            )

        close = ohlcv.close      # (N, T_full)
        symbols = ohlcv.symbols  # (N,)
        dates = ohlcv.dates      # (T_full,)
        N, T_full = close.shape

        logger.info(f"{self.FACTOR_NAME}: close shape ({N}, {T_full}), "
                    f"J={self.FORMATION_DAYS}, K={self.HOLDING_DAYS}, skip={self.SKIP_DAYS}")

        # Step 1: raw momentum = close[t - skip] / close[t - skip - formation] - 1
        # Vectorized shift via slicing
        skip = self.SKIP_DAYS
        form = self.FORMATION_DAYS
        hold = self.HOLDING_DAYS

        raw_mom = np.full((N, T_full), np.nan)
        # close_recent = close at (t - skip), close_past = close at (t - skip - form)
        # For index t: recent = t - skip, past = t - skip - form
        # Valid range: t >= skip + form
        start_idx = skip + form
        if start_idx < T_full:
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_mom[:, start_idx:] = (
                    close[:, start_idx - skip: T_full - skip]
                    / close[:, start_idx - skip - form: T_full - skip - form]
                    - 1.0
                )

        # Step 2: SMA over holding_days (approximates overlapping portfolio)
        # Vectorized cumsum-based SMA
        factor_values = np.full((N, T_full), np.nan)

        if hold <= 1:
            factor_values = raw_mom
        else:
            # Use cumsum trick for SMA, handle NaN by treating as 0
            # For each stock, compute rolling mean of raw_mom over `hold` days
            # Replace NaN with 0 for cumsum, track valid counts
            not_nan = (~np.isnan(raw_mom)).astype(np.float64)
            raw_filled = np.where(np.isnan(raw_mom), 0.0, raw_mom)

            # Cumulative sums along time axis
            cum_val = np.cumsum(raw_filled, axis=1)   # (N, T_full)
            cum_cnt = np.cumsum(not_nan, axis=1)       # (N, T_full)

            # SMA[t] = (cum_val[t] - cum_val[t-hold]) / (cum_cnt[t] - cum_cnt[t-hold])
            # Valid when t >= hold - 1
            sma_start = hold
            if sma_start < T_full:
                window_sum = cum_val[:, sma_start:] - cum_val[:, :T_full - sma_start]
                window_cnt = cum_cnt[:, sma_start:] - cum_cnt[:, :T_full - sma_start]

                # Require at least half the window to be valid
                min_cnt = hold / 2.0
                valid = window_cnt >= min_cnt
                factor_values[:, sma_start:] = np.where(
                    valid,
                    window_sum / window_cnt,
                    np.nan,
                )

        # Trim to requested date range
        target_start = np.datetime64(fundamental_data.start_date)
        date_mask = dates >= target_start
        if not date_mask.any():
            raise ValueError(f"{self.FACTOR_NAME}: no dates in target range")

        trim_start = np.argmax(date_mask)
        factor_trimmed = factor_values[:, trim_start:]
        dates_trimmed = dates[trim_start:]

        # Mainboard filter (SHSE.60xxxx, SZSE.00xxxx)
        mainboard_mask = np.array([
            s.startswith('SHSE.60') or s.startswith('SZSE.00')
            for s in symbols
        ])
        factor_trimmed = factor_trimmed[mainboard_mask]
        symbols_trimmed = symbols[mainboard_mask]

        nan_ratio = np.isnan(factor_trimmed).mean()
        logger.info(f"{self.FACTOR_NAME}: output ({len(symbols_trimmed)}, {len(dates_trimmed)}), "
                    f"nan_ratio={nan_ratio:.1%}")

        return FactorData(
            values=factor_trimmed,
            symbols=symbols_trimmed,
            dates=dates_trimmed,
            name=self.name,
            params=self.params,
        )


# ======================================================================
# 6 concrete configurations: J × K = {3,6,12} × {3,6} months
# ======================================================================

class JT_MOM_3_3(JTMomentum):
    """J=3m, K=3m, skip=1m"""
    FACTOR_NAME = "JT_MOM_3_3"
    FORMATION_DAYS = 63    # 3 months
    HOLDING_DAYS = 63      # 3 months
    SKIP_DAYS = 21


class JT_MOM_3_6(JTMomentum):
    """J=3m, K=6m, skip=1m"""
    FACTOR_NAME = "JT_MOM_3_6"
    FORMATION_DAYS = 63
    HOLDING_DAYS = 126
    SKIP_DAYS = 21


class JT_MOM_6_3(JTMomentum):
    """J=6m, K=3m, skip=1m"""
    FACTOR_NAME = "JT_MOM_6_3"
    FORMATION_DAYS = 126
    HOLDING_DAYS = 63
    SKIP_DAYS = 21


class JT_MOM_6_6(JTMomentum):
    """J=6m, K=6m, skip=1m"""
    FACTOR_NAME = "JT_MOM_6_6"
    FORMATION_DAYS = 126
    HOLDING_DAYS = 126
    SKIP_DAYS = 21


class JT_MOM_12_3(JTMomentum):
    """J=12m, K=3m, skip=1m"""
    FACTOR_NAME = "JT_MOM_12_3"
    FORMATION_DAYS = 252
    HOLDING_DAYS = 63
    SKIP_DAYS = 21


class JT_MOM_12_6(JTMomentum):
    """J=12m, K=6m, skip=1m"""
    FACTOR_NAME = "JT_MOM_12_6"
    FORMATION_DAYS = 252
    HOLDING_DAYS = 126
    SKIP_DAYS = 21
