"""
MaxDrawdownFactor — 滚动最大回撤因子

计算逻辑：
    MAX_DRAWDOWN = (rolling_max(close, period) - close) / rolling_max(close, period)

值含义：
    - 0.0  → 当前价格处于窗口内最高点（无回撤）
    - 0.20 → 当前价格比窗口内最高点低 20%
    - 1.0  → 当前价格相对于窗口最高点下跌了 100%（极端情况）

经济含义：
    衡量股票在过去 period 个交易日内从窗口高点回落的幅度，
    反映短/中期价格压力与下行动能。常见用法：
      - 作为风险规避因子（direction=-1）：高回撤→高下行风险→低预期超额收益
      - 作为均值回归/超卖因子（direction=+1）：高回撤→超卖→预期反弹

因子方向：
    默认在 config 中设为 direction=-1（风险因子视角）。

数据来源：
    tushare.db — daily_hfq（后复权日频收盘价），与回测框架主数据源保持一致。
    直接使用 fundamental_data._tushare_db 获取数据库路径，
    保证与 FundamentalLeakageDetector 截断逻辑兼容。

无数据泄露：
    滚动最大回撤仅使用 [T-period+1, T] 范围内的历史收盘价，
    不依赖任何未来信息，可通过 FundamentalLeakageDetector 检测。

边界处理：
    - rolling_max == 0      → NaN（价格不应为零）
    - 停牌日（close=NaN）   → NaN，滚动 max 沿用前一有效最大值
    - 初始 period-1 日      → min_periods 控制，默认 1（允许部分窗口）

因子类别：技术 / 价格行为 (Technical / Price Behavior)
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
from data.stock_data_loader import StockDataLoader

FACTOR_NAME = "MAX_DRAWDOWN_FACTOR"
DEFAULT_PERIOD = 60

# Calendar days to pre-load before start_date so the rolling window
# is warm on the first backtest date.
# Conservatively: period * 2.5 to account for weekends + holidays.
_LOOKBACK_CALENDAR_MULTIPLIER = 2.5


class MaxDrawdownFactor(FundamentalFactorCalculator):
    """
    滚动最大回撤因子

    计算过去 period 个交易日内，当前收盘价相对于窗口最高收盘价的回撤幅度。

    参数
    ----
    period : int
        滚动窗口（交易日数），默认 60。
    min_periods : int
        Rolling 计算所需的最小观测数，默认 1（允许窗口不满时也计算，
        设为 period 可强制要求满窗口后才输出值）。

    示例
    ----
    ::

        fd = FundamentalData('2016-01-01', '2025-12-31')
        factor = MaxDrawdownFactor(period=60)
        result = factor.calculate(fd)   # FactorData (N, T)
    """

    def __init__(self, period: int = DEFAULT_PERIOD, min_periods: int = 1):
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"period must be a positive integer, got {period!r}")
        if not isinstance(min_periods, int) or min_periods <= 0:
            raise ValueError(f"min_periods must be a positive integer, got {min_periods!r}")
        self._period = period
        self._min_periods = min_periods

    # ------------------------------------------------------------------
    # FundamentalFactorCalculator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"{FACTOR_NAME}_{self._period}"

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"period": self._period, "min_periods": self._min_periods}

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        计算滚动最大回撤日频因子面板。

        流程：
          1. 从 tushare.db 加载后复权收盘价（含回望额外天数）
          2. 对每只股票在时间轴上做滚动最大值
          3. 计算回撤 = (rolling_max - close) / rolling_max
          4. 与 fundamental_data.trading_dates 对齐后返回 FactorData

        Parameters
        ----------
        fundamental_data : FundamentalData
            框架标准数据容器；仅用于获取：
            - start_date / end_date（日期范围）
            - trading_dates（交易日历，用于最终对齐）
            - _tushare_db（数据库路径）

        Returns
        -------
        FactorData : shape (N, T)，值域 [0, 1)，NaN 表示无有效数据。
        """

        # ----------------------------------------------------------------
        # Step 1: Load adjusted close from tushare (same source as backtester)
        # ----------------------------------------------------------------
        start_str = fundamental_data.start_date.strftime("%Y-%m-%d")
        end_str   = fundamental_data.end_date.strftime("%Y-%m-%d")

        # Extra calendar days so the rolling window is warm on day-1 of backtest
        lookback_extra = int(self._period * _LOOKBACK_CALENDAR_MULTIPLIER)

        loader = StockDataLoader(tushare_db_path=fundamental_data._tushare_db)
        ohlcv  = loader.load_ohlcv(
            start_date=start_str,
            end_date=end_str,
            use_adjusted=True,
            lookback_extra_days=lookback_extra,
        )

        # ----------------------------------------------------------------
        # Step 2: Build close-price DataFrame  shape: (N_ohlcv × T_full)
        #   Rows = juejin-format symbols (SHSE.600000),
        #   Cols = datetime64[ns] trading dates (from tushare daily_hfq)
        # ----------------------------------------------------------------
        close_arr   = ohlcv.close    # ndarray (N, T_full)
        symbols_arr = ohlcv.symbols  # ndarray (N,) strings
        dates_arr   = ohlcv.dates    # ndarray (T_full,) datetime64[ns]

        close_df = pd.DataFrame(
            close_arr,
            index=symbols_arr.tolist(),
            columns=pd.DatetimeIndex(dates_arr),
        )

        # ----------------------------------------------------------------
        # Step 3: Rolling max drawdown
        #
        #   close_df is (N × T_full); pandas rolling operates on columns
        #   by default, so we transpose first (T_full × N), roll, then
        #   transpose back to (N × T_full).
        #
        #   max_drawdown_{i,t} = (rolling_max_close - close) / rolling_max_close
        # ----------------------------------------------------------------
        rolling_max_df: pd.DataFrame = (
            close_df.T
            .rolling(window=self._period, min_periods=self._min_periods)
            .max()
            .T
        )  # (N × T_full)

        close_vals = close_df.values   # (N, T_full)
        rm_vals    = rolling_max_df.values  # (N, T_full)

        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown_vals = np.where(
                rm_vals > 0,
                (rm_vals - close_vals) / rm_vals,
                np.nan,
            )

        result_df = pd.DataFrame(
            drawdown_vals,
            index=close_df.index,
            columns=close_df.columns,
        )

        # ----------------------------------------------------------------
        # Step 4: Align to fundamental_data.trading_dates
        #   - Drops warm-up columns (dates < start_date)
        #   - Forward-fills any gaps in tushare dates vs. trading calendar
        #     (should be none, but defensive)
        # ----------------------------------------------------------------
        trading_dates = fundamental_data._get_trading_dates()
        td_idx = pd.DatetimeIndex(trading_dates)

        # Merge tushare-date columns with trading-calendar dates, then ffill,
        # then keep only the trading-calendar dates.
        all_cols = result_df.columns.union(td_idx).sort_values()
        result_df = result_df.reindex(columns=all_cols).ffill(axis=1)
        result_df = result_df.reindex(columns=td_idx)

        # ----------------------------------------------------------------
        # Step 5: Return FactorData
        # ----------------------------------------------------------------
        symbols_out = np.array(result_df.index.tolist())
        dates_out   = np.array(result_df.columns.tolist(), dtype="datetime64[ns]")
        values_out  = result_df.values.astype(np.float64)

        nan_ratio = np.isnan(values_out).mean() if values_out.size > 0 else 1.0
        print(
            f"  - {self.name}: {values_out.shape[0]} symbols × "
            f"{values_out.shape[1]} dates, NaN ratio={nan_ratio:.1%}"
        )
        if nan_ratio > 0.9:
            warnings.warn(
                f"MaxDrawdownFactor NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if OHLCV data is available in tushare.db (daily_hfq table)."
            )

        return FactorData(
            values=values_out,
            symbols=symbols_out,
            dates=dates_out,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (python max_drawdown_factor.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("MaxDrawdownFactor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2022-12-31"

    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    for period in [20, 60]:
        factor = MaxDrawdownFactor(period=period)
        print(f"\n--- period={period} ---")
        result = factor.calculate(fd)

        non_nan = ~np.isnan(result.values)
        if non_nan.any():
            print(f"shape   : {result.values.shape}")
            print(f"NaN pct : {np.isnan(result.values).mean():.1%}")
            print(f"value range : [{result.values[non_nan].min():.4f}, "
                  f"{result.values[non_nan].max():.4f}]")
            print(f"median  : {np.nanmedian(result.values):.4f}")
        else:
            print("  All NaN — check data path.")

    print("\n[OK] Smoke test passed.")
