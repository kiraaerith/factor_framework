"""
交易上下文模块

封装A股交易约束（停牌、涨跌停、新股过滤、退市），提供统一的约束接口供评估器使用。

主要功能：
  - 停牌掩码：is_suspended
  - 涨停掩码：is_limit_up（开盘价≈涨停价时）
  - 跌停掩码：is_limit_down（开盘价≈跌停价时）
  - 新股过滤：is_new_stock（上市不足 N 天的股票）
  - 退市掩码：is_delisted（当前日期 >= 退市日期）
  - 综合可买入/卖出掩码：can_buy / can_sell
"""

from typing import Optional
import numpy as np
import pandas as pd


class TradeContext:
    """
    A股交易上下文

    封装停牌、涨跌停、新股、退市等交易约束，为回测框架提供统一接口。
    内部使用 numpy ndarray 存储，消除 pandas 元数据开销。

    Attributes:
        is_suspended: ndarray (N, T) bool，停牌掩码
        is_limit_up:  ndarray (N, T) bool，涨停掩码（开盘涨停，无法买入）
        is_limit_down:ndarray (N, T) bool，跌停掩码（开盘跌停，无法卖出）
        is_new_stock: ndarray (N, T) bool，新股掩码（上市不足 new_stock_filter_days 天）
        is_delisted:  ndarray (N, T) bool，退市掩码（当前日期 >= 退市日期）
        can_buy:      ndarray (N, T) bool，可买入掩码
        can_sell:     ndarray (N, T) bool，可卖出掩码
    """

    def __init__(
        self,
        symbols: np.ndarray,
        dates: np.ndarray,
        is_suspended: np.ndarray,
        upper_limit: np.ndarray,
        lower_limit: np.ndarray,
        raw_open: np.ndarray,
        is_st: Optional[np.ndarray] = None,
        listed_dates: Optional[pd.Series] = None,
        delisted_dates: Optional[pd.Series] = None,
        new_stock_filter_days: int = 365,
        suspended_value_mode: str = "freeze",
    ):
        """
        Args:
            symbols: 股票代码数组 (N,)
            dates: 日期数组 (T,)
            is_suspended: 停牌掩码 (N, T) bool
            upper_limit: 涨停价矩阵 (N, T) float，0 表示无限制
            lower_limit: 跌停价矩阵 (N, T) float，0 表示无限制
            raw_open: 不复权开盘价矩阵 (N, T) float
            is_st: ST 标记矩阵 (N, T) bool，保留供未来使用
            listed_dates: 各股票上市日期 Series，index=symbol（用于新股过滤）
            delisted_dates: 各股票退市日期 Series，index=symbol（用于退市判断）
            new_stock_filter_days: 新股过滤天数（日历天），0 表示不过滤
            suspended_value_mode: 'freeze'（停牌日收益=0）或 'zero'（首日停牌=-1）
        """
        self.symbols = np.asarray(symbols)
        self.dates = np.asarray(dates)
        self.suspended_value_mode = suspended_value_mode

        N, T = len(self.symbols), len(self.dates)

        self.is_suspended = np.asarray(is_suspended, dtype=bool)

        # 涨停判断：不复权开盘价 ≈ 涨停价，且涨停价有效（>0）
        has_upper = upper_limit > 0
        self.is_limit_up = (
            np.isclose(raw_open, upper_limit, atol=0.002, rtol=0) & has_upper
        )

        # 跌停判断：不复权开盘价 ≈ 跌停价，且跌停价有效（>0）
        has_lower = lower_limit > 0
        self.is_limit_down = (
            np.isclose(raw_open, lower_limit, atol=0.002, rtol=0) & has_lower
        )

        # 新股过滤掩码
        if listed_dates is not None and new_stock_filter_days > 0:
            self.is_new_stock = self._compute_new_stock_mask(
                listed_dates, new_stock_filter_days
            )
        else:
            self.is_new_stock = np.zeros((N, T), dtype=bool)

        # 退市掩码
        if delisted_dates is not None:
            self.is_delisted = self._compute_delisted_mask(delisted_dates)
        else:
            self.is_delisted = np.zeros((N, T), dtype=bool)

        # 综合约束
        is_tradable = ~self.is_suspended & ~self.is_new_stock
        self.can_buy = is_tradable & ~self.is_limit_up
        self.can_sell = is_tradable & ~self.is_limit_down

    def _compute_new_stock_mask(
        self, listed_dates: pd.Series, filter_days: int
    ) -> np.ndarray:
        """计算新股过滤掩码（上市后 filter_days 日历天内为新股）"""
        dates_ts = pd.DatetimeIndex(self.dates)
        mask = np.zeros((len(self.symbols), len(self.dates)), dtype=bool)
        sym_to_idx = {sym: i for i, sym in enumerate(self.symbols.tolist())}

        for symbol, listed_date in listed_dates.items():
            if pd.isna(listed_date) or symbol not in sym_to_idx:
                continue
            cutoff = pd.Timestamp(listed_date) + pd.Timedelta(days=filter_days)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
            i = sym_to_idx[symbol]
            mask[i, :] = dates_ts < cutoff

        return mask

    def _compute_delisted_mask(self, delisted_dates: pd.Series) -> np.ndarray:
        """计算退市掩码：当前日期 >= 退市日期的位置为 True"""
        dates_ts = pd.DatetimeIndex(self.dates)
        mask = np.zeros((len(self.symbols), len(self.dates)), dtype=bool)
        sym_to_idx = {sym: i for i, sym in enumerate(self.symbols.tolist())}

        for symbol, delist_date in delisted_dates.items():
            if pd.isna(delist_date) or symbol not in sym_to_idx:
                continue
            delist_ts = pd.Timestamp(delist_date)
            if delist_ts.tzinfo is not None:
                delist_ts = delist_ts.tz_localize(None)
            i = sym_to_idx[symbol]
            mask[i, :] = dates_ts >= delist_ts

        return mask

    def adjust_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        统一调整收益率矩阵，优先级从高到低：退市 > 停牌 > 不调整

        优先级 1 (最高): 退市 → return = -1.0 (-100%)
            股票已摘牌，持仓归零，不管是否停牌
        优先级 2: 停牌（未退市）→ return = 0.0 (freeze)
            股票还在，只是暂停交易，价值冻结
        其他: 不调整，使用原始收益率

        注意：NaN 兜底（有持仓但 return 仍为 NaN → -100%）不在此方法中处理，
        因为需要 position 信息，由 evaluator._calculate_portfolio_returns 负责。

        Args:
            returns: 原始日收益率矩阵 (N, T)

        Returns:
            调整后的收益率矩阵 (N, T)
        """
        adjusted = returns.copy()

        # Priority 2: 停牌 → freeze (先写，后面退市会覆盖)
        susp = self.is_suspended
        if self.suspended_value_mode == "freeze":
            adjusted[susp] = 0.0
        elif self.suspended_value_mode == "zero":
            prev_susp = np.zeros_like(susp)
            prev_susp[:, 1:] = susp[:, :-1]
            is_first = susp & ~prev_susp
            adjusted[susp & ~is_first] = 0.0
            adjusted[is_first] = -1.0

        # Priority 1: 退市 → -100% (覆盖停牌的 freeze)
        if self.is_delisted.any():
            adjusted[self.is_delisted] = -1.0

        return adjusted

    def get_suspended_return_adjustment(self, returns: np.ndarray) -> np.ndarray:
        """向后兼容，调用 adjust_returns"""
        return self.adjust_returns(returns)

    def align_to(self, symbols: np.ndarray, dates: np.ndarray) -> "TradeContext":
        """
        对齐到指定 symbols 和 dates 的子集，返回新的 TradeContext

        Args:
            symbols: 目标股票代码数组
            dates: 目标日期数组

        Returns:
            对齐后的 TradeContext（属性引用新矩阵）
        """
        sym_idx = np.searchsorted(self.symbols, symbols)
        date_idx = np.searchsorted(self.dates, dates)

        def _slice(arr):
            return arr[sym_idx, :][:, date_idx]

        ctx = object.__new__(TradeContext)
        ctx.symbols = np.asarray(symbols)
        ctx.dates = np.asarray(dates)
        ctx.suspended_value_mode = self.suspended_value_mode
        ctx.is_suspended = _slice(self.is_suspended)
        ctx.is_limit_up = _slice(self.is_limit_up)
        ctx.is_limit_down = _slice(self.is_limit_down)
        ctx.is_new_stock = _slice(self.is_new_stock)
        ctx.is_delisted = _slice(self.is_delisted)

        is_tradable = ~ctx.is_suspended & ~ctx.is_new_stock
        ctx.can_buy = is_tradable & ~ctx.is_limit_up
        ctx.can_sell = is_tradable & ~ctx.is_limit_down

        return ctx
