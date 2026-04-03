"""
主评估器模块

整合所有评估指标的主评估器类，提供一站式因子评估功能。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.factor_data import FactorData
from core.position_data import PositionData
from core.ohlcv_data import OHLCVData

from .metrics import (
    # 收益类
    ReturnsMetricsCalculator,
    calculate_portfolio_returns,
    # 风险类
    RiskMetricsCalculator,
    # 风险调整类
    RiskAdjustedMetricsCalculator,
    # 换手率类
    TurnoverMetricsCalculator,
    # IC类
    ICMetricsCalculator,
    calculate_forward_returns,
)

from .visualization import (
    plot_cumulative_ic,
    plot_ic_distribution,
    plot_cumulative_returns,
    plot_drawdown,
    plot_returns_and_drawdown,
    plot_factor_distribution,
    plot_position_returns_distribution,
    plot_rolling_ic,
    create_evaluation_report_figure,
)


class FactorEvaluator:
    """
    因子评估器

    整合所有评估指标的一站式因子评估类。
    支持ETF（CSV数据/收盘价执行）和A股（DuckDB/开盘价执行）两种模式。

    Attributes:
        factor_data: 因子数据
        ohlcv_data: OHLCV数据
        position_data: 仓位数据（可选）
        forward_period: 前瞻期数（用于IC计算）
        periods_per_year: 每年的周期数
        execution_price: 执行价格模式，'close' 或 'open'
        trade_context: A股交易上下文（停牌/涨跌停约束），可选
    """

    def __init__(
        self,
        factor_data: FactorData,
        ohlcv_data: OHLCVData,
        position_data: Optional[PositionData] = None,
        forward_period: int = 1,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
        commission_rate: float = 0.0002,
        slippage_rate: float = 0.0,
        delay: int = 1,
        rebalance_freq: int = 1,
        # A股扩展参数
        execution_price: str = 'close',
        trade_context=None,
        buy_commission_rate: Optional[float] = None,
        sell_commission_rate: Optional[float] = None,
        stamp_tax_rate: float = 0.0,
        benchmark_returns: Optional[Dict[str, pd.Series]] = None,
        hold_mode: str = 'buyhold',
        blocked_buy_mode: str = 'wait_rebalance',
        blocked_sell_mode: str = 'asap',
    ):
        """
        初始化因子评估器

        Args:
            factor_data: 因子数据容器
            ohlcv_data: OHLCV数据容器
            position_data: 仓位数据容器（可选）
            forward_period: 前瞻期数（用于IC计算）
            periods_per_year: 每年的周期数
            risk_free_rate: 无风险利率
            commission_rate: 通用手续费率（买卖双边，若设置了 buy/sell_commission_rate 则被覆盖）
            slippage_rate: 滑点率
            delay: 调仓延迟（T日信号在 T+delay 日执行）
            rebalance_freq: 调仓频率（每N根K线调仓一次）
            execution_price: 执行价格 'close'（默认）或 'open'（A股T+1开盘）
            trade_context: TradeContext 对象，提供A股停牌/涨跌停约束（可选）
            buy_commission_rate: 买入佣金率（None=使用 commission_rate）
            sell_commission_rate: 卖出佣金率（None=使用 commission_rate）
            stamp_tax_rate: 印花税率（A股卖出时收取，默认0）
            benchmark_returns: 基准收益率字典，key为基准名（如'csi300'），value为日收益率Series
            hold_mode: 持仓模式
                - 'rebalance'（默认）: 调仓间隔内保持恒定权重，等价于每日再平衡到初始权重
                - 'buyhold': 调仓间隔内权重随价格漂移，模拟真实买入持有
            blocked_buy_mode: 买入受阻策略 'wait_rebalance'（默认）或 'asap'
            blocked_sell_mode: 卖出受阻策略 'wait_rebalance' 或 'asap'（默认）
        """
        self.factor_data = factor_data
        self.ohlcv_data = ohlcv_data
        self.position_data = position_data
        self.forward_period = forward_period
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.delay = delay
        self.rebalance_freq = rebalance_freq
        self.execution_price = execution_price
        self.trade_context = trade_context
        self.buy_commission_rate = buy_commission_rate if buy_commission_rate is not None else commission_rate
        self.sell_commission_rate = sell_commission_rate if sell_commission_rate is not None else commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        if hold_mode not in ('rebalance', 'buyhold'):
            raise ValueError(f"hold_mode must be 'rebalance' or 'buyhold', got '{hold_mode}'")
        self.hold_mode = hold_mode
        self.blocked_buy_mode = blocked_buy_mode
        self.blocked_sell_mode = blocked_sell_mode
        self.benchmark_returns = benchmark_returns or {}

        # 对齐数据（numpy searchsorted）
        self._align_data()

        # 计算前瞻收益（IC用）- 全 numpy
        self.forward_returns_arr = self._compute_forward_returns_numpy()

        # 计算组合收益（如果有仓位数据）
        self.portfolio_returns = None
        self.net_portfolio_returns = None
        if self.position_data is not None:
            self._calculate_portfolio_returns()

    def _align_data(self):
        """对齐所有数据的日期和标的，存储为 ndarray（np.intersect1d + dict 索引）"""
        factor_dates = self.factor_data.dates      # (T1,) datetime64
        ohlcv_dates = self.ohlcv_data.dates        # (T2,) datetime64
        factor_symbols = self.factor_data.symbols  # (N1,)
        ohlcv_symbols = self.ohlcv_data.symbols    # (N2,)

        # 共同日期/标的（sorted）
        common_dates = np.intersect1d(factor_dates, ohlcv_dates)
        common_symbols = np.intersect1d(factor_symbols, ohlcv_symbols)

        self.common_symbols = common_symbols
        self.common_dates = common_dates

        # 建立 symbol/date → 行列索引 的映射（dict 查找，O(1)）
        fac_sym_map = {s: i for i, s in enumerate(factor_symbols)}
        ohlcv_sym_map = {s: i for i, s in enumerate(ohlcv_symbols)}
        fac_date_map = {d: i for i, d in enumerate(factor_dates)}
        ohlcv_date_map = {d: i for i, d in enumerate(ohlcv_dates)}

        fac_sym_idx = np.array([fac_sym_map[s] for s in common_symbols])
        ohlcv_sym_idx = np.array([ohlcv_sym_map[s] for s in common_symbols])
        fac_date_idx = np.array([fac_date_map[d] for d in common_dates])
        ohlcv_date_idx = np.array([ohlcv_date_map[d] for d in common_dates])

        # 因子值矩阵对齐
        self.aligned_factor_values = self.factor_data.values[np.ix_(fac_sym_idx, fac_date_idx)]

        # 收盘价对齐
        self.aligned_close = self.ohlcv_data.close[np.ix_(ohlcv_sym_idx, ohlcv_date_idx)]

        # 开盘价对齐（A股模式）
        if self.execution_price == 'open':
            self.aligned_open = self.ohlcv_data.open[np.ix_(ohlcv_sym_idx, ohlcv_date_idx)]
        else:
            self.aligned_open = None

        # 日收益率对齐
        if self.execution_price == 'open':
            full_returns = self.ohlcv_data.get_open_returns(periods=1)  # (N2, T2)
        else:
            full_returns = self.ohlcv_data.get_returns()                # (N2, T2)
        self.aligned_returns = full_returns[np.ix_(ohlcv_sym_idx, ohlcv_date_idx)]

        # 对齐 TradeContext
        if self.trade_context is not None:
            self.aligned_trade_context = self.trade_context.align_to(
                common_symbols, common_dates
            )
        else:
            self.aligned_trade_context = None

        # ── 逐日 OHLCV 有效性检查：OHLCV 为 NaN 的位置（退市/缺失），因子值也置 NaN ──
        ohlcv_nan_mask = np.isnan(self.aligned_close)  # (N, T)
        n_masked_cells = int(np.sum(ohlcv_nan_mask & ~np.isnan(self.aligned_factor_values)))
        if n_masked_cells > 0:
            self.aligned_factor_values[ohlcv_nan_mask] = np.nan

        # 仓位数据对齐
        if self.position_data is not None:
            pos_sym_map = {s: i for i, s in enumerate(self.position_data.symbols)}
            pos_date_map = {d: i for i, d in enumerate(self.position_data.dates)}

            N, T = len(common_symbols), len(common_dates)
            aligned_pos = np.zeros((N, T), dtype=np.float64)

            # 仅对 position_data 中存在的 symbol/date 做填充
            sym_in_pos = np.array([s in pos_sym_map for s in common_symbols])
            date_in_pos = np.array([d in pos_date_map for d in common_dates])

            if sym_in_pos.any() and date_in_pos.any():
                pos_sym_idx = np.array([pos_sym_map[s] for s in common_symbols[sym_in_pos]])
                pos_date_idx = np.array([pos_date_map[d] for d in common_dates[date_in_pos]])
                subset = self.position_data.weights[np.ix_(pos_sym_idx, pos_date_idx)]
                aligned_pos[np.ix_(np.where(sym_in_pos)[0], np.where(date_in_pos)[0])] = subset

            self.aligned_position = aligned_pos
        else:
            self.aligned_position = None

    def _compute_forward_returns_numpy(self) -> np.ndarray:
        """计算前瞻收益矩阵 (N, T)，纯 numpy 实现"""
        p = self.forward_period
        N, T = self.aligned_close.shape
        result = np.full((N, T), np.nan, dtype=np.float64)

        if self.execution_price == 'open' and self.aligned_open is not None:
            # T+1日开盘买入，T+1+p日开盘卖出：result[t] = open[t+1+p] / open[t+1] - 1
            if T >= 2 + p:
                result[:, :T - 1 - p] = (
                    self.aligned_open[:, 1 + p:] / self.aligned_open[:, 1:T - p] - 1
                )
        else:
            # 收盘价：result[t] = close[t+p] / close[t] - 1
            if T > p:
                result[:, :T - p] = (
                    self.aligned_close[:, p:] / self.aligned_close[:, :T - p] - 1
                )

        return result

    def _apply_rebalance_freq(self, position: np.ndarray) -> np.ndarray:
        """
        应用调仓频率，将每日目标仓位转换为按频率调仓的实际仓位

        使用 boolean mask + np.searchsorted，无 Python 循环。

        Args:
            position: 每日目标仓位 ndarray (N, T)

        Returns:
            np.ndarray: 按调仓频率调整后的仓位 (N, T)
        """
        if self.rebalance_freq <= 1:
            return position

        T = position.shape[1]

        # 生成调仓日掩码：索引 0, freq, 2*freq, ...
        rebalance_mask = np.zeros(T, dtype=bool)
        rebalance_mask[::self.rebalance_freq] = True

        # 每个调仓日的实际列索引
        rebalance_day_indices = np.where(rebalance_mask)[0]  # (K,)

        # 对每个时刻 t，找到 t 对应的最近调仓日索引（用 searchsorted）
        t_to_rebalance = np.searchsorted(rebalance_day_indices, np.arange(T), side='right') - 1
        t_to_rebalance = np.clip(t_to_rebalance, 0, len(rebalance_day_indices) - 1)
        source_indices = rebalance_day_indices[t_to_rebalance]  # (T,)

        # 向量化索引：每列 t 取自 source_indices[t] 列
        return position[:, source_indices]

    def _apply_buyhold_drift(self, position: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        将恒定权重仓位转换为买入持有（漂移权重）。

        调仓日保持原始权重不变，调仓间隔内权重随各股票日收益率累积漂移，
        使组合收益等价于"调仓日等权买入、期间不做再平衡"。

        数学原理：
            调仓日 t0 权重 w0[i]
            第 t 天权重 w[i,t] = w0[i] * cum_ret[i,t0→t-1] / Σ_j(w0[j] * cum_ret[j,t0→t-1])
            其中 cum_ret = Π(1 + r[s])，s 从 t0 到 t-1

        Args:
            position: 恒定权重仓位 (N, T)，调仓间隔内同一列向量
            returns: 日收益率矩阵 (N, T)，returns[i,t] = close[t]/close[t-1] - 1

        Returns:
            np.ndarray: 漂移权重仓位 (N, T)
        """
        if self.rebalance_freq <= 1:
            return position

        N, T = position.shape
        result = position.copy()

        rebalance_indices = list(range(0, T, self.rebalance_freq))

        for k, t0 in enumerate(rebalance_indices):
            t_end = rebalance_indices[k + 1] if k + 1 < len(rebalance_indices) else T
            span = t_end - t0 - 1  # number of days to drift
            if span <= 0:
                continue

            w0 = position[:, t0]  # (N,) initial weights on rebalance day

            # returns[:, t0:t_end-1] covers days t0, t0+1, ..., t_end-2
            # cum[k] = Π(1+r[s], s=t0..t0+k) → weight for day t0+k+1
            period_ret = returns[:, t0:t_end - 1].copy()  # (N, span)
            period_ret = np.where(np.isnan(period_ret), 0.0, period_ret)
            cum = np.cumprod(1.0 + period_ret, axis=1)    # (N, span)

            values = w0[:, None] * cum                     # (N, span)
            col_sums = np.nansum(values, axis=0, keepdims=True)
            col_sums = np.where(col_sums < 1e-10, 1.0, col_sums)
            result[:, t0 + 1:t_end] = values / col_sums

        return result

    def _calculate_portfolio_returns(self):
        """计算组合收益率序列，纯 numpy 矩阵运算"""
        # ── 1. 应用调仓频率 + 交易约束（A股模式）──────────────────────────
        if self.aligned_trade_context is not None:
            # A股模式：PositionAdjuster 仍接收 DataFrame，包装后调用
            from mappers.position_adjuster import PositionAdjuster
            pos_df = pd.DataFrame(
                self.aligned_position,
                index=self.common_symbols,
                columns=pd.DatetimeIndex(self.common_dates),
            )
            adjuster = PositionAdjuster(
                trade_context=self.aligned_trade_context,
                delay=self.delay,
                blocked_buy_mode=self.blocked_buy_mode,
                blocked_sell_mode=self.blocked_sell_mode,
            )
            rebalanced_df = adjuster.adjust(pos_df, rebalance_freq=self.rebalance_freq)
            rebalanced_position = rebalanced_df.values
        elif self.rebalance_freq > 1:
            rebalanced_position = self._apply_rebalance_freq(self.aligned_position)
        else:
            rebalanced_position = self.aligned_position

        # ── 1.5 买入持有漂移：将恒定权重转为按日漂移权重 ──────────────────
        if self.hold_mode == 'buyhold' and self.rebalance_freq > 1:
            rebalanced_position = self._apply_buyhold_drift(
                rebalanced_position, self.aligned_returns
            )

        # ── 2. 收益率调整（优先级：退市 > 停牌 > NaN兜底）──────────────────
        aligned_returns = self.aligned_returns.copy()
        if self.aligned_trade_context is not None:
            # adjust_returns 统一处理退市(-100%)和停牌(freeze=0)
            aligned_returns = self.aligned_trade_context.adjust_returns(
                aligned_returns
            )

        # ── 2.5 退市清仓：退市日之后立即清零仓位 ──────────────���───────────
        # trade_context 将退市股每日收益设为 -100%，但仓位权重在调仓间隔内不变。
        # 退市当日保留仓位（承受 -100% 损失），次日起清零（股票已不存在）。
        if self.aligned_trade_context is not None:
            is_delist_return = (aligned_returns == -1.0)
            ever_delisted = np.cumsum(is_delist_return, axis=1) > 0
            post_delist = np.zeros_like(ever_delisted, dtype=bool)
            post_delist[:, 1:] = ever_delisted[:, :-1]
            rebalanced_position[post_delist] = 0.0

        # ── 3. 计算日组合收益（T日仓位在 T+delay 日产生收益）───────────────
        N, T = rebalanced_position.shape
        if self.delay > 0:
            position_shifted = np.empty_like(rebalanced_position)
            position_shifted[:, :self.delay] = np.nan
            position_shifted[:, self.delay:] = rebalanced_position[:, :-self.delay]
        else:
            position_shifted = rebalanced_position

        # ── NaN 兜底：有仓位但收益率仍为 NaN → 视为持平 (0%) ──
        # trade_context.adjust_returns 已处理退市(-100%)和停牌(0%)，
        # 残留 NaN 来自数据缺口或边界效应（如 open 执行模式最后一天），不应视为全额亏损
        has_position = (position_shifted > 0) & ~np.isnan(position_shifted)
        has_no_return = np.isnan(aligned_returns)
        nan_fallback_mask = has_position & has_no_return
        n_nan_fallback = int(nan_fallback_mask.sum())
        if n_nan_fallback > 0:
            aligned_returns[nan_fallback_mask] = 0.0

        # 按列求加权收益（nansum → 跳过 NaN 权重，与 pandas skipna=True 一致）
        daily_returns_arr = np.nansum(position_shifted * aligned_returns, axis=0)  # (T,)

        # 排除无效列：首 delay 列（仓位全 NaN）+ 收益率全 NaN 的边界日
        # （open 执行模式下，最后一天 open return 需要 T+1 日数据，不存在故全 NaN）
        valid_mask = ~np.all(np.isnan(position_shifted), axis=0)  # (T,)
        all_returns_nan = np.all(np.isnan(self.aligned_returns), axis=0)
        valid_mask = valid_mask & ~all_returns_nan
        daily_returns_arr = daily_returns_arr[valid_mask]
        valid_dates = self.common_dates[valid_mask]

        self.portfolio_returns = pd.Series(
            daily_returns_arr, index=pd.DatetimeIndex(valid_dates)
        )

        # ── 4. 计算交易成本（买卖不对称）───────────────────────────────────
        weight_change = np.diff(rebalanced_position, axis=1)          # (N, T-1)
        buy_turnover = np.sum(np.maximum(weight_change, 0.0), axis=0)  # (T-1,)
        sell_turnover = np.sum(np.maximum(-weight_change, 0.0), axis=0)

        buy_cost = buy_turnover * (self.buy_commission_rate + self.slippage_rate)
        sell_cost = sell_turnover * (
            self.sell_commission_rate + self.stamp_tax_rate + self.slippage_rate
        )
        total_cost_arr = buy_cost + sell_cost  # (T-1,)

        cost_series = pd.Series(
            total_cost_arr,
            index=pd.DatetimeIndex(self.common_dates[1:]),
        ).reindex(pd.DatetimeIndex(valid_dates)).fillna(0.0)

        # ── 5. 净收益 ────────────────────────────────────────────────────────
        self.net_portfolio_returns = self.portfolio_returns - cost_series

    def _make_factor_df(self) -> pd.DataFrame:
        """将 aligned_factor_values 包装为 DataFrame（供 metrics/visualization 使用）"""
        return pd.DataFrame(
            self.aligned_factor_values,
            index=self.common_symbols,
            columns=pd.DatetimeIndex(self.common_dates),
        )

    def _make_fwd_df(self) -> pd.DataFrame:
        """将 forward_returns_arr 包装为 DataFrame（供 ICMetricsCalculator 使用）"""
        return pd.DataFrame(
            self.forward_returns_arr,
            index=self.common_symbols,
            columns=pd.DatetimeIndex(self.common_dates),
        )

    def _make_position_df(self) -> pd.DataFrame:
        """将 aligned_position 包装为 DataFrame（供 TurnoverMetricsCalculator 使用）"""
        return pd.DataFrame(
            self.aligned_position,
            index=self.common_symbols,
            columns=pd.DatetimeIndex(self.common_dates),
        )

    def calculate_ic_metrics(self) -> Dict[str, float]:
        """
        计算IC类指标

        Returns:
            dict: IC类指标字典
        """
        calculator = ICMetricsCalculator(
            self._make_factor_df(),
            self._make_fwd_df(),
            self.periods_per_year,
        )
        return calculator.get_all_metrics()

    def calculate_returns_metrics(self, net: bool = True) -> Dict[str, float]:
        """
        计算收益类指标

        Args:
            net: 是否使用扣除成本后的收益

        Returns:
            dict: 收益类指标字典
        """
        returns = self.net_portfolio_returns if net else self.portfolio_returns
        if returns is None:
            return {'error': 'No position data available'}

        calculator = ReturnsMetricsCalculator(returns, self.periods_per_year)
        return calculator.get_all_metrics()

    def calculate_risk_metrics(self, net: bool = True) -> Dict[str, float]:
        """
        计算风险类指标

        Args:
            net: 是否使用扣除成本后的收益

        Returns:
            dict: 风险类指标字典
        """
        returns = self.net_portfolio_returns if net else self.portfolio_returns
        if returns is None:
            return {'error': 'No position data available'}

        calculator = RiskMetricsCalculator(returns, self.periods_per_year)
        return calculator.get_all_metrics()

    def calculate_risk_adjusted_metrics(self, net: bool = True, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算风险调整收益指标

        Args:
            net: 是否使用扣除成本后的收益
            benchmark_returns: 基准收益率Series（可选，用于计算IR/Treynor）

        Returns:
            dict: 风险调整收益指标字典
        """
        returns = self.net_portfolio_returns if net else self.portfolio_returns
        if returns is None:
            return {'error': 'No position data available'}

        calculator = RiskAdjustedMetricsCalculator(
            returns,
            self.periods_per_year,
            self.risk_free_rate,
            benchmark_returns=benchmark_returns,
        )
        return calculator.get_all_metrics()

    def calculate_turnover_metrics(self) -> Dict[str, float]:
        """
        计算换手率指标

        Returns:
            dict: 换手率指标字典
        """
        if self.aligned_position is None:
            return {'error': 'No position data available'}

        calculator = TurnoverMetricsCalculator(
            self._make_position_df(),
            self.periods_per_year
        )
        return calculator.get_all_metrics()

    def calculate_decile_returns(self, n_groups: int = 10) -> Dict[int, pd.Series]:
        """
        计算十分位（或任意 n 分位）日频等权组合收益

        按因子值将股票等分为 n_groups 组，每组计算等权日收益序列。
        遵循与主策略相同的 rebalance_freq 和 delay。
        不计交易成本（用于因子区分度评估）。

        Args:
            n_groups: 分组数，默认 10

        Returns:
            dict: key=分组编号(0=因子值最低, n_groups-1=最高),
                  value=pd.Series(日收益率, index=DatetimeIndex)
        """
        N, T = self.aligned_factor_values.shape

        # ── 1. 确定调仓日 ─────────────────────────────────────────────
        rb_freq = max(1, self.rebalance_freq)
        rebalance_day_indices = np.arange(0, T, rb_freq)
        n_rb = len(rebalance_day_indices)

        # ── 2. 在每个调仓日分组 ───────────────────────────────────────
        # decile_at_rb: (N, n_rb)，值 0..n_groups-1 或 NaN
        decile_at_rb = np.full((N, n_rb), np.nan, dtype=np.float64)
        # Loop: ~n_rb 次 (典型 400-800)，每次 argsort ~N(5000) 元素
        for i, rb_idx in enumerate(rebalance_day_indices):
            factor_col = self.aligned_factor_values[:, rb_idx]
            valid_mask = ~np.isnan(factor_col)
            n_valid = int(valid_mask.sum())
            if n_valid < n_groups:
                continue
            # argsort 等分：按因子值排名，均匀分配到 n_groups 组
            valid_values = factor_col[valid_mask]
            ranks = np.argsort(np.argsort(valid_values))  # 0-based rank
            deciles = np.minimum(ranks * n_groups // n_valid, n_groups - 1)
            decile_at_rb[valid_mask, i] = deciles

        # ── 3. 将调仓日分组传播到所有日期 (searchsorted) ──────────────
        t_indices = np.arange(T)
        rb_pos = np.searchsorted(rebalance_day_indices, t_indices, side='right') - 1
        rb_pos = np.clip(rb_pos, 0, n_rb - 1)
        decile_assignment = decile_at_rb[:, rb_pos]  # (N, T)

        # ── 4. 应用 delay ────────────────────────────────────────────
        if self.delay > 0:
            shifted = np.full_like(decile_assignment, np.nan)
            shifted[:, self.delay:] = decile_assignment[:, :-self.delay]
            decile_assignment = shifted

        # ── 5. 收益率调整（与主策略一致）──────────────────────────────
        adj_returns = self.aligned_returns.copy()
        if self.aligned_trade_context is not None:
            adj_returns = self.aligned_trade_context.adjust_returns(adj_returns)
            # 退市后收益设为 NaN（不参与分组收益计算）
            is_delist = (adj_returns == -1.0)
            ever_delisted = np.cumsum(is_delist, axis=1) > 0
            post_delist = np.zeros_like(ever_delisted, dtype=bool)
            post_delist[:, 1:] = ever_delisted[:, :-1]
            adj_returns[post_delist] = np.nan

        # ── 6. 计算每组日收益 ────────────────────────────────────────
        # valid_mask: 与主策略保持一致的有效日期
        all_returns_nan = np.all(np.isnan(self.aligned_returns), axis=0)
        has_assignment = np.any(~np.isnan(decile_assignment), axis=0)
        valid_mask = has_assignment & ~all_returns_nan
        valid_dates = self.common_dates[valid_mask]

        decile_results = {}
        # Loop: n_groups 次 (10)
        import warnings
        for d in range(n_groups):
            in_decile = (decile_assignment == d)  # (N, T) bool
            masked = np.where(in_decile, adj_returns, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                daily_ret = np.nanmean(masked, axis=0)  # (T,)
            daily_ret = daily_ret[valid_mask]
            decile_results[d] = pd.Series(
                daily_ret, index=pd.DatetimeIndex(valid_dates)
            )

        return decile_results

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        运行完整评估

        Returns:
            dict: 所有评估指标字典
        """
        results = {
            'factor_name': self.factor_data.name,
            'factor_type': self.factor_data.factor_type,
            'factor_params': self.factor_data.params,
            'forward_period': self.forward_period,
        }

        # IC类指标
        results['ic_metrics'] = self.calculate_ic_metrics()

        # 组合指标（如果有仓位数据）
        if self.portfolio_returns is not None:
            results['returns_metrics'] = self.calculate_returns_metrics()
            results['risk_metrics'] = self.calculate_risk_metrics()
            results['risk_adjusted_metrics'] = self.calculate_risk_adjusted_metrics()
            results['turnover_metrics'] = self.calculate_turnover_metrics()
            # 日频净收益曲线（pd.Series, index=DatetimeIndex）
            results['daily_returns'] = self.net_portfolio_returns

        # 十分位日频收益曲线
        results['decile_daily_returns'] = self.calculate_decile_returns()

        # 计算各基准的超额收益指标
        if self.benchmark_returns and self.portfolio_returns is not None:
            benchmark_metrics = {}
            for bm_name, bm_ret in self.benchmark_returns.items():
                bm_adj = self.calculate_risk_adjusted_metrics(benchmark_returns=bm_ret)
                benchmark_metrics[bm_name] = {
                    'information_ratio': bm_adj.get('information_ratio', float('nan')),
                    'treynor_ratio': bm_adj.get('treynor_ratio', float('nan')),
                }
                # 计算超额年化收益
                net_ret = self.net_portfolio_returns
                if net_ret is not None:
                    common_idx = net_ret.index.intersection(bm_ret.index)
                    if len(common_idx) > 0:
                        excess = net_ret.loc[common_idx] - bm_ret.loc[common_idx]
                        ann_excess = float(excess.mean() * self.periods_per_year)
                        benchmark_metrics[bm_name]['excess_annual_return'] = ann_excess
            results['benchmark_metrics'] = benchmark_metrics

        return results

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成文本报告

        Args:
            output_path: 报告保存路径

        Returns:
            str: 报告文本
        """
        results = self.run_full_evaluation()

        lines = []
        lines.append("=" * 60)
        lines.append(f"因子评估报告: {results['factor_name']}")
        lines.append("=" * 60)
        lines.append(f"\n因子参数: {results['factor_params']}")
        lines.append(f"前瞻期数: {results['forward_period']}")

        # IC指标
        if 'ic_metrics' in results:
            lines.append("\n" + "-" * 40)
            lines.append("IC类指标")
            lines.append("-" * 40)
            ic = results['ic_metrics']
            lines.append(f"IC Mean:          {ic.get('ic_mean', np.nan):.4f}")
            lines.append(f"IC Std:           {ic.get('ic_std', np.nan):.4f}")
            lines.append(f"IC IR:            {ic.get('ic_ir', np.nan):.4f}")
            lines.append(f"IC Positive Ratio:{ic.get('ic_positive_ratio', np.nan):.2%}")
            lines.append(f"Rank IC Mean:     {ic.get('rank_ic_mean', np.nan):.4f}")
            lines.append(f"Rank IC IR:       {ic.get('rank_ic_ir', np.nan):.4f}")

        # 收益指标
        if 'returns_metrics' in results:
            lines.append("\n" + "-" * 40)
            lines.append("收益类指标")
            lines.append("-" * 40)
            ret = results['returns_metrics']
            lines.append(f"总收益:           {ret.get('total_return', np.nan):.2%}")
            lines.append(f"年化收益:         {ret.get('annualized_return', np.nan):.2%}")

        # 风险指标
        if 'risk_metrics' in results:
            lines.append("\n" + "-" * 40)
            lines.append("风险类指标")
            lines.append("-" * 40)
            risk = results['risk_metrics']
            lines.append(f"最大回撤:         {risk.get('max_drawdown', np.nan):.2%}")
            lines.append(f"年化波动率:       {risk.get('annualized_volatility', np.nan):.2%}")

        # 风险调整指标
        if 'risk_adjusted_metrics' in results:
            lines.append("\n" + "-" * 40)
            lines.append("风险调整指标")
            lines.append("-" * 40)
            adj = results['risk_adjusted_metrics']
            lines.append(f"夏普比率:         {adj.get('sharpe_ratio', np.nan):.4f}")
            lines.append(f"卡玛比率:         {adj.get('calmar_ratio', np.nan):.4f}")
            lines.append(f"索提诺比率:       {adj.get('sortino_ratio', np.nan):.4f}")

        # 换手率指标
        if 'turnover_metrics' in results:
            lines.append("\n" + "-" * 40)
            lines.append("换手率指标")
            lines.append("-" * 40)
            to = results['turnover_metrics']
            lines.append(f"日均换手率:       {to.get('avg_daily_turnover', np.nan):.2%}")
            lines.append(f"年化换手率:       {to.get('annualized_turnover', np.nan):.2%}")

        lines.append("\n" + "=" * 60)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def plot(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        生成可视化图表

        Args:
            output_dir: 图表保存目录

        Returns:
            dict: 图表对象字典
        """
        figures = {}

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # 构建 IC 序列（使用包装好的 DataFrame）
        ic_calc = ICMetricsCalculator(
            self._make_factor_df(),
            self._make_fwd_df(),
            self.periods_per_year,
        )
        ic_series = ic_calc.rank_ic_series()

        factor_df = self._make_factor_df()

        # 1. 累积IC曲线
        save_path = str(output_path / 'cumulative_ic.png') if output_dir else None
        figures['cumulative_ic'] = plot_cumulative_ic(
            ic_series,
            title=f"累积IC曲线 - {self.factor_data.name}",
            save_path=save_path
        )

        # 2. IC分布
        save_path = str(output_path / 'ic_distribution.png') if output_dir else None
        figures['ic_distribution'] = plot_ic_distribution(
            ic_series,
            title=f"IC分布 - {self.factor_data.name}",
            save_path=save_path
        )

        # 3. 因子值分布
        save_path = str(output_path / 'factor_distribution.png') if output_dir else None
        figures['factor_distribution'] = plot_factor_distribution(
            factor_df,
            title=f"因子值分布 - {self.factor_data.name}",
            save_path=save_path
        )

        # 4. 组合相关图表（如果有仓位数据）
        if self.net_portfolio_returns is not None:
            # 收益回撤曲线
            save_path = str(output_path / 'returns_drawdown.png') if output_dir else None
            figures['returns_drawdown'] = plot_returns_and_drawdown(
                self.net_portfolio_returns,
                title=f"收益与回撤 - {self.factor_data.name}",
                save_path=save_path
            )

            # 收益分布
            save_path = str(output_path / 'returns_distribution.png') if output_dir else None
            figures['returns_distribution'] = plot_position_returns_distribution(
                self.net_portfolio_returns,
                title=f"收益分布 - {self.factor_data.name}",
                save_path=save_path
            )

            # 滚动IC
            save_path = str(output_path / 'rolling_ic.png') if output_dir else None
            figures['rolling_ic'] = plot_rolling_ic(
                ic_series,
                window=min(60, len(ic_series) // 4),
                title=f"滚动IC - {self.factor_data.name}",
                save_path=save_path
            )

            # 综合报告图
            save_path = str(output_path / 'evaluation_report.png') if output_dir else None
            figures['evaluation_report'] = create_evaluation_report_figure(
                ic_series,
                self.net_portfolio_returns,
                factor_df,
                title=f"因子评估报告 - {self.factor_data.name}",
                save_path=save_path
            )

        return figures

    def print_report(self):
        """打印评估报告到控制台"""
        report = self.generate_report()
        print(report)


class BatchFactorEvaluator:
    """
    批量因子评估器

    用于批量评估多个因子。
    """

    def __init__(
        self,
        ohlcv_data: OHLCVData,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ):
        """
        初始化批量因子评估器

        Args:
            ohlcv_data: OHLCV数据
            periods_per_year: 每年的周期数
            risk_free_rate: 无风险利率
        """
        self.ohlcv_data = ohlcv_data
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.results = []

    def evaluate(
        self,
        factor_data: FactorData,
        position_data: Optional[PositionData] = None,
        forward_period: int = 1,
    ) -> Dict[str, Any]:
        """
        评估单个因子

        Args:
            factor_data: 因子数据
            position_data: 仓位数据（可选）
            forward_period: 前瞻期数

        Returns:
            dict: 评估结果
        """
        evaluator = FactorEvaluator(
            factor_data=factor_data,
            ohlcv_data=self.ohlcv_data,
            position_data=position_data,
            forward_period=forward_period,
            periods_per_year=self.periods_per_year,
            risk_free_rate=self.risk_free_rate,
        )

        result = evaluator.run_full_evaluation()
        self.results.append(result)
        return result

    def evaluate_multiple(
        self,
        factor_data_list: List[FactorData],
        position_data_list: Optional[List[PositionData]] = None,
        forward_period: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        批量评估多个因子

        Args:
            factor_data_list: 因子数据列表
            position_data_list: 仓位数据列表（可选）
            forward_period: 前瞻期数

        Returns:
            list: 评估结果列表
        """
        if position_data_list is None:
            position_data_list = [None] * len(factor_data_list)

        results = []
        for factor_data, position_data in zip(factor_data_list, position_data_list):
            result = self.evaluate(factor_data, position_data, forward_period)
            results.append(result)

        return results

    def get_comparison_table(self) -> pd.DataFrame:
        """
        获取因子对比表

        Returns:
            DataFrame: 对比表
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            row = {
                'factor_name': result.get('factor_name', 'Unknown'),
            }

            # 提取关键指标
            if 'ic_metrics' in result:
                ic = result['ic_metrics']
                row['ic_mean'] = ic.get('ic_mean', np.nan)
                row['rank_ic_mean'] = ic.get('rank_ic_mean', np.nan)
                row['rank_ic_ir'] = ic.get('rank_ic_ir', np.nan)

            if 'risk_adjusted_metrics' in result:
                adj = result['risk_adjusted_metrics']
                row['sharpe_ratio'] = adj.get('sharpe_ratio', np.nan)
                row['calmar_ratio'] = adj.get('calmar_ratio', np.nan)

            if 'returns_metrics' in result:
                ret = result['returns_metrics']
                row['annualized_return'] = ret.get('annualized_return', np.nan)

            if 'risk_metrics' in result:
                risk = result['risk_metrics']
                row['max_drawdown'] = risk.get('max_drawdown', np.nan)

            rows.append(row)

        return pd.DataFrame(rows)

    def save_comparison_table(self, path: str):
        """
        保存对比表到CSV

        Args:
            path: 保存路径
        """
        table = self.get_comparison_table()
        table.to_csv(path, index=False, encoding='utf-8-sig')
