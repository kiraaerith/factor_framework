"""
仓位调整器模块

根据A股交易限制（停牌、涨跌停）调整目标仓位，使回测更接近真实场景。

调整规则：
  - 想买入但涨停/停牌：不买入，维持 0（或原有持仓）
  - 想卖出但跌停/停牌：不卖出，维持原有持仓
  - 调整后对剩余可执行权重重新归一化到总和为 1
"""

from typing import Optional
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trade_context import TradeContext


class PositionAdjuster:
    """
    A股仓位调整器

    在因子映射产生目标仓位后，根据下一个执行日的交易约束（停牌/涨跌停）
    逐步调整可执行持仓。实现为前向顺序循环，正确追踪实际持仓状态。

    使用示例::

        adjuster = PositionAdjuster(trade_context, delay=1)
        actual_position = adjuster.adjust(target_position, rebalance_freq=5)
    """

    def __init__(self, trade_context: TradeContext, delay: int = 1):
        """
        Args:
            trade_context: 交易上下文（包含 can_buy / can_sell 掩码）
            delay: 执行延迟（T日信号在 T+delay 日执行），默认 1
        """
        self.trade_context = trade_context
        self.delay = delay

    def adjust(
        self,
        target_position: pd.DataFrame,
        rebalance_freq: int = 1,
    ) -> pd.DataFrame:
        """
        根据交易约束调整目标仓位，返回实际可执行持仓矩阵

        算法：
          1. 确定调仓日（每 rebalance_freq 天一次）
          2. 在调仓日，查看 T+delay 日的 can_buy / can_sell 约束
          3. 调整当日目标仓位：
             - 想买入但 can_buy=False：维持持仓 0（不开仓）
             - 想卖出但 can_sell=False：维持原有持仓
          4. 对调整后的仓位重新归一化（正权重之和 = 1）
          5. 非调仓日保持上一个调仓日的实际持仓

        Args:
            target_position: 目标仓位矩阵 (N×T)，来自因子映射器
            rebalance_freq: 调仓频率（每 N 根 K 线调仓一次）

        Returns:
            实际持仓矩阵 (N×T)，已根据约束调整并归一化
        """
        # 兼容 numpy API：若 can_buy / can_sell 为 ndarray，包装为 DataFrame
        ctx = self.trade_context
        can_buy = ctx.can_buy
        can_sell = ctx.can_sell
        if isinstance(can_buy, np.ndarray):
            can_buy = pd.DataFrame(can_buy, index=ctx.symbols, columns=ctx.dates)
            can_sell = pd.DataFrame(can_sell, index=ctx.symbols, columns=ctx.dates)

        dates = target_position.columns
        n_periods = len(dates)
        dates_list = list(dates)

        result = target_position.copy() * 0.0
        current_held = pd.Series(0.0, index=target_position.index)

        # 确定调仓日标记
        rebalance_mask = [False] * n_periods
        for i in range(0, n_periods, rebalance_freq):
            rebalance_mask[i] = True

        for i, date in enumerate(dates_list):
            if rebalance_mask[i]:
                target = target_position[date].copy()
                exec_idx = i + self.delay

                if exec_idx < n_periods:
                    exec_date = dates_list[exec_idx]
                    adjusted = self._apply_constraints(
                        target, current_held, exec_date, can_buy, can_sell
                    )
                    current_held = adjusted
                else:
                    # 超出范围：直接采纳目标仓位（末端无法验证）
                    current_held = target

            result[date] = current_held

        return result

    def _apply_constraints(
        self,
        target: pd.Series,
        prev_held: pd.Series,
        exec_date,
        can_buy: pd.DataFrame,
        can_sell: pd.DataFrame,
    ) -> pd.Series:
        """
        对单个调仓日应用交易约束

        Args:
            target: 目标持仓权重
            prev_held: 前一期实际持仓权重
            exec_date: 执行日期
            can_buy: 可买入掩码 DataFrame (N×T)
            can_sell: 可卖出掩码 DataFrame (N×T)

        Returns:
            约束后的持仓权重（已归一化）
        """
        if exec_date not in can_buy.columns:
            return target

        can_buy_today = can_buy[exec_date].reindex(target.index, fill_value=True)
        can_sell_today = can_sell[exec_date].reindex(target.index, fill_value=True)

        # 判断交易意图
        buy_intent = target > prev_held + 1e-9      # 增仓/新买
        sell_intent = target < prev_held - 1e-9     # 减仓/清仓

        # 无法执行的操作：维持现有持仓
        cant_execute = (buy_intent & ~can_buy_today) | (sell_intent & ~can_sell_today)

        adjusted = target.copy()
        adjusted[cant_execute] = prev_held[cant_execute]

        # 重新归一化正权重（确保多头权重之和为 1）
        pos_mask = adjusted > 1e-9
        pos_sum = adjusted[pos_mask].sum()
        if pos_sum > 1e-9:
            adjusted = adjusted.where(~pos_mask, adjusted / pos_sum)

        return adjusted
