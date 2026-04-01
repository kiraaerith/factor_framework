"""
仓位调整器模块

根据A股交易限制（停牌、涨跌停）调整目标仓位，使回测更接近真实场景。

调整规则：
  - 想买入但涨停/停牌：不买入，维持 0（或原有持仓）
  - 想卖出但跌停/停牌：不卖出，维持原有持仓
  - 调整后对剩余可执行权重重新归一化到总和为 1

blocked_buy_mode / blocked_sell_mode 控制约束处理策略：
  - 'wait_rebalance': 约束日放弃操作，等下一个调仓日重新评估（默认买入行为）
  - 'asap':           约束日起逐日检查，第一个可交易日立刻执行（默认卖出行为）
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

        adjuster = PositionAdjuster(
            trade_context, delay=1,
            blocked_buy_mode='wait_rebalance',
            blocked_sell_mode='asap',
        )
        actual_position = adjuster.adjust(target_position, rebalance_freq=5)
    """

    def __init__(
        self,
        trade_context: TradeContext,
        delay: int = 1,
        blocked_buy_mode: str = "wait_rebalance",
        blocked_sell_mode: str = "asap",
    ):
        """
        Args:
            trade_context: 交易上下文（包含 can_buy / can_sell 掩码）
            delay: 执行延迟（T日信号在 T+delay 日执行），默认 1
            blocked_buy_mode: 买入受阻时的策略
                - 'wait_rebalance': 等下次调仓重新评估（默认）
                - 'asap': 逐日检查，可买入且仍在目标持仓中时立刻执行
            blocked_sell_mode: 卖出受阻时的策略
                - 'wait_rebalance': 等下次调仓重新评估
                - 'asap': 逐日检查，可卖出时立刻执行（默认）
        """
        valid_modes = ("wait_rebalance", "asap")
        if blocked_buy_mode not in valid_modes:
            raise ValueError(f"blocked_buy_mode must be one of {valid_modes}")
        if blocked_sell_mode not in valid_modes:
            raise ValueError(f"blocked_sell_mode must be one of {valid_modes}")

        self.trade_context = trade_context
        self.delay = delay
        self.blocked_buy_mode = blocked_buy_mode
        self.blocked_sell_mode = blocked_sell_mode

    def adjust(
        self,
        target_position: pd.DataFrame,
        rebalance_freq: int = 1,
    ) -> pd.DataFrame:
        """
        根据交易约束调整目标仓位，返回实际可执行持仓矩阵

        Args:
            target_position: 目标仓位矩阵 (N×T)，来自因子映射器
            rebalance_freq: 调仓频率（每 N 根 K 线调仓一次）

        Returns:
            实际持仓矩阵 (N×T)，已根据约束调整并归一化
        """
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

        # 调仓日标记
        rebalance_mask = [False] * n_periods
        for i in range(0, n_periods, rebalance_freq):
            rebalance_mask[i] = True

        # 待执行队列（asap 模式用）
        # pending_buy: {symbol: target_weight} — 等待买入的股票及其目标权重
        # pending_sell: set of symbols — 等待卖出的股票
        pending_buy = {}
        pending_sell = set()
        # 当期目标持仓（用于 asap 买入时判断股票是否仍在持仓范围内）
        current_target = pd.Series(0.0, index=target_position.index)

        for i, date in enumerate(dates_list):
            if rebalance_mask[i]:
                # --- 调仓日：清空待执行队列，重新评估 ---
                pending_buy.clear()
                pending_sell.clear()
                current_target = target_position[date].copy()

                target = current_target.copy()
                exec_idx = i + self.delay

                if exec_idx < n_periods:
                    exec_date = dates_list[exec_idx]
                    adjusted, new_pending_buy, new_pending_sell = (
                        self._apply_constraints_with_pending(
                            target, current_held, exec_date, can_buy, can_sell
                        )
                    )
                    current_held = adjusted

                    # 根据模式决定是否加入待执行队列
                    if self.blocked_buy_mode == "asap" and new_pending_buy:
                        pending_buy.update(new_pending_buy)
                    if self.blocked_sell_mode == "asap" and new_pending_sell:
                        pending_sell.update(new_pending_sell)
                else:
                    current_held = target

            else:
                # --- 非调仓日：处理 asap 待执行队列 ---
                exec_idx = i + self.delay
                if exec_idx < n_periods and (pending_buy or pending_sell):
                    exec_date = dates_list[exec_idx]
                    changed = False

                    # 处理待卖出
                    if pending_sell:
                        resolved_sell = set()
                        for sym in pending_sell:
                            if exec_date in can_sell.columns and can_sell.at[sym, exec_date]:
                                current_held[sym] = 0.0
                                resolved_sell.add(sym)
                                changed = True
                        pending_sell -= resolved_sell

                    # 处理待买入
                    if pending_buy:
                        resolved_buy = set()
                        for sym, tgt_w in list(pending_buy.items()):
                            if exec_date not in can_buy.columns:
                                continue
                            if not can_buy.at[sym, exec_date]:
                                continue
                            # 检查该股票是否仍在当期目标持仓中
                            if current_target[sym] > 1e-9:
                                current_held[sym] = tgt_w
                                resolved_buy.add(sym)
                                changed = True
                            else:
                                resolved_buy.add(sym)  # 已不在目标中，取消
                        for sym in resolved_buy:
                            pending_buy.pop(sym, None)

                    # 归一化
                    if changed:
                        current_held = self._normalize(current_held)

            result[date] = current_held

        return result

    def _apply_constraints_with_pending(
        self,
        target: pd.Series,
        prev_held: pd.Series,
        exec_date,
        can_buy: pd.DataFrame,
        can_sell: pd.DataFrame,
    ):
        """
        对单个调仓日应用交易约束，同时返回无法执行的待买入/待卖出列表

        Returns:
            (adjusted_position, pending_buy_dict, pending_sell_set)
        """
        pending_buy = {}
        pending_sell = set()

        if exec_date not in can_buy.columns:
            return target, pending_buy, pending_sell

        can_buy_today = can_buy[exec_date].reindex(target.index, fill_value=True)
        can_sell_today = can_sell[exec_date].reindex(target.index, fill_value=True)

        buy_intent = target > prev_held + 1e-9
        sell_intent = target < prev_held - 1e-9

        cant_buy = buy_intent & ~can_buy_today
        cant_sell = sell_intent & ~can_sell_today

        adjusted = target.copy()
        # 买不进：维持原持仓（0 或之前的值）
        adjusted[cant_buy] = prev_held[cant_buy]
        # 卖不出：维持原持仓
        adjusted[cant_sell] = prev_held[cant_sell]

        # 记录待执行
        for sym in cant_buy[cant_buy].index:
            pending_buy[sym] = target[sym]

        for sym in cant_sell[cant_sell].index:
            pending_sell.add(sym)

        # 归一化
        adjusted = self._normalize(adjusted)

        return adjusted, pending_buy, pending_sell

    @staticmethod
    def _normalize(position: pd.Series) -> pd.Series:
        """正权重归一化到总和为 1"""
        pos_mask = position > 1e-9
        pos_sum = position[pos_mask].sum()
        if pos_sum > 1e-9:
            position = position.where(~pos_mask, position / pos_sum)
        return position
