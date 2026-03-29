"""
换手率指标计算模块

提供日均换手率、换手率序列等指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Union


def turnover_rate(
    position_weights: pd.DataFrame,
    normalize: bool = True
) -> pd.Series:
    """
    计算换手率序列
    
    换手率 = sum(|w_t - w_{t-1}|) / 2
    
    Args:
        position_weights: 仓位权重矩阵 (N × T)
        normalize: 是否归一化（除以2）
        
    Returns:
        Series: 换手率序列 (T-1,)
    """
    # 计算权重变化
    weight_diff = position_weights.diff(axis=1).iloc[:, 1:]  # 去掉第一列NaN
    
    # 计算每日换手率 = sum(|delta_weight|)
    turnover = weight_diff.abs().sum(axis=0)
    
    if normalize:
        # 标准定义：双边换手率需要除以2
        turnover = turnover / 2.0
    
    return turnover


def average_turnover(
    position_weights: pd.DataFrame,
    normalize: bool = True
) -> float:
    """
    计算平均换手率（日均换手率）
    
    Args:
        position_weights: 仓位权重矩阵 (N × T)
        normalize: 是否归一化
        
    Returns:
        float: 平均换手率
        
    Example:
        >>> weights = pd.DataFrame(
        ...     [[0.5, 0.3, 0.2], [0.5, 0.7, 0.8]],
        ...     index=['A', 'B'],
        ...     columns=pd.date_range('2024-01-01', periods=3)
        ... )
        >>> average_turnover(weights)
        0.15
    """
    turnover = turnover_rate(position_weights, normalize)
    return turnover.mean()


def annualized_turnover(
    position_weights: pd.DataFrame,
    periods_per_year: int = 252,
    normalize: bool = True
) -> float:
    """
    计算年化换手率
    
    Args:
        position_weights: 仓位权重矩阵
        periods_per_year: 每年的周期数
        normalize: 是否归一化
        
    Returns:
        float: 年化换手率
    """
    avg_turnover = average_turnover(position_weights, normalize)
    return avg_turnover * periods_per_year


def turnover_volatility(
    position_weights: pd.DataFrame,
    normalize: bool = True
) -> float:
    """
    计算换手率的波动率
    
    Args:
        position_weights: 仓位权重矩阵
        normalize: 是否归一化
        
    Returns:
        float: 换手率波动率
    """
    turnover = turnover_rate(position_weights, normalize)
    return turnover.std()


def position_change_count(
    position_weights: pd.DataFrame,
    threshold: float = 1e-10
) -> pd.Series:
    """
    计算每日持仓变化数量
    
    Args:
        position_weights: 仓位权重矩阵
        threshold: 判定为变化的阈值
        
    Returns:
        Series: 每日持仓变化数量
    """
    weight_diff = position_weights.diff(axis=1).iloc[:, 1:]
    changes = (weight_diff.abs() > threshold).sum(axis=0)
    return changes


def active_position_ratio(
    position_weights: pd.DataFrame,
    threshold: float = 1e-10
) -> float:
    """
    计算平均持仓比例（非零仓位占比）
    
    Args:
        position_weights: 仓位权重矩阵
        threshold: 判定为有持仓的阈值
        
    Returns:
        float: 平均持仓比例
    """
    active = (position_weights.abs() > threshold).sum(axis=0)
    n_assets = position_weights.shape[0]
    return (active / n_assets).mean()


class TurnoverMetricsCalculator:
    """
    换手率指标计算器
    
    用于计算组合的换手率相关指标。
    """
    
    def __init__(
        self,
        position_weights: pd.DataFrame,
        periods_per_year: int = 252
    ):
        """
        初始化换手率指标计算器
        
        Args:
            position_weights: 仓位权重矩阵 (N × T)
            periods_per_year: 每年的周期数
        """
        self.position_weights = position_weights
        self.periods_per_year = periods_per_year
    
    def turnover_rate(self) -> pd.Series:
        """计算换手率序列"""
        return turnover_rate(self.position_weights)
    
    def average_turnover(self) -> float:
        """计算平均换手率（日均换手率）"""
        return average_turnover(self.position_weights)
    
    def annualized_turnover(self) -> float:
        """计算年化换手率"""
        return annualized_turnover(self.position_weights, self.periods_per_year)
    
    def turnover_volatility(self) -> float:
        """计算换手率波动率"""
        return turnover_volatility(self.position_weights)
    
    def position_change_count(self) -> pd.Series:
        """计算每日持仓变化数量"""
        return position_change_count(self.position_weights)
    
    def active_position_ratio(self) -> float:
        """计算平均持仓比例"""
        return active_position_ratio(self.position_weights)
    
    def get_all_metrics(self) -> dict:
        """
        获取所有换手率指标
        
        Returns:
            dict: 包含所有换手率指标的字典
        """
        return {
            'avg_daily_turnover': self.average_turnover(),
            'annualized_turnover': self.annualized_turnover(),
            'turnover_volatility': self.turnover_volatility(),
            'active_position_ratio': self.active_position_ratio(),
        }


def calculate_transaction_costs(
    position_weights: pd.DataFrame,
    commission_rate: float = 0.0002,
    slippage_rate: float = 0.0
) -> pd.Series:
    """
    计算交易成本序列
    
    交易成本 = 换手率 × (手续费率 + 滑点率)
    
    Args:
        position_weights: 仓位权重矩阵
        commission_rate: 单边手续费率
        slippage_rate: 单边滑点率
        
    Returns:
        Series: 每日交易成本
    """
    turnover = turnover_rate(position_weights, normalize=False)
    total_cost_rate = commission_rate * 2 + slippage_rate * 2  # 双边
    return turnover * total_cost_rate


def turnover_attribution(
    position_weights: pd.DataFrame,
    threshold: float = 1e-10
) -> dict:
    """
    换手率归因分析
    
    分析换手率的来源：新增仓位、减少仓位、完全平仓、新开仓
    
    Args:
        position_weights: 仓位权重矩阵
        threshold: 判定为有持仓的阈值
        
    Returns:
        dict: 换手率归因结果
    """
    weight_diff = position_weights.diff(axis=1).iloc[:, 1:]
    
    # 新增仓位（权重增加）
    increased = weight_diff[weight_diff > 0].sum().sum() / weight_diff.shape[1]
    
    # 减少仓位（权重减少）
    decreased = weight_diff[weight_diff < 0].abs().sum().sum() / weight_diff.shape[1]
    
    # 新增持仓（从0变为非0）
    prev_weights = position_weights.shift(1, axis=1).iloc[:, 1:]
    new_entries = ((prev_weights.abs() <= threshold) & (position_weights.iloc[:, 1:].abs() > threshold))
    new_entries_pct = new_entries.sum().sum() / (position_weights.shape[0] * weight_diff.shape[1])
    
    # 完全平仓（从非0变为0）
    exits = ((prev_weights.abs() > threshold) & (position_weights.iloc[:, 1:].abs() <= threshold))
    exits_pct = exits.sum().sum() / (position_weights.shape[0] * weight_diff.shape[1])
    
    return {
        'increased_weight': increased,
        'decreased_weight': decreased,
        'new_entries_ratio': new_entries_pct,
        'exits_ratio': exits_pct,
    }
