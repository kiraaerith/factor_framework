"""
风险类指标计算模块

提供最大回撤、年化波动率等风险类指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def max_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> float:
    """
    计算最大回撤
    
    最大回撤 = (峰值 - 谷底) / 峰值
    
    Args:
        returns: 收益率序列
        
    Returns:
        float: 最大回撤（负值）
        
    Example:
        >>> returns = pd.Series([0.1, -0.05, 0.08, -0.12, 0.03])
        >>> max_drawdown(returns)
        -0.1496...  # 约-15%
    """
    if isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Series for max_drawdown calculation")
    
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    # 计算累积收益
    cumulative = (1 + returns).cumprod()
    
    # 计算滚动最大值（历史峰值）
    running_max = cumulative.expanding().max()
    
    # 计算回撤
    drawdown = (cumulative - running_max) / running_max
    
    # 返回最大回撤（最小值，即最负的值）
    return drawdown.min()


def max_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    计算回撤序列
    
    Args:
        returns: 收益率序列
        
    Returns:
        Series: 回撤序列
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return pd.Series(dtype=float)
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown


def drawdown_duration(returns: pd.Series) -> int:
    """
    计算最长回撤持续期
    
    Args:
        returns: 收益率序列
        
    Returns:
        int: 最长回撤持续期（周期数）
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    
    # 标记是否在回撤中
    in_drawdown = cumulative < running_max
    
    # 计算连续回撤期
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252
) -> float:
    """
    计算年化波动率
    
    Args:
        returns: 收益率序列
        periods_per_year: 每年的周期数，默认252（日度数据）
        
    Returns:
        float: 年化波动率
        
    Example:
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        >>> annualized_volatility(returns)
        0.315...  # 约31.5%
    """
    if isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Series for volatility calculation")
    
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    return returns.std() * np.sqrt(periods_per_year)


def downside_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    计算下行波动率（半方差）
    
    只考虑低于目标收益的收益率的波动率。
    
    Args:
        returns: 收益率序列
        periods_per_year: 每年的周期数
        target_return: 目标收益率，默认0
        
    Returns:
        float: 下行波动率
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    # 只取低于目标的收益
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) < 2:
        return 0.0
    
    return downside_returns.std() * np.sqrt(periods_per_year)


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> float:
    """
    计算风险价值（VaR）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平，默认0.05（95% VaR）
        
    Returns:
        float: VaR值（负值）
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    return np.percentile(returns, confidence_level * 100)


def conditional_var(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> float:
    """
    计算条件风险价值（CVaR/Expected Shortfall）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
        
    Returns:
        float: CVaR值（负值）
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


class RiskMetricsCalculator:
    """
    风险类指标计算器
    
    用于计算组合的风险类指标。
    
    Attributes:
        returns: 组合收益率序列 (T,)
        periods_per_year: 每年的周期数
    """
    
    def __init__(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ):
        """
        初始化风险类指标计算器
        
        Args:
            returns: 组合收益率序列
            periods_per_year: 每年的周期数
        """
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year
        self.n_periods = len(self.returns)
    
    def max_drawdown(self) -> float:
        """计算最大回撤"""
        return max_drawdown(self.returns)
    
    def max_drawdown_series(self) -> pd.Series:
        """计算回撤序列"""
        return max_drawdown_series(self.returns)
    
    def drawdown_duration(self) -> int:
        """计算最长回撤持续期"""
        return drawdown_duration(self.returns)
    
    def annualized_volatility(self) -> float:
        """计算年化波动率"""
        return annualized_volatility(self.returns, self.periods_per_year)
    
    def downside_volatility(self, target_return: float = 0.0) -> float:
        """计算下行波动率"""
        return downside_volatility(self.returns, self.periods_per_year, target_return)
    
    def value_at_risk(self, confidence_level: float = 0.05) -> float:
        """计算VaR"""
        return value_at_risk(self.returns, confidence_level)
    
    def conditional_var(self, confidence_level: float = 0.05) -> float:
        """计算CVaR"""
        return conditional_var(self.returns, confidence_level)
    
    def get_all_metrics(self) -> dict:
        """
        获取所有风险类指标
        
        Returns:
            dict: 包含所有风险类指标的字典
        """
        return {
            'max_drawdown': self.max_drawdown(),
            'annualized_volatility': self.annualized_volatility(),
            'downside_volatility': self.downside_volatility(),
            'var_95': self.value_at_risk(0.05),
            'cvar_95': self.conditional_var(0.05),
            'drawdown_duration': self.drawdown_duration(),
        }
