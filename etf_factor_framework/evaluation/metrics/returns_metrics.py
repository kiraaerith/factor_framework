"""
收益类指标计算模块

提供总收益、年化收益等收益类指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def total_return(
    returns: Union[pd.Series, pd.DataFrame],
    axis: Optional[int] = None
) -> float:
    """
    计算总收益率
    
    使用几何复利方式计算总收益率：
    total_return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    
    Args:
        returns: 收益率序列或矩阵
        axis: 计算方向，对于DataFrame需要指定
        
    Returns:
        float: 总收益率
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015])
        >>> total_return(returns)
        0.03978...
    """
    if isinstance(returns, pd.DataFrame):
        if axis is None:
            raise ValueError("axis must be specified for DataFrame input")
        returns = returns.sum(axis=axis) if axis == 1 else returns.sum(axis=axis)
    
    # 处理NaN值
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    return (1 + returns).prod() - 1


def annualized_return(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252,
    axis: Optional[int] = None
) -> float:
    """
    计算年化收益率
    
    公式：annualized_return = (1 + total_return)^(periods_per_year / n_periods) - 1
    
    Args:
        returns: 收益率序列或矩阵
        periods_per_year: 每年的周期数，默认252（日度数据）
        axis: 计算方向，对于DataFrame需要指定
        
    Returns:
        float: 年化收益率
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015])
        >>> annualized_return(returns, periods_per_year=252)
        10.23...  # 假设这是4天的数据，年化后约为10倍
    """
    if isinstance(returns, pd.DataFrame):
        if axis is None:
            raise ValueError("axis must be specified for DataFrame input")
        # 对DataFrame，假设每一列是一个时间序列
        if axis == 0:
            returns = returns.sum(axis=0)
        else:
            returns = returns.sum(axis=1)
    
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    n_periods = len(returns)
    total_ret = total_return(returns)
    
    if total_ret <= -1:
        return -1.0
    
    return (1 + total_ret) ** (periods_per_year / n_periods) - 1


def cumulative_returns(returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    计算累积收益率序列
    
    Args:
        returns: 收益率序列或矩阵
        
    Returns:
        Series/DataFrame: 累积收益率序列
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015])
        >>> cumulative_returns(returns)
        0    0.010000
        1    0.004950
        2    0.025050
        3    0.039726
        dtype: float64
    """
    if isinstance(returns, pd.DataFrame):
        return (1 + returns).cumprod(axis=1) - 1
    return (1 + returns).cumprod() - 1


def average_return(
    returns: Union[pd.Series, pd.DataFrame],
    axis: Optional[int] = None
) -> float:
    """
    计算平均收益率（算术平均）
    
    Args:
        returns: 收益率序列或矩阵
        axis: 计算方向
        
    Returns:
        float: 平均收益率
    """
    if isinstance(returns, pd.DataFrame):
        if axis is None:
            raise ValueError("axis must be specified for DataFrame input")
        returns = returns.mean(axis=axis)
    
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    return returns.mean()


class ReturnsMetricsCalculator:
    """
    收益类指标计算器
    
    用于计算组合的收益类指标。
    
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
        初始化收益类指标计算器
        
        Args:
            returns: 组合收益率序列
            periods_per_year: 每年的周期数，默认252（日度数据）
        """
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year
        self.n_periods = len(self.returns)
    
    def total_return(self) -> float:
        """计算总收益率"""
        return total_return(self.returns)
    
    def annualized_return(self) -> float:
        """计算年化收益率"""
        return annualized_return(self.returns, self.periods_per_year)
    
    def cumulative_returns(self) -> pd.Series:
        """计算累积收益率序列"""
        return cumulative_returns(self.returns)
    
    def average_return(self) -> float:
        """计算平均收益率"""
        return average_return(self.returns)
    
    def get_all_metrics(self) -> dict:
        """
        获取所有收益类指标
        
        Returns:
            dict: 包含所有收益类指标的字典
        """
        return {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'average_return': self.average_return(),
        }


def calculate_portfolio_returns(
    position_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    delay: int = 1
) -> pd.Series:
    """
    计算组合收益率序列
    
    基于仓位权重和资产收益率计算组合收益率。
    
    Args:
        position_weights: 仓位权重矩阵 (N × T)，index=symbol, columns=date
        asset_returns: 资产收益率矩阵 (N × T)，index=symbol, columns=date
        delay: 调仓延迟，默认1（T日的仓位在T+1日产生收益）
        
    Returns:
        Series: 组合收益率序列 (T,)
        
    Example:
        >>> weights = pd.DataFrame(
        ...     [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
        ...     index=['A', 'B'],
        ...     columns=pd.date_range('2024-01-01', periods=3)
        ... )
        >>> returns = pd.DataFrame(
        ...     [[0.01, 0.02, -0.01], [0.015, -0.01, 0.005]],
        ...     index=['A', 'B'],
        ...     columns=pd.date_range('2024-01-01', periods=3)
        ... )
        >>> portfolio_returns = calculate_portfolio_returns(weights, returns)
    """
    # 确保对齐
    common_symbols = position_weights.index.intersection(asset_returns.index)
    common_dates = position_weights.columns.intersection(asset_returns.columns)
    
    weights = position_weights.loc[common_symbols, common_dates]
    returns = asset_returns.loc[common_symbols, common_dates]
    
    # 应用延迟：T日的仓位在T+delay日产生收益
    if delay > 0:
        weights = weights.shift(delay, axis=1)
    
    # 计算每日组合收益 = sum(weight_i * return_i)
    # 注意：weights和returns需要对齐
    portfolio_returns = (weights * returns).sum(axis=0)
    
    # 去掉第一个delay个NaN
    portfolio_returns = portfolio_returns.dropna()
    
    return portfolio_returns
