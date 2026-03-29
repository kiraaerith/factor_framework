"""
风险调整收益指标计算模块

提供夏普比率、Calmar比率、Sortino比率等风险调整收益指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

from .returns_metrics import annualized_return
from .risk_metrics import annualized_volatility, max_drawdown, downside_volatility


def sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    计算夏普比率
    
    公式：Sharpe = (年化收益 - 无风险利率) / 年化波动率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的周期数
        
    Returns:
        float: 夏普比率
        
    Example:
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
        >>> sharpe_ratio(returns)
        0.397...
    """
    if isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Series")
    
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    
    return (ann_return - risk_free_rate) / ann_vol


def calmar_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252
) -> float:
    """
    计算Calmar比率
    
    公式：Calmar = 年化收益 / |最大回撤|
    
    Args:
        returns: 收益率序列
        periods_per_year: 每年的周期数
        
    Returns:
        float: Calmar比率
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.02, 0.01])
        >>> calmar_ratio(returns)
        1.25...
    """
    if isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Series")
    
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    
    if mdd >= 0 or mdd == 0:  # 没有回撤或全为0
        return np.nan if ann_return <= 0 else np.inf
    
    return ann_return / abs(mdd)


def sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    计算Sortino比率
    
    公式：Sortino = (年化收益 - 无风险利率) / 下行波动率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的周期数
        target_return: 目标收益率（用于计算下行波动率）
        
    Returns:
        float: Sortino比率
    """
    if isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a Series")
    
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    downside_vol = downside_volatility(returns, periods_per_year, target_return)
    
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan
    
    return (ann_return - risk_free_rate) / downside_vol


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    计算信息比率
    
    公式：IR = (组合收益 - 基准收益) / 跟踪误差
    
    Args:
        returns: 组合收益率序列
        benchmark_returns: 基准收益率序列
        periods_per_year: 每年的周期数
        
    Returns:
        float: 信息比率
    """
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
    # 对齐
    common_index = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    if len(returns) < 2:
        return np.nan
    
    # 超额收益
    excess_returns = returns - benchmark_returns
    
    # 年化超额收益
    ann_excess_return = annualized_return(excess_returns, periods_per_year)
    
    # 跟踪误差（年化）
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan
    
    return ann_excess_return / tracking_error


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    计算Treynor比率
    
    公式：Treynor = (年化收益 - 无风险利率) / Beta
    
    Args:
        returns: 组合收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的周期数
        
    Returns:
        float: Treynor比率
    """
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
    common_index = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    if len(returns) < 2:
        return np.nan
    
    # 计算Beta
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return np.nan
    
    beta = covariance / benchmark_variance
    
    if beta == 0:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    
    return (ann_return - risk_free_rate) / beta


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    计算Omega比率
    
    Omega = 正收益和 / |负收益和|
    
    Args:
        returns: 收益率序列
        threshold: 阈值，默认0
        
    Returns:
        float: Omega比率
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    excess = returns - threshold
    positive_sum = excess[excess > 0].sum()
    negative_sum = abs(excess[excess < 0].sum())
    
    if negative_sum == 0:
        return np.inf if positive_sum > 0 else np.nan
    
    return positive_sum / negative_sum


def sterling_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    计算Sterling比率
    
    Sterling = 年化收益 / 平均回撤
    
    Args:
        returns: 收益率序列
        periods_per_year: 每年的周期数
        
    Returns:
        float: Sterling比率
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    ann_return = annualized_return(returns, periods_per_year)
    
    # 计算回撤序列
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (running_max - cumulative) / running_max
    
    avg_drawdown = drawdowns[drawdowns > 0].mean()
    
    if avg_drawdown == 0 or np.isnan(avg_drawdown):
        return np.nan
    
    return ann_return / avg_drawdown


class RiskAdjustedMetricsCalculator:
    """
    风险调整收益指标计算器
    
    用于计算组合的风险调整收益指标。
    """
    
    def __init__(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
        benchmark_returns: Optional[pd.Series] = None
    ):
        """
        初始化风险调整收益指标计算器
        
        Args:
            returns: 组合收益率序列
            periods_per_year: 每年的周期数
            risk_free_rate: 无风险利率
            benchmark_returns: 基准收益率序列（可选）
        """
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
    
    def sharpe_ratio(self) -> float:
        """计算夏普比率"""
        return sharpe_ratio(self.returns, self.risk_free_rate, self.periods_per_year)
    
    def calmar_ratio(self) -> float:
        """计算Calmar比率"""
        return calmar_ratio(self.returns, self.periods_per_year)
    
    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """计算Sortino比率"""
        return sortino_ratio(self.returns, self.risk_free_rate, self.periods_per_year, target_return)
    
    def information_ratio(self) -> float:
        """计算信息比率（需要基准）"""
        if self.benchmark_returns is None:
            return np.nan
        return information_ratio(self.returns, self.benchmark_returns, self.periods_per_year)
    
    def treynor_ratio(self) -> float:
        """计算Treynor比率（需要基准）"""
        if self.benchmark_returns is None:
            return np.nan
        return treynor_ratio(self.returns, self.benchmark_returns, self.risk_free_rate, self.periods_per_year)
    
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """计算Omega比率"""
        return omega_ratio(self.returns, threshold)
    
    def sterling_ratio(self) -> float:
        """计算Sterling比率"""
        return sterling_ratio(self.returns, self.periods_per_year)
    
    def get_all_metrics(self) -> dict:
        """
        获取所有风险调整收益指标
        
        Returns:
            dict: 包含所有风险调整收益指标的字典
        """
        metrics = {
            'sharpe_ratio': self.sharpe_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'omega_ratio': self.omega_ratio(),
            'sterling_ratio': self.sterling_ratio(),
        }
        
        if self.benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio()
            metrics['treynor_ratio'] = self.treynor_ratio()
        
        return metrics
