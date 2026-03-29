"""
IC类指标计算模块

提供IC（信息系数）、Rank IC、ICIR、Rank ICIR等因子的预测能力指标计算。
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats


def calculate_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = 'pearson'
) -> float:
    """
    计算单期IC（信息系数）
    
    IC = corr(因子值, 未来收益)
    
    Args:
        factor_values: 单期因子值序列 (N,)
        forward_returns: 对应未来收益序列 (N,)
        method: 相关系数方法，'pearson'或'spearman'
        
    Returns:
        float: IC值
        
    Example:
        >>> factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> calculate_ic(factor, returns)
        0.999...
    """
    # 对齐并去除NaN
    df = pd.DataFrame({'factor': factor_values, 'returns': forward_returns}).dropna()
    
    if len(df) < 3:
        return np.nan
    
    if method == 'pearson':
        return df['factor'].corr(df['returns'], method='pearson')
    elif method == 'spearman':
        return df['factor'].corr(df['returns'], method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_ic_series(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'pearson'
) -> pd.Series:
    """
    计算IC时间序列
    
    Args:
        factor_values: 因子值矩阵 (N × T)
        forward_returns: 未来收益矩阵 (N × T)
        method: 相关系数方法
        
    Returns:
        Series: IC时间序列 (T,)
    """
    # 确保列（日期）对齐
    common_dates = factor_values.columns.intersection(forward_returns.columns)
    factor_values = factor_values[common_dates]
    forward_returns = forward_returns[common_dates]
    
    ic_series = []
    for date in common_dates:
        ic = calculate_ic(
            factor_values[date],
            forward_returns[date],
            method
        )
        ic_series.append(ic)
    
    return pd.Series(ic_series, index=common_dates)


def calculate_rank_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame
) -> pd.Series:
    """
    计算Rank IC时间序列（Spearman相关系数）
    
    Rank IC是比IC更稳健的指标，使用排名而非原始值。
    
    Args:
        factor_values: 因子值矩阵 (N × T)
        forward_returns: 未来收益矩阵 (N × T)
        
    Returns:
        Series: Rank IC时间序列
    """
    return calculate_ic_series(factor_values, forward_returns, method='spearman')


def calculate_icir(
    ic_series: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    计算ICIR（信息比率）
    
    ICIR = mean(IC) / std(IC)
    
    Args:
        ic_series: IC时间序列
        annualize: 是否年化
        periods_per_year: 每年的周期数
        
    Returns:
        float: ICIR值
        
    Example:
        >>> ic = pd.Series([0.05, 0.03, 0.04, -0.01, 0.02])
        >>> calculate_icir(ic, annualize=False)
        1.581...
    """
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return np.nan
    
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    
    if std_ic == 0 or np.isnan(std_ic):
        return np.nan
    
    icir = mean_ic / std_ic
    
    if annualize:
        icir = icir * np.sqrt(periods_per_year)
    
    return icir


def calculate_rank_icir(
    rank_ic_series: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    计算Rank ICIR
    
    Args:
        rank_ic_series: Rank IC时间序列
        annualize: 是否年化
        periods_per_year: 每年的周期数
        
    Returns:
        float: Rank ICIR值
    """
    return calculate_icir(rank_ic_series, annualize, periods_per_year)


def calculate_ic_statistics(ic_series: pd.Series) -> dict:
    """
    计算IC的统计指标
    
    Args:
        ic_series: IC时间序列
        
    Returns:
        dict: IC统计指标字典
    """
    ic_series = ic_series.dropna()
    if len(ic_series) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'positive_ratio': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
        }
    
    # t检验
    t_stat, p_value = stats.ttest_1samp(ic_series, 0)
    
    return {
        'mean': ic_series.mean(),
        'std': ic_series.std(),
        'min': ic_series.min(),
        'max': ic_series.max(),
        'median': ic_series.median(),
        'positive_ratio': (ic_series > 0).mean(),
        't_stat': t_stat,
        'p_value': p_value,
    }


def calculate_ic_decay(
    factor_values: pd.DataFrame,
    returns_dict: dict,
    method: str = 'spearman'
) -> pd.Series:
    """
    计算IC衰减（不同前瞻期的IC）
    
    Args:
        factor_values: 因子值矩阵
        returns_dict: 不同前瞻期的收益矩阵字典，如 {1: ret_1d, 5: ret_5d, ...}
        method: 相关系数方法
        
    Returns:
        Series: 各前瞻期的平均IC
    """
    decay_results = {}
    
    for period, forward_returns in returns_dict.items():
        ic_series = calculate_ic_series(factor_values, forward_returns, method)
        decay_results[period] = ic_series.mean()
    
    return pd.Series(decay_results)


def calculate_quantile_returns(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    计算分位数收益（因子分层收益）
    
    将每期因子值分为n_quantiles组，计算每组平均收益。
    
    Args:
        factor_values: 因子值矩阵
        forward_returns: 未来收益矩阵
        n_quantiles: 分位数数量
        
    Returns:
        DataFrame: 各分位数的时间序列收益
    """
    common_dates = factor_values.columns.intersection(forward_returns.columns)
    
    quantile_returns = []
    for date in common_dates:
        f = factor_values[date]
        r = forward_returns[date]
        
        df = pd.DataFrame({'factor': f, 'returns': r}).dropna()
        if len(df) < n_quantiles:
            continue
        
        # 分位数分组
        df['quantile'] = pd.qcut(df['factor'], n_quantiles, labels=False, duplicates='drop')
        
        # 计算每组平均收益
        group_returns = df.groupby('quantile')['returns'].mean()
        quantile_returns.append(group_returns)
    
    if len(quantile_returns) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(quantile_returns, index=common_dates[:len(quantile_returns)])


class ICMetricsCalculator:
    """
    IC类指标计算器
    
    用于计算因子的IC类指标，评估因子的预测能力。
    
    Attributes:
        factor_values: 因子值矩阵 (N × T)
        forward_returns: 未来收益矩阵 (N × T)
        periods_per_year: 每年的周期数
    """
    
    def __init__(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods_per_year: int = 252
    ):
        """
        初始化IC类指标计算器
        
        Args:
            factor_values: 因子值矩阵
            forward_returns: 未来收益矩阵
            periods_per_year: 每年的周期数
        """
        # 对齐
        self.common_dates = factor_values.columns.intersection(forward_returns.columns)
        self.factor_values = factor_values[self.common_dates]
        self.forward_returns = forward_returns[self.common_dates]
        self.periods_per_year = periods_per_year
    
    def ic_series(self) -> pd.Series:
        """计算IC时间序列（Pearson）"""
        return calculate_ic_series(self.factor_values, self.forward_returns, method='pearson')
    
    def rank_ic_series(self) -> pd.Series:
        """计算Rank IC时间序列（Spearman）"""
        return calculate_ic_series(self.factor_values, self.forward_returns, method='spearman')
    
    def mean_ic(self) -> float:
        """计算平均IC"""
        return self.ic_series().mean()
    
    def mean_rank_ic(self) -> float:
        """计算平均Rank IC"""
        return self.rank_ic_series().mean()
    
    def icir(self, annualize: bool = True) -> float:
        """计算ICIR"""
        return calculate_icir(self.ic_series(), annualize, self.periods_per_year)
    
    def rank_icir(self, annualize: bool = True) -> float:
        """计算Rank ICIR"""
        return calculate_icir(self.rank_ic_series(), annualize, self.periods_per_year)
    
    def ic_stats(self) -> dict:
        """计算IC统计指标"""
        return calculate_ic_statistics(self.ic_series())
    
    def rank_ic_stats(self) -> dict:
        """计算Rank IC统计指标"""
        return calculate_ic_statistics(self.rank_ic_series())
    
    def get_all_metrics(self) -> dict:
        """
        获取所有IC类指标
        
        Returns:
            dict: 包含所有IC类指标的字典
        """
        ic_series = self.ic_series()
        rank_ic_series = self.rank_ic_series()
        
        ic_stats = calculate_ic_statistics(ic_series)
        rank_ic_stats = calculate_ic_statistics(rank_ic_series)
        
        return {
            'ic_mean': ic_stats['mean'],
            'ic_std': ic_stats['std'],
            'ic_ir': self.icir(),
            'ic_positive_ratio': ic_stats['positive_ratio'],
            'ic_t_stat': ic_stats['t_stat'],
            'ic_p_value': ic_stats['p_value'],
            'rank_ic_mean': rank_ic_stats['mean'],
            'rank_ic_std': rank_ic_stats['std'],
            'rank_ic_ir': self.rank_icir(),
            'rank_ic_positive_ratio': rank_ic_stats['positive_ratio'],
            'rank_ic_t_stat': rank_ic_stats['t_stat'],
            'rank_ic_p_value': rank_ic_stats['p_value'],
        }


def calculate_forward_returns(
    close_prices: pd.DataFrame,
    periods: Union[int, list] = 1,
    open_prices: Optional[pd.DataFrame] = None,
    execution_price: str = 'close',
) -> Union[pd.DataFrame, dict]:
    """
    计算前瞻收益

    支持两种执行价格模式：

    - 'close'（默认，兼容ETF）：
        T日因子 → T+periods日收盘 vs T日收盘的收益率

    - 'open'（A股T+1开盘执行）：
        T日因子 → T+1日开盘买入 → T+1+periods日开盘卖出的收益率

    Args:
        close_prices: 收盘价矩阵 (N × T)
        periods: 前瞻期数，可以是单个整数或列表
        open_prices: 开盘价矩阵 (N × T)，当 execution_price='open' 时必须提供
        execution_price: 执行价格模式，'close' 或 'open'

    Returns:
        DataFrame 或 dict: 前瞻收益矩阵
    """
    def _compute_single(p: int) -> pd.DataFrame:
        if execution_price == 'open' and open_prices is not None:
            # T+1 日开盘买入，T+1+p 日开盘卖出
            entry = open_prices.shift(-1, axis=1)           # T+1 日开盘
            exit_ = open_prices.shift(-(1 + p), axis=1)    # T+1+p 日开盘
            return exit_ / entry - 1
        else:
            # 原有逻辑：T 日收盘 → T+p 日收盘
            future = close_prices.shift(-p, axis=1)
            return future / close_prices - 1

    if isinstance(periods, int):
        return _compute_single(periods)
    else:
        return {p: _compute_single(p) for p in periods}
