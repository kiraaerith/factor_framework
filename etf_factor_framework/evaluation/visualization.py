"""
可视化模块

提供因子评估相关的可视化图表功能。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_cumulative_ic(
    ic_series: pd.Series,
    title: str = "累积IC曲线",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制累积IC曲线
    
    Args:
        ic_series: IC时间序列
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算累积IC
    cumsum_ic = ic_series.cumsum()
    
    ax.plot(cumsum_ic.index, cumsum_ic.values, linewidth=1.5, color='#2E86AB')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    # 填充正负区域
    ax.fill_between(cumsum_ic.index, 0, cumsum_ic.values, 
                    where=cumsum_ic.values >= 0, alpha=0.3, color='green', label='Positive')
    ax.fill_between(cumsum_ic.index, 0, cumsum_ic.values,
                    where=cumsum_ic.values < 0, alpha=0.3, color='red', label='Negative')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative IC', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    title: str = "IC分布直方图",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制IC分布直方图
    
    Args:
        ic_series: IC时间序列
        title: 图表标题
        figsize: 图表大小
        bins: 直方图柱数
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 去除NaN
    ic_clean = ic_series.dropna()
    
    # 绘制直方图
    ax.hist(ic_clean.values, bins=bins, edgecolor='black', alpha=0.7, color='#4A90E2')
    
    # 添加均值线
    mean_ic = ic_clean.mean()
    ax.axvline(mean_ic, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ic:.4f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('IC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cumulative_returns(
    returns: pd.Series,
    title: str = "累积收益曲线",
    figsize: Tuple[int, int] = (12, 6),
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制累积收益曲线
    
    Args:
        returns: 收益率序列
        title: 图表标题
        figsize: 图表大小
        benchmark_returns: 基准收益率序列（可选）
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算累积收益
    cum_returns = (1 + returns).cumprod() - 1
    
    ax.plot(cum_returns.index, cum_returns.values * 100, 
            linewidth=1.5, label='Strategy', color='#2E86AB')
    
    # 添加基准
    if benchmark_returns is not None:
        # 对齐
        common_idx = cum_returns.index.intersection(benchmark_returns.index)
        bench_cum = (1 + benchmark_returns.loc[common_idx]).cumprod() - 1
        ax.plot(bench_cum.index, bench_cum.values * 100,
                linewidth=1.5, label='Benchmark', color='gray', linestyle='--')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 添加Y轴百分比格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "回撤曲线",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制回撤曲线
    
    Args:
        returns: 收益率序列
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    color='red', alpha=0.5, label='Drawdown')
    
    ax.plot(drawdown.index, drawdown.values * 100, 
            linewidth=1, color='darkred')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 标记最大回撤
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax.scatter([max_dd_idx], [max_dd * 100], color='darkred', s=100, zorder=5)
    ax.annotate(f'Max DD: {max_dd*100:.2f}%',
                xy=(max_dd_idx, max_dd * 100),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y轴百分比格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_and_drawdown(
    returns: pd.Series,
    title: str = "收益与回撤",
    figsize: Tuple[int, int] = (12, 8),
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制收益和回撤组合图（上下两个子图）
    
    Args:
        returns: 收益率序列
        title: 图表标题
        figsize: 图表大小
        benchmark_returns: 基准收益率序列（可选）
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # 上子图：累积收益
    cum_returns = (1 + returns).cumprod() - 1
    ax1.plot(cum_returns.index, cum_returns.values * 100,
             linewidth=1.5, label='Strategy', color='#2E86AB')
    
    if benchmark_returns is not None:
        common_idx = cum_returns.index.intersection(benchmark_returns.index)
        bench_cum = (1 + benchmark_returns.loc[common_idx]).cumprod() - 1
        ax1.plot(bench_cum.index, bench_cum.values * 100,
                linewidth=1.5, label='Benchmark', color='gray', linestyle='--')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 下子图：回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color='red', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values * 100,
             linewidth=1, color='darkred')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_factor_distribution(
    factor_values: Union[pd.Series, pd.DataFrame],
    title: str = "因子值分布",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制因子值分布直方图
    
    Args:
        factor_values: 因子值（Series或DataFrame）
        title: 图表标题
        figsize: 图表大小
        bins: 直方图柱数
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 展平数据
    if isinstance(factor_values, pd.DataFrame):
        values = factor_values.values.flatten()
    else:
        values = factor_values.values
    
    # 去除NaN和Inf
    values = values[np.isfinite(values)]
    
    # 限制极端值以便更好展示
    q1, q99 = np.percentile(values, [1, 99])
    values_clipped = np.clip(values, q1, q99)
    
    ax.hist(values_clipped, bins=bins, edgecolor='black', alpha=0.7, color='#4A90E2')
    
    # 添加统计信息
    mean_val = np.mean(values)
    median_val = np.median(values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.4f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Factor Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_position_returns_distribution(
    position_returns: pd.Series,
    title: str = "仓位收益分布",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制仓位收益分布直方图
    
    Args:
        position_returns: 每日组合收益序列
        title: 图表标题
        figsize: 图表大小
        bins: 直方图柱数
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    returns_clean = position_returns.dropna()
    values = returns_clean.values * 100  # 转为百分比
    
    ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='#50C878')
    
    # 添加统计信息
    mean_ret = np.mean(values)
    std_ret = np.std(values)
    ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_ret:.3f}%')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加文本信息
    textstr = f'Mean: {mean_ret:.3f}%\nStd: {std_ret:.3f}%\nSkew: {pd.Series(values).skew():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_quantile_returns(
    quantile_returns: pd.DataFrame,
    title: str = "分位数收益",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制因子分位数收益图
    
    Args:
        quantile_returns: 各分位数的收益DataFrame
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_quantiles = len(quantile_returns.columns)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_quantiles))
    
    for i, col in enumerate(quantile_returns.columns):
        cum_ret = (1 + quantile_returns[col]).cumprod() - 1
        label = f'Q{int(col)+1}' if isinstance(col, (int, float)) else str(col)
        ax.plot(cum_ret.index, cum_ret.values * 100,
                linewidth=1.5, label=label, color=colors[i])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(title='Quantile', loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rolling_ic(
    ic_series: pd.Series,
    window: int = 60,
    title: str = "滚动IC",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制滚动IC图
    
    Args:
        ic_series: IC时间序列
        window: 滚动窗口大小
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算滚动均值和标准差
    rolling_mean = ic_series.rolling(window=window).mean()
    rolling_std = ic_series.rolling(window=window).std()
    
    ax.plot(ic_series.index, ic_series.values, 
            alpha=0.3, color='gray', label='IC')
    ax.plot(rolling_mean.index, rolling_mean.values,
            linewidth=2, color='#2E86AB', label=f'{window}-period MA')
    
    # 添加置信区间
    ax.fill_between(rolling_mean.index,
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.2, color='#2E86AB', label='±1 Std')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=rolling_mean.mean(), color='red', linestyle='--', 
               linewidth=1, label=f'Mean: {rolling_mean.mean():.4f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('IC', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_evaluation_report_figure(
    ic_series: pd.Series,
    returns: pd.Series,
    factor_values: pd.DataFrame,
    title: str = "因子评估报告",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建综合评估报告图（多子图）
    
    Args:
        ic_series: IC时间序列
        returns: 组合收益序列
        factor_values: 因子值矩阵
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        Figure: matplotlib图表对象
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 累积IC
    ax1 = fig.add_subplot(gs[0, 0])
    cumsum_ic = ic_series.cumsum()
    ax1.plot(cumsum_ic.index, cumsum_ic.values, color='#2E86AB', linewidth=1.5)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax1.fill_between(cumsum_ic.index, 0, cumsum_ic.values,
                     where=cumsum_ic.values >= 0, alpha=0.3, color='green')
    ax1.fill_between(cumsum_ic.index, 0, cumsum_ic.values,
                     where=cumsum_ic.values < 0, alpha=0.3, color='red')
    ax1.set_title('累积IC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. IC分布
    ax2 = fig.add_subplot(gs[0, 1])
    ic_clean = ic_series.dropna()
    ax2.hist(ic_clean.values, bins=30, edgecolor='black', alpha=0.7, color='#4A90E2')
    ax2.axvline(ic_clean.mean(), color='red', linestyle='--', linewidth=2)
    ax2.set_title('IC分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 收益曲线
    ax3 = fig.add_subplot(gs[1, 0])
    cum_returns = (1 + returns).cumprod() - 1
    ax3.plot(cum_returns.index, cum_returns.values * 100, color='#50C878', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax3.set_title('累积收益', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 回撤曲线
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax4.fill_between(drawdown.index, drawdown.values * 100, 0, color='red', alpha=0.5)
    ax4.plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1)
    ax4.set_title('回撤', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 因子分布
    ax5 = fig.add_subplot(gs[2, 0])
    values = factor_values.values.flatten()
    values = values[np.isfinite(values)]
    q1, q99 = np.percentile(values, [1, 99])
    values_clipped = np.clip(values, q1, q99)
    ax5.hist(values_clipped, bins=50, edgecolor='black', alpha=0.7, color='#9B59B6')
    ax5.set_title('因子值分布', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 月度收益分布
    ax6 = fig.add_subplot(gs[2, 1])
    monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum()
    ax6.bar(range(len(monthly_returns)), monthly_returns.values * 100,
            color=['green' if r > 0 else 'red' for r in monthly_returns.values],
            alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.set_title('月度收益', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Return (%)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
