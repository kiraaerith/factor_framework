"""
评估模块

提供因子评估的完整功能，包括各类指标计算和可视化。

主要组件：
- metrics: 各类评估指标计算
- visualization: 可视化图表
- evaluator: 主评估器类

使用示例：
    >>> from evaluation import FactorEvaluator
    >>> evaluator = FactorEvaluator(
    ...     factor_data=factor_data,
    ...     ohlcv_data=ohlcv_data,
    ...     position_data=position_data
    ... )
    >>> results = evaluator.run_full_evaluation()
    >>> evaluator.print_report()
    >>> figures = evaluator.plot(output_dir='results/')
"""

from .metrics import (
    # 收益类
    total_return,
    annualized_return,
    cumulative_returns,
    ReturnsMetricsCalculator,
    calculate_portfolio_returns,
    
    # 风险类
    max_drawdown,
    max_drawdown_series,
    annualized_volatility,
    downside_volatility,
    RiskMetricsCalculator,
    
    # 风险调整类
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    information_ratio,
    omega_ratio,
    RiskAdjustedMetricsCalculator,
    
    # 换手率类
    turnover_rate,
    average_turnover,
    annualized_turnover,
    TurnoverMetricsCalculator,
    calculate_transaction_costs,
    
    # IC类
    calculate_ic,
    calculate_ic_series,
    calculate_rank_ic,
    calculate_icir,
    calculate_rank_icir,
    calculate_ic_statistics,
    calculate_quantile_returns,
    calculate_forward_returns,
    ICMetricsCalculator,
)

from .visualization import (
    plot_cumulative_ic,
    plot_ic_distribution,
    plot_cumulative_returns,
    plot_drawdown,
    plot_returns_and_drawdown,
    plot_factor_distribution,
    plot_position_returns_distribution,
    plot_quantile_returns,
    plot_rolling_ic,
    create_evaluation_report_figure,
)

from .evaluator import FactorEvaluator, BatchFactorEvaluator

# 从storage模块导入存储类
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from storage import ResultStorage, StorageConfig

__all__ = [
    # 指标计算类
    'ReturnsMetricsCalculator',
    'RiskMetricsCalculator',
    'RiskAdjustedMetricsCalculator',
    'TurnoverMetricsCalculator',
    'ICMetricsCalculator',
    
    # 收益类函数
    'total_return',
    'annualized_return',
    'cumulative_returns',
    'calculate_portfolio_returns',
    
    # 风险类函数
    'max_drawdown',
    'max_drawdown_series',
    'annualized_volatility',
    'downside_volatility',
    
    # 风险调整类函数
    'sharpe_ratio',
    'calmar_ratio',
    'sortino_ratio',
    'information_ratio',
    'omega_ratio',
    
    # 换手率类函数
    'turnover_rate',
    'average_turnover',
    'annualized_turnover',
    'calculate_transaction_costs',
    
    # IC类函数
    'calculate_ic',
    'calculate_ic_series',
    'calculate_rank_ic',
    'calculate_icir',
    'calculate_rank_icir',
    'calculate_ic_statistics',
    'calculate_quantile_returns',
    'calculate_forward_returns',
    
    # 可视化函数
    'plot_cumulative_ic',
    'plot_ic_distribution',
    'plot_cumulative_returns',
    'plot_drawdown',
    'plot_returns_and_drawdown',
    'plot_factor_distribution',
    'plot_position_returns_distribution',
    'plot_quantile_returns',
    'plot_rolling_ic',
    'create_evaluation_report_figure',
    
    # 评估器类
    'FactorEvaluator',
    'BatchFactorEvaluator',
    
    # 存储类
    'ResultStorage',
    'StorageConfig',
]
