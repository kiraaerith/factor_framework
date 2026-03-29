"""
评估指标模块

提供各类评估指标的计算功能。

分类：
- 收益类指标：总收益、年化收益
- 风险类指标：最大回撤、年化波动率
- 风险调整类指标：夏普比率、Calmar比率、Sortino比率
- 换手率类指标：日均换手率
- IC类指标：IC、Rank IC、ICIR、Rank ICIR
"""

# 收益类指标
from .returns_metrics import (
    total_return,
    annualized_return,
    cumulative_returns,
    average_return,
    calculate_portfolio_returns,
    ReturnsMetricsCalculator,
)

# 风险类指标
from .risk_metrics import (
    max_drawdown,
    max_drawdown_series,
    drawdown_duration,
    annualized_volatility,
    downside_volatility,
    value_at_risk,
    conditional_var,
    RiskMetricsCalculator,
)

# 风险调整类指标
from .risk_adjusted_metrics import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    information_ratio,
    treynor_ratio,
    omega_ratio,
    sterling_ratio,
    RiskAdjustedMetricsCalculator,
)

# 换手率类指标
from .turnover_metrics import (
    turnover_rate,
    average_turnover,
    annualized_turnover,
    turnover_volatility,
    position_change_count,
    active_position_ratio,
    calculate_transaction_costs,
    turnover_attribution,
    TurnoverMetricsCalculator,
)

# IC类指标
from .ic_metrics import (
    calculate_ic,
    calculate_ic_series,
    calculate_rank_ic,
    calculate_icir,
    calculate_rank_icir,
    calculate_ic_statistics,
    calculate_ic_decay,
    calculate_quantile_returns,
    calculate_forward_returns,
    ICMetricsCalculator,
)

__all__ = [
    # 收益类
    'total_return',
    'annualized_return',
    'cumulative_returns',
    'average_return',
    'calculate_portfolio_returns',
    'ReturnsMetricsCalculator',
    
    # 风险类
    'max_drawdown',
    'max_drawdown_series',
    'drawdown_duration',
    'annualized_volatility',
    'downside_volatility',
    'value_at_risk',
    'conditional_var',
    'RiskMetricsCalculator',
    
    # 风险调整类
    'sharpe_ratio',
    'calmar_ratio',
    'sortino_ratio',
    'information_ratio',
    'treynor_ratio',
    'omega_ratio',
    'sterling_ratio',
    'RiskAdjustedMetricsCalculator',
    
    # 换手率类
    'turnover_rate',
    'average_turnover',
    'annualized_turnover',
    'turnover_volatility',
    'position_change_count',
    'active_position_ratio',
    'calculate_transaction_costs',
    'turnover_attribution',
    'TurnoverMetricsCalculator',
    
    # IC类
    'calculate_ic',
    'calculate_ic_series',
    'calculate_rank_ic',
    'calculate_icir',
    'calculate_rank_icir',
    'calculate_ic_statistics',
    'calculate_ic_decay',
    'calculate_quantile_returns',
    'calculate_forward_returns',
    'ICMetricsCalculator',
]
