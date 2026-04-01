"""
技术价格行为因子模块（fundamental 框架兼容版）

与 factors/technical_factors.py 中的 ETF 横截面因子不同，
此模块中的因子实现了 FundamentalFactorCalculator 接口，
可通过 run_factor_grid_v3.py 在全 A 股截面上评估。

数据来源：tushare.db 日频 OHLCV（后复权），与回测框架主数据源保持一致。
"""

from .max_drawdown_factor import MaxDrawdownFactor

__all__ = ["MaxDrawdownFactor"]
