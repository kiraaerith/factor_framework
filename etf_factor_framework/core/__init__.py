"""
ETF因子评估系统 - 核心模块

本模块定义了因子评估系统的基础数据结构和抽象接口，包括：
- FactorData: 因子数据容器 (N × T)
- PositionData: 仓位数据容器 (N × T)
- OHLCVData: OHLCV数据容器 (N × T × 5)
- 抽象基类: FactorCalculator, PositionMapper, Evaluator
"""

from .factor_data import FactorData
from .position_data import PositionData
from .ohlcv_data import OHLCVData
from .base_interfaces import FactorCalculator, PositionMapper, Evaluator

try:
    from .trade_context import TradeContext
    _TRADE_CONTEXT_AVAILABLE = True
except ImportError:
    _TRADE_CONTEXT_AVAILABLE = False

__all__ = [
    'FactorData',
    'PositionData',
    'OHLCVData',
    'FactorCalculator',
    'PositionMapper',
    'Evaluator',
    'TradeContext',
]
