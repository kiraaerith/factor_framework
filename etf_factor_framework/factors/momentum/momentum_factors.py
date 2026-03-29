"""
动量因子实现模块

基于收益率的动量因子，支持延迟和回看周期参数。
公式：momentum = close(t-X) / close(t-X-Y) - 1
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

try:
    from ..ohlcv_calculator import OHLCVFactorCalculator
    from ...core.ohlcv_data import OHLCVData
    from ...core.factor_data import FactorData
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from factors.ohlcv_calculator import OHLCVFactorCalculator
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData


class MomentumFactor(OHLCVFactorCalculator):
    """
    动量因子
    
    原理：计算从X周期前开始，回溯Y个周期的收益率
    
    公式：momentum = close(t-X) / close(t-X-Y) - 1
    
    Attributes:
        offset: 延迟周期X（从当前往过去数X个周期开始）
        lookback: 回看周期Y（往回看Y个周期的收益率）
    """
    
    def __init__(self, offset: int = 0, lookback: int = 20):
        self._offset = offset
        self._lookback = lookback
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._offset, int) or self._offset < 0:
            raise ValueError(f"offset must be non-negative integer, got {self._offset}")
        if not isinstance(self._lookback, int) or self._lookback <= 0:
            raise ValueError(f"lookback must be positive integer, got {self._lookback}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        # 获取收盘价
        close = ohlcv_data.close
        
        # 计算动量：close(t-X) / close(t-X-Y) - 1
        # 注意：shift(1)表示向前推1期（取前一天数据）
        price_recent = close.shift(self._offset, axis=1)
        price_past = close.shift(self._offset + self._lookback, axis=1)
        
        # 计算收益率
        momentum = price_recent / price_past - 1
        
        # 验证输出
        self._validate_output(momentum, ohlcv_data)
        
        return FactorData(
            values=momentum,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        return f"Momentum_o{self._offset}_l{self._lookback}"
    
    @property
    def factor_type(self) -> str:
        return "MomentumFactor"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'offset': self._offset,
            'lookback': self._lookback
        }
