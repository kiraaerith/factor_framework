"""
全时段量价相关性因子 - CTC量价因子改造

基于CTC Institute文章中"3.1 全时段量价相关性"的日频改造版本

原始逻辑：使用整个小时60根K线，计算PV、dPV、PdV、dPdV四类量价相关性，考虑0-2阶领先滞后关系。
日频改造：使用过去N个交易日计算上述四类相关系数

因子列表：
- pv_corr: 收盘价与成交量的相关系数
- dpv_corr: 日收益率与成交量的相关系数  
- pdv_corr: 收盘价与成交量变化的相关系数
- dpdv_corr: 日收益率与成交量变化相关系数
- 各特征的lag1, lag2, lead1, lead2版本
"""

from typing import Dict, Any, Optional
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


class PriceVolumeCorrelationBase(OHLCVFactorCalculator):
    """
    量价相关性因子基类
    
    参数:
        window: 滚动窗口天数，默认20
        correlation_type: 相关性类型 ('pv', 'dpv', 'pdv', 'dpdv')
        lag: 滞后阶数，0表示当期，负数表示领先，正数表示滞后
    """
    
    CORRELATION_TYPES = {
        'pv': '收盘价与成交量相关系数',
        'dpv': '收益率与成交量相关系数',
        'pdv': '收盘价与成交量变化相关系数',
        'dpdv': '收益率与成交量变化相关系数',
    }
    
    def __init__(self, window: int = 20, correlation_type: str = 'pv', lag: int = 0):
        self._window = window
        self._correlation_type = correlation_type
        self._lag = lag
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if self._correlation_type not in self.CORRELATION_TYPES:
            raise ValueError(f"correlation_type must be one of {list(self.CORRELATION_TYPES.keys())}, got {self._correlation_type}")
        if not isinstance(self._lag, int):
            raise ValueError(f"lag must be integer, got {self._lag}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """计算量价相关性因子"""
        self._validate_params()
        
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        # 根据correlation_type准备数据
        if self._correlation_type == 'pv':
            # PV: 收盘价与成交量
            x = close
            y = volume
        elif self._correlation_type == 'dpv':
            # dPV: 收益率与成交量
            x = close.pct_change()
            y = volume
        elif self._correlation_type == 'pdv':
            # PdV: 收盘价与成交量变化
            x = close
            y = volume.diff()
        elif self._correlation_type == 'dpdv':
            # dPdV: 收益率与成交量变化
            x = close.pct_change()
            y = volume.diff()
        
        # 应用滞后/领先
        if self._lag > 0:
            # 滞后：y向后移动（用过去的y与当前的x相关）
            y = y.shift(self._lag)
        elif self._lag < 0:
            # 领先：y向前移动（用未来的y与当前的x相关）
            y = y.shift(self._lag)
        
        # 滚动计算相关系数
        factor_values = self._rolling_correlation(x, y, self._window)
        
        # 验证输出
        self._validate_output(factor_values, ohlcv_data)
        
        return FactorData(
            values=factor_values,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_correlation(self, x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动计算相关系数
        
        对每个symbol分别计算rolling correlation
        """
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        
        for symbol in x.index:
            x_series = x.loc[symbol]
            y_series = y.loc[symbol]
            
            # 使用pandas的rolling corr - 需要在Series上调用
            corr_series = x_series.rolling(window=window, min_periods=window//2).corr(y_series)
            result.loc[symbol] = corr_series.values
        
        return result
    
    @property
    def name(self) -> str:
        lag_str = f"_lag{self._lag}" if self._lag > 0 else f"_lead{-self._lag}" if self._lag < 0 else ""
        return f"PriceVolumeCorrelation_{self._correlation_type}_w{self._window}{lag_str}"
    
    @property
    def factor_type(self) -> str:
        return "PriceVolumeCorrelation"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'correlation_type': self._correlation_type,
            'lag': self._lag
        }


# 便捷因子类定义
class PVCorr(PriceVolumeCorrelationBase):
    """收盘价与成交量相关系数 (PV)"""
    def __init__(self, window: int = 20, lag: int = 0):
        super().__init__(window, 'pv', lag)


class DPVCorr(PriceVolumeCorrelationBase):
    """收益率与成交量相关系数 (dPV)"""
    def __init__(self, window: int = 20, lag: int = 0):
        super().__init__(window, 'dpv', lag)


class PdVCorr(PriceVolumeCorrelationBase):
    """收盘价与成交量变化相关系数 (PdV)"""
    def __init__(self, window: int = 20, lag: int = 0):
        super().__init__(window, 'pdv', lag)


class DPdVCorr(PriceVolumeCorrelationBase):
    """收益率与成交量变化相关系数 (dPdV)"""
    def __init__(self, window: int = 20, lag: int = 0):
        super().__init__(window, 'dpdv', lag)


# 别名定义（方便使用）
PriceVolumeCorrelation = PVCorr
