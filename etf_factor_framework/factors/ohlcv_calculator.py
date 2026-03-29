"""
OHLCV因子计算器基类模块

定义基于OHLCV数据的因子计算器基类，提供通用的计算框架和工具方法。
"""

from typing import Optional, Dict, Any
from abc import abstractmethod
import numpy as np
import pandas as pd

try:
    from ..core.base_interfaces import FactorCalculator
    from ..core.ohlcv_data import OHLCVData
    from ..core.factor_data import FactorData
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from core.base_interfaces import FactorCalculator
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData


class OHLCVFactorCalculator(FactorCalculator):
    """
    OHLCV因子计算器基类
    
    所有基于OHLCV数据的因子计算器的基类，继承自 FactorCalculator。
    提供了一些通用的工具方法，如计算移动平均、标准差等。
    
    Attributes:
        name: 因子名称
        params: 因子参数字典
        
    Example:
        >>> class MyFactor(OHLCVFactorCalculator):
        ...     def __init__(self, period=20):
        ...         self._period = period
        ...     
        ...     def calculate(self, ohlcv_data):
        ...         close = ohlcv_data.close
        ...         ma = self._rolling_mean(close, self._period)
        ...         factor_value = close / ma
        ...         return FactorData(factor_value, name=self.name, params=self.params)
        ...     
        ...     @property
        ...     def name(self):
        ...         return f"MyFactor_{self._period}"
        ...     
        ...     @property
        ...     def params(self):
        ...         return {'period': self._period}
    """
    
    def __init__(self):
        """初始化OHLCV因子计算器"""
        self._validate_params()
    
    @abstractmethod
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算因子值
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 计算得到的因子数据 (N × T)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取因子名称
        
        Returns:
            str: 因子名称（包含参数的具体实例名称）
        """
        pass
    
    @property
    @abstractmethod
    def factor_type(self) -> str:
        """
        获取因子类型
        
        Returns:
            str: 因子类型（不包含参数的通用类型名）
        """
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """
        获取因子参数
        
        Returns:
            Dict: 因子参数字典
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取因子参数（兼容基类接口）
        
        Returns:
            Dict: 因子参数字典
        """
        return self.params
    
    def _validate_params(self):
        """
        验证参数有效性
        
        Raises:
            ValueError: 如果参数无效
        """
        pass
    
    # ==================== 通用工具方法 ====================
    
    def _rolling_mean(self, data: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动均值
        
        Args:
            data: 输入数据 (N × T)
            window: 滚动窗口大小
            min_periods: 最小周期数，默认为window
            
        Returns:
            DataFrame: 滚动均值 (N × T)
        """
        min_periods = min_periods or window
        return data.rolling(window=window, axis=1, min_periods=min_periods).mean()
    
    def _rolling_std(self, data: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动标准差
        
        Args:
            data: 输入数据 (N × T)
            window: 滚动窗口大小
            min_periods: 最小周期数，默认为window
            
        Returns:
            DataFrame: 滚动标准差 (N × T)
        """
        min_periods = min_periods or window
        return data.rolling(window=window, axis=1, min_periods=min_periods).std()
    
    def _rolling_sum(self, data: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动求和
        
        Args:
            data: 输入数据 (N × T)
            window: 滚动窗口大小
            min_periods: 最小周期数，默认为window
            
        Returns:
            DataFrame: 滚动求和 (N × T)
        """
        min_periods = min_periods or 1
        return data.rolling(window=window, axis=1, min_periods=min_periods).sum()
    
    def _rolling_max(self, data: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动最大值
        
        Args:
            data: 输入数据 (N × T)
            window: 滚动窗口大小
            min_periods: 最小周期数，默认为1
            
        Returns:
            DataFrame: 滚动最大值 (N × T)
        """
        min_periods = min_periods or 1
        return data.rolling(window=window, axis=1, min_periods=min_periods).max()
    
    def _rolling_min(self, data: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算滚动最小值
        
        Args:
            data: 输入数据 (N × T)
            window: 滚动窗口大小
            min_periods: 最小周期数，默认为1
            
        Returns:
            DataFrame: 滚动最小值 (N × T)
        """
        min_periods = min_periods or 1
        return data.rolling(window=window, axis=1, min_periods=min_periods).min()
    
    def _ema(self, data: pd.DataFrame, span: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        计算指数移动平均 (EMA)
        
        Args:
            data: 输入数据 (N × T)
            span: EMA跨度
            min_periods: 最小周期数，默认为1
            
        Returns:
            DataFrame: EMA值 (N × T)
        """
        min_periods = min_periods or 1
        return data.ewm(span=span, axis=1, min_periods=min_periods, adjust=False).mean()
    
    def _shift(self, data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        时间平移
        
        Args:
            data: 输入数据 (N × T)
            periods: 平移周期数，正数表示向后平移
            
        Returns:
            DataFrame: 平移后的数据 (N × T)
        """
        return data.shift(periods, axis=1)
    
    def _change(self, data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        计算变化量
        
        Args:
            data: 输入数据 (N × T)
            periods: 周期数
            
        Returns:
            DataFrame: 变化量 (N × T)
        """
        return data.diff(periods, axis=1)
    
    def _pct_change(self, data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        计算百分比变化
        
        Args:
            data: 输入数据 (N × T)
            periods: 周期数
            
        Returns:
            DataFrame: 百分比变化 (N × T)
        """
        return data.pct_change(periods, axis=1)
    
    def _log_change(self, data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        计算对数变化
        
        Args:
            data: 输入数据 (N × T)
            periods: 周期数
            
        Returns:
            DataFrame: 对数变化 (N × T)
        """
        return np.log(data / data.shift(periods, axis=1))
    
    def _cross_sectional_rank(self, data: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
        """
        计算横截面排名
        
        Args:
            data: 输入数据 (N × T)
            axis: 排名方向，0=按列(横截面)
            
        Returns:
            DataFrame: 排名结果 (N × T)
        """
        return data.rank(axis=axis)
    
    def _cross_sectional_zscore(self, data: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
        """
        计算横截面Z-Score标准化
        
        Args:
            data: 输入数据 (N × T)
            axis: 标准化方向，0=按列(横截面)
            
        Returns:
            DataFrame: Z-Score标准化结果 (N × T)
        """
        mean = data.mean(axis=axis)
        std = data.std(axis=axis)
        return data.sub(mean, axis=1-axis).div(std, axis=1-axis)
    
    def _validate_output(self, result: pd.DataFrame, ohlcv_data: OHLCVData) -> None:
        """
        验证输出形状是否与输入一致
        
        Args:
            result: 计算结果
            ohlcv_data: 输入的OHLCV数据
            
        Raises:
            ValueError: 如果输出形状不匹配
        """
        expected_shape = (ohlcv_data.n_assets, ohlcv_data.n_periods)
        if result.shape != expected_shape:
            raise ValueError(
                f"Output shape {result.shape} does not match expected shape {expected_shape}"
            )
        
        # 检查索引是否一致
        if not list(result.index) == ohlcv_data.symbols:
            raise ValueError("Output index does not match input symbols")
        
        if not list(result.columns) == ohlcv_data.dates:
            raise ValueError("Output columns do not match input dates")
