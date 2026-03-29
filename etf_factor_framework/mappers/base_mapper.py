"""
仓位映射器基类模块

定义仓位映射器的抽象基类和通用功能。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import sys
import os

try:
    # 相对导入（包内使用）
    from ..core.base_interfaces import PositionMapper
except ImportError:
    # 绝对导入（独立运行）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from core.base_interfaces import PositionMapper


class BasePositionMapper(PositionMapper, ABC):
    """
    仓位映射器基类
    
    所有仓位映射器的基类，继承自 PositionMapper 抽象接口。
    提供通用的参数管理和名称生成方法。
    
    Attributes:
        name: 映射器名称
        params: 映射器参数字典
        
    Example:
        >>> class MyMapper(BasePositionMapper):
        ...     def __init__(self, threshold=0.5):
        ...         self.threshold = threshold
        ...     
        ...     def map_to_position(self, factor_data):
        ...         # 实现映射逻辑
        ...         pass
        ...     
        ...     @property
        ...     def name(self):
        ...         return f"ThresholdMapper_{self.threshold}"
        ...     
        ...     def get_params(self):
        ...         return {'threshold': self.threshold}
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        初始化映射器
        
        Args:
            name: 映射器名称，默认使用类名
            **kwargs: 其他参数
        """
        self._custom_name = name
        self._params = kwargs
    
    @property
    def name(self) -> str:
        """
        获取映射器名称
        
        Returns:
            str: 映射器名称
        """
        if self._custom_name:
            return self._custom_name
        return self.__class__.__name__
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取映射器参数
        
        Returns:
            Dict: 参数字典
        """
        return self._params.copy()
    
    @abstractmethod
    def map_to_position(self, factor_data) -> 'PositionData':
        """
        将因子值映射为仓位权重
        
        Args:
            factor_data: 因子数据容器 (FactorData)
            
        Returns:
            PositionData: 仓位权重数据
        """
        pass
