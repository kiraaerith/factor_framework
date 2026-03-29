"""
基础接口定义模块

定义抽象基类：
- FactorCalculator: 因子计算器接口
- PositionMapper: 仓位映射器接口
- Evaluator: 评估器接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

try:
    from .factor_data import FactorData
    from .position_data import PositionData
    from .ohlcv_data import OHLCVData
except ImportError:
    from factor_data import FactorData
    from position_data import PositionData
    from ohlcv_data import OHLCVData


class FactorCalculator(ABC):
    """
    因子计算器抽象基类
    
    所有因子计算器的基类，定义了因子计算的标准接口。
    
    Example:
        >>> class MyFactorCalculator(FactorCalculator):
        ...     def __init__(self, period=20):
        ...         self.period = period
        ...     
        ...     def calculate(self, ohlcv_data):
        ...         close = ohlcv_data.close
        ...         ma = close.rolling(self.period, axis=1).mean()
        ...         factor_value = close / ma
        ...         return FactorData(factor_value, name='CloseOverMA', factor_type='CloseOverMA', params={'period': self.period})
        ...     
        ...     @property
        ...     def name(self):
        ...         return f"CloseOverMA_{self.period}"
        ...     
        ...     @property
        ...     def factor_type(self):
        ...         return "CloseOverMA"
        ...     
        ...     def get_params(self):
        ...         return {'period': self.period}
    """
    
    @abstractmethod
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算因子值
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 计算得到的因子数据
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取因子名称
        
        Returns:
            str: 因子名称（包含参数的具体实例名称，如 RSI_14）
        """
        pass
    
    @property
    @abstractmethod
    def factor_type(self) -> str:
        """
        获取因子类型
        
        Returns:
            str: 因子类型（不包含参数的通用类型名，如 RSI）
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        获取因子参数
        
        Returns:
            Dict: 因子参数字典
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.factor_type}', params={self.get_params()})"
    
    def __call__(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        使实例可调用，调用 calculate 方法
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 计算得到的因子数据
        """
        return self.calculate(ohlcv_data)


class PositionMapper(ABC):
    """
    仓位映射器抽象基类
    
    将因子值转换为仓位权重的抽象接口。
    
    Example:
        >>> class TopKMapper(PositionMapper):
        ...     def __init__(self, k=5):
        ...         self.k = k
        ...     
        ...     def map_to_position(self, factor_data):
        ...         values = factor_data.values
        ...         # 选排名前k的标的
        ...         ranks = values.rank(axis=0, ascending=False)
        ...         selected = (ranks <= self.k).astype(float)
        ...         # 等权分配
        ...         weights = selected.div(selected.sum(axis=0), axis=1).fillna(0)
        ...         return PositionData(weights, name=f'Top{self.k}', params={'k': self.k})
        ...     
        ...     @property
        ...     def name(self):
        ...         return f"Top{self.k}"
    """
    
    @abstractmethod
    def map_to_position(self, factor_data: FactorData) -> PositionData:
        """
        将因子值映射为仓位权重
        
        Args:
            factor_data: 因子数据容器
            
        Returns:
            PositionData: 仓位权重数据
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取映射器名称
        
        Returns:
            str: 映射器名称
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __call__(self, factor_data: FactorData) -> PositionData:
        """
        使实例可调用，调用 map_to_position 方法
        
        Args:
            factor_data: 因子数据容器
            
        Returns:
            PositionData: 仓位权重数据
        """
        return self.map_to_position(factor_data)


class Evaluator(ABC):
    """
    评估器抽象基类
    
    评估因子或策略绩效的抽象接口。
    
    Example:
        >>> class ICEvaluator(Evaluator):
        ...     def __init__(self, forward_period=1):
        ...         self.forward_period = forward_period
        ...     
        ...     def evaluate(self, factor_data, ohlcv_data, position_data=None):
        ...         # 计算IC
        ...         close = ohlcv_data.close
        ...         future_returns = close.shift(-self.forward_period, axis=1) / close - 1
        ...         
        ...         ic_series = []
        ...         for date in factor_data.dates:
        ...             if date in future_returns.columns:
        ...                 f = factor_data.get_cross_section(date)
        ...                 r = future_returns[date]
        ...                 ic = f.corr(r)
        ...                 ic_series.append(ic)
        ...         
        ...         mean_ic = np.mean(ic_series)
        ...         return {'IC': mean_ic, 'ICIR': mean_ic / np.std(ic_series)}
        ...     
        ...     @property
        ...     def metrics(self):
        ...         return ['IC', 'ICIR']
    """
    
    @abstractmethod
    def evaluate(
        self,
        factor_data: FactorData,
        ohlcv_data: OHLCVData,
        position_data: Optional[PositionData] = None
    ) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            factor_data: 因子数据容器
            ohlcv_data: OHLCV数据容器
            position_data: 仓位数据容器，可选
            
        Returns:
            Dict: 评估指标字典
        """
        pass
    
    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        """
        获取该评估器支持的指标列表
        
        Returns:
            List[str]: 指标名称列表
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取评估器名称
        
        Returns:
            str: 评估器名称
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', metrics={self.metrics})"
    
    def __call__(
        self,
        factor_data: FactorData,
        ohlcv_data: OHLCVData,
        position_data: Optional[PositionData] = None
    ) -> Dict[str, float]:
        """
        使实例可调用，调用 evaluate 方法
        
        Args:
            factor_data: 因子数据容器
            ohlcv_data: OHLCV数据容器
            position_data: 仓位数据容器，可选
            
        Returns:
            Dict: 评估指标字典
        """
        return self.evaluate(factor_data, ohlcv_data, position_data)


class CompositeCalculator(FactorCalculator):
    """
    组合因子计算器
    
    将多个因子计算器组合，支持因子加权合成。
    """
    
    def __init__(self, calculators: List[FactorCalculator], weights: Optional[List[float]] = None):
        """
        初始化组合计算器
        
        Args:
            calculators: 因子计算器列表
            weights: 权重列表，None表示等权
        """
        self.calculators = calculators
        self.weights = weights or [1.0 / len(calculators)] * len(calculators)
        
        if len(self.calculators) != len(self.weights):
            raise ValueError("calculators and weights must have the same length")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算组合因子值
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 加权组合后的因子数据
        """
        factors = [calc.calculate(ohlcv_data) for calc in self.calculators]
        
        # 对每个因子进行Z-Score标准化后加权
        normalized_factors = [f.zscore() for f in factors]
        
        # 加权求和
        combined_values = sum(
            w * nf.values for w, nf in zip(self.weights, normalized_factors)
        )
        
        return FactorData(
            combined_values,
            name=self.name,
            factor_type=self.factor_type,
            params=self.get_params()
        )
    
    @property
    def name(self) -> str:
        """获取组合因子名称"""
        calculator_names = [calc.name for calc in self.calculators]
        return f"Composite({'+'.join(calculator_names)})"
    
    @property
    def factor_type(self) -> str:
        """获取组合因子类型"""
        calculator_types = [calc.factor_type for calc in self.calculators]
        return f"Composite({'+'.join(calculator_types)})"
    
    def get_params(self) -> Dict[str, Any]:
        """获取组合因子参数"""
        return {
            'calculators': [calc.name for calc in self.calculators],
            'weights': self.weights
        }


class PipelineEvaluator(Evaluator):
    """
    管道评估器
    
    将多个评估器组合，依次执行评估。
    """
    
    def __init__(self, evaluators: List[Evaluator]):
        """
        初始化管道评估器
        
        Args:
            evaluators: 评估器列表
        """
        self.evaluators = evaluators
    
    def evaluate(
        self,
        factor_data: FactorData,
        ohlcv_data: OHLCVData,
        position_data: Optional[PositionData] = None
    ) -> Dict[str, float]:
        """
        执行所有评估器的评估
        
        Args:
            factor_data: 因子数据容器
            ohlcv_data: OHLCV数据容器
            position_data: 仓位数据容器，可选
            
        Returns:
            Dict: 合并后的评估指标字典
        """
        results = {}
        for evaluator in self.evaluators:
            metrics = evaluator.evaluate(factor_data, ohlcv_data, position_data)
            # 添加前缀避免冲突
            prefix = f"{evaluator.name}_"
            results.update({f"{prefix}{k}": v for k, v in metrics.items()})
        return results
    
    @property
    def metrics(self) -> List[str]:
        """获取所有支持的指标"""
        all_metrics = []
        for evaluator in self.evaluators:
            all_metrics.extend([f"{evaluator.name}_{m}" for m in evaluator.metrics])
        return all_metrics
    
    @property
    def name(self) -> str:
        """获取评估器名称"""
        return f"Pipeline({'+'.join(e.name for e in self.evaluators)})"
