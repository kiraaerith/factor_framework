"""
技术因子实现模块

提供基于OHLCV数据的技术分析因子实现，包括：
- CloseOverMA: 收盘价与均线的比率
- RSI: 相对强弱指标
- Momentum: 动量因子
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

try:
    from .ohlcv_calculator import OHLCVFactorCalculator
    from ..core.ohlcv_data import OHLCVData
    from ..core.factor_data import FactorData
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from factors.ohlcv_calculator import OHLCVFactorCalculator
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData


class CloseOverMA(OHLCVFactorCalculator):
    """
    收盘价与均线比率因子
    
    计算公式：Close / MA(Close, period)
    
    当比值 > 1 时，表示价格高于均线，可能为上涨趋势；
    当比值 < 1 时，表示价格低于均线，可能为下跌趋势。
    
    Attributes:
        period: 均线周期
        field: 使用的字段，默认'close'
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=30)
        >>> close = pd.DataFrame(np.random.randn(2, 30).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 30)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算因子
        >>> factor = CloseOverMA(period=20)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.shape)  # (2, 30)
    """
    
    def __init__(self, period: int = 20, field: str = 'close'):
        """
        初始化CloseOverMA因子计算器
        
        Args:
            period: 均线周期，必须大于0
            field: 使用的字段，可选 'open', 'high', 'low', 'close', 'volume'
            
        Raises:
            ValueError: 如果参数无效
        """
        self._period = period
        self._field = field
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._period, int) or self._period <= 0:
            raise ValueError(f"period must be a positive integer, got {self._period}")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算收盘价与均线比率因子
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 计算移动平均
        ma = self._rolling_mean(price, self._period, min_periods=1)
        
        # 计算比率
        ratio = price / ma
        
        # 验证输出
        self._validate_output(ratio, ohlcv_data)
        
        return FactorData(
            values=ratio,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        """获取因子名称"""
        return f"CloseOverMA_{self._period}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "CloseOverMA"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'period': self._period,
            'field': self._field
        }


class RSI(OHLCVFactorCalculator):
    """
    相对强弱指标 (Relative Strength Index)
    
    RSI是一种动量指标，用于衡量价格变动的速度和幅度。
    计算公式：RSI = 100 - 100 / (1 + RS)
    其中 RS = 平均上涨幅度 / 平均下跌幅度
    
    RSI取值范围0-100：
    - RSI > 70: 超买状态
    - RSI < 30: 超卖状态
    - RSI = 50: 多空均衡
    
    Attributes:
        period: RSI计算周期
        field: 使用的字段，默认'close'
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=30)
        >>> close = pd.DataFrame(np.random.randn(2, 30).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 30)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算因子
        >>> factor = RSI(period=14)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.values.min(), result.values.max())  # 应该在0-100范围内
    """
    
    def __init__(self, period: int = 14, field: str = 'close'):
        """
        初始化RSI因子计算器
        
        Args:
            period: RSI计算周期，通常使用14
            field: 使用的字段，可选 'open', 'high', 'low', 'close', 'volume'
            
        Raises:
            ValueError: 如果参数无效
        """
        self._period = period
        self._field = field
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._period, int) or self._period <= 0:
            raise ValueError(f"period must be a positive integer, got {self._period}")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算RSI因子
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)，取值范围0-100
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 计算价格变化
        delta = price.diff(axis=1)
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 计算平均上涨和下跌（使用指数移动平均）
        avg_gain = gain.ewm(span=self._period, axis=1, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(span=self._period, axis=1, min_periods=1, adjust=False).mean()
        
        # 计算RS
        rs = avg_gain / avg_loss.replace(0, np.nan)
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 填充NaN值
        rsi = rsi.fillna(50)  # 没有涨跌时默认为50
        
        # 验证输出
        self._validate_output(rsi, ohlcv_data)
        
        return FactorData(
            values=rsi,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        """获取因子名称"""
        return f"RSI_{self._period}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "RSI"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'period': self._period,
            'field': self._field
        }


class Momentum(OHLCVFactorCalculator):
    """
    动量因子
    
    动量因子衡量价格在一定时期内的变化率。
    计算公式：(Close_t - Close_{t-n}) / Close_{t-n} 或 Close_t / Close_{t-n}
    
    动量因子是技术分析中最基础也是最重要的因子之一，
    反映了价格的惯性趋势。
    
    Attributes:
        period: 动量计算周期
        field: 使用的字段，默认'close'
        log_return: 是否使用对数收益率，默认False
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=30)
        >>> close = pd.DataFrame(np.random.randn(2, 30).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 30)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算因子
        >>> factor = Momentum(period=20)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.shape)  # (2, 30)
    """
    
    def __init__(self, period: int = 20, field: str = 'close', log_return: bool = False):
        """
        初始化Momentum因子计算器
        
        Args:
            period: 动量计算周期，必须大于0
            field: 使用的字段，可选 'open', 'high', 'low', 'close', 'volume'
            log_return: 是否使用对数收益率，False使用简单收益率
            
        Raises:
            ValueError: 如果参数无效
        """
        self._period = period
        self._field = field
        self._log_return = log_return
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._period, int) or self._period <= 0:
            raise ValueError(f"period must be a positive integer, got {self._period}")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算动量因子
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 计算动量
        if self._log_return:
            # 对数收益率: log(Close_t / Close_{t-n})
            momentum = self._log_change(price, self._period)
        else:
            # 简单收益率: (Close_t - Close_{t-n}) / Close_{t-n}
            momentum = self._pct_change(price, self._period)
        
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
        """获取因子名称"""
        return f"Momentum_{self._period}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "Momentum"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'period': self._period,
            'field': self._field,
            'log_return': self._log_return
        }


class MACD(OHLCVFactorCalculator):
    """
    MACD (Moving Average Convergence Divergence) 因子
    
    MACD是趋势跟踪动量指标，显示两条移动平均线之间的关系。
    计算公式：
    - DIF = EMA(Close, fast) - EMA(Close, slow)
    - DEA = EMA(DIF, signal)
    - MACD = (DIF - DEA) * 2
    
    Attributes:
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26
        signal_period: 信号线周期，默认9
        field: 使用的字段，默认'close'
        use_histogram: 是否使用MACD柱状图作为因子值，默认True
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=50)
        >>> close = pd.DataFrame(np.random.randn(2, 50).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 50)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算因子
        >>> factor = MACD(fast_period=12, slow_period=26, signal_period=9)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.shape)  # (2, 50)
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        field: str = 'close',
        use_histogram: bool = True
    ):
        """
        初始化MACD因子计算器
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            field: 使用的字段
            use_histogram: 是否使用MACD柱状图作为因子值
            
        Raises:
            ValueError: 如果参数无效
        """
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period
        self._field = field
        self._use_histogram = use_histogram
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._fast_period, int) or self._fast_period <= 0:
            raise ValueError(f"fast_period must be a positive integer, got {self._fast_period}")
        if not isinstance(self._slow_period, int) or self._slow_period <= 0:
            raise ValueError(f"slow_period must be a positive integer, got {self._slow_period}")
        if not isinstance(self._signal_period, int) or self._signal_period <= 0:
            raise ValueError(f"signal_period must be a positive integer, got {self._signal_period}")
        if self._fast_period >= self._slow_period:
            raise ValueError(f"fast_period ({self._fast_period}) must be less than slow_period ({self._slow_period})")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算MACD因子
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 计算EMA
        ema_fast = self._ema(price, self._fast_period)
        ema_slow = self._ema(price, self._slow_period)
        
        # 计算DIF（快线减去慢线）
        dif = ema_fast - ema_slow
        
        # 计算DEA（DIF的EMA）
        dea = self._ema(dif, self._signal_period)
        
        # 计算MACD柱状图
        macd_histogram = (dif - dea) * 2
        
        # 选择输出
        if self._use_histogram:
            result = macd_histogram
        else:
            result = dif
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        """获取因子名称"""
        return f"MACD_{self._fast_period}_{self._slow_period}_{self._signal_period}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "MACD"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'fast_period': self._fast_period,
            'slow_period': self._slow_period,
            'signal_period': self._signal_period,
            'field': self._field,
            'use_histogram': self._use_histogram
        }


class BollingerBands(OHLCVFactorCalculator):
    """
    布林带 (Bollinger Bands) 位置因子
    
    衡量价格相对于布林带的位置。
    计算公式：(Close - LowerBand) / (UpperBand - LowerBand)
    
    结果解释：
    - 接近0：价格接近下轨，可能超卖
    - 接近0.5：价格在中轨附近
    - 接近1：价格接近上轨，可能超买
    
    Attributes:
        period: 计算周期，默认20
        std_multiplier: 标准差乘数，默认2
        field: 使用的字段，默认'close'
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=30)
        >>> close = pd.DataFrame(np.random.randn(2, 30).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 30)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算因子
        >>> factor = BollingerBands(period=20, std_multiplier=2)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.values.min(), result.values.max())  # 应该在0-1范围内
    """
    
    def __init__(self, period: int = 20, std_multiplier: float = 2.0, field: str = 'close'):
        """
        初始化布林带位置因子计算器
        
        Args:
            period: 计算周期
            std_multiplier: 标准差乘数
            field: 使用的字段
            
        Raises:
            ValueError: 如果参数无效
        """
        self._period = period
        self._std_multiplier = std_multiplier
        self._field = field
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._period, int) or self._period <= 0:
            raise ValueError(f"period must be a positive integer, got {self._period}")
        if self._std_multiplier <= 0:
            raise ValueError(f"std_multiplier must be positive, got {self._std_multiplier}")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算布林带位置因子
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)，取值范围0-1
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 计算中轨（移动平均线）
        middle_band = self._rolling_mean(price, self._period)
        
        # 计算标准差
        std = self._rolling_std(price, self._period)
        
        # 计算上轨和下轨
        upper_band = middle_band + self._std_multiplier * std
        lower_band = middle_band - self._std_multiplier * std
        
        # 计算价格在布林带中的位置
        band_width = upper_band - lower_band
        
        # 处理带宽为0的情况（价格没有波动）
        position = pd.DataFrame(
            np.where(
                band_width > 0,
                (price - lower_band) / band_width,
                0.5  # 如果带宽为0，默认位置在中间
            ),
            index=price.index,
            columns=price.columns
        )
        
        # 限制范围在0-1之间
        position = position.clip(lower=0, upper=1)
        
        # 验证输出
        self._validate_output(position, ohlcv_data)
        
        return FactorData(
            values=position,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        """获取因子名称"""
        return f"BollingerBands_{self._period}_{self._std_multiplier}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "BollingerBands"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'period': self._period,
            'std_multiplier': self._std_multiplier,
            'field': self._field
        }


class FutureReturn(OHLCVFactorCalculator):
    """
    未来收益率因子 (前瞻因子 / Baseline因子)
    
    ⚠️ 警告：此因子使用了未来信息（Future Information / Lookahead Bias），
    仅用于作为理论极限情况的baseline参考，不应用于实际交易！
    
    计算公式：
    - 简单收益率: (Close_{t+n} - Close_t) / Close_t
    - 对数收益率: log(Close_{t+n} / Close_t)
    
    该因子将下一周期（或未来N周期）的实际收益率作为当前时刻的因子值，
    代表了理论上"完美预测"的上限。在回测中，此因子可用于：
    1. 验证回测框架的正确性（应该获得接近完美的收益）
    2. 作为其他因子的性能上限参考（任何实际因子的IC都应低于此因子）
    3. 评估因子的信息损失程度
    
    Attributes:
        period: 前瞻周期数，默认1（下一周期）
        field: 使用的字段，默认'close'
        log_return: 是否使用对数收益率，默认False
        
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例数据
        >>> symbols = ['ETF1', 'ETF2']
        >>> dates = pd.date_range('2024-01-01', periods=30)
        >>> close = pd.DataFrame(np.random.randn(2, 30).cumsum() + 100, index=symbols, columns=dates)
        >>> ohlcv = OHLCVData(
        ...     open=close * 0.99,
        ...     high=close * 1.02,
        ...     low=close * 0.98,
        ...     close=close,
        ...     volume=pd.DataFrame(np.abs(np.random.randn(2, 30)) * 1000, index=symbols, columns=dates)
        ... )
        >>> 
        >>> # 计算未来1期收益率因子
        >>> factor = FutureReturn(period=1)
        >>> result = factor.calculate(ohlcv)
        >>> print(result.shape)  # (2, 30)
        >>> 
        >>> # 计算未来5期对数收益率因子
        >>> factor_log = FutureReturn(period=5, log_return=True)
        >>> result_log = factor_log.calculate(ohlcv)
    """
    
    def __init__(self, period: int = 1, field: str = 'close', log_return: bool = False):
        """
        初始化未来收益率因子计算器
        
        Args:
            period: 前瞻周期数，必须大于0。例如period=1表示使用下一周期的收益率
            field: 使用的字段，可选 'open', 'high', 'low', 'close', 'volume'
            log_return: 是否使用对数收益率，False使用简单收益率
            
        Raises:
            ValueError: 如果参数无效
        """
        self._period = period
        self._field = field
        self._log_return = log_return
        super().__init__()
    
    def _validate_params(self):
        """验证参数有效性"""
        if not isinstance(self._period, int) or self._period <= 0:
            raise ValueError(f"period must be a positive integer, got {self._period}")
        if self._field not in ['open', 'high', 'low', 'close', 'volume']:
            raise ValueError(f"field must be one of 'open', 'high', 'low', 'close', 'volume', got {self._field}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算未来收益率因子
        
        使用 shift(-period) 来获取未来的价格，计算收益率。
        注意：结果的最后period列会是NaN，因为未来数据不存在。
        
        Args:
            ohlcv_data: OHLCV数据容器
            
        Returns:
            FactorData: 因子数据 (N × T)，最后period列为NaN
        """
        # 获取价格数据
        price = ohlcv_data.get_field(self._field)
        
        # 获取未来价格（负向shift = 向前看）
        future_price = price.shift(-self._period, axis=1)
        
        # 计算未来收益率
        if self._log_return:
            # 对数收益率: log(Close_{t+n} / Close_t)
            future_return = np.log(future_price / price)
        else:
            # 简单收益率: (Close_{t+n} - Close_t) / Close_t
            future_return = (future_price - price) / price
        
        # 验证输出（允许NaN存在，因为最后几期确实没有未来数据）
        # 这里我们手动检查形状和索引，而不调用 _validate_output
        expected_shape = (ohlcv_data.n_assets, ohlcv_data.n_periods)
        if future_return.shape != expected_shape:
            raise ValueError(
                f"Output shape {future_return.shape} does not match expected shape {expected_shape}"
            )
        if not list(future_return.index) == ohlcv_data.symbols:
            raise ValueError("Output index does not match input symbols")
        if not list(future_return.columns) == ohlcv_data.dates:
            raise ValueError("Output columns do not match input dates")
        
        return FactorData(
            values=future_return,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        """获取因子名称"""
        return f"FutureReturn_{self._period}"
    
    @property
    def factor_type(self) -> str:
        """获取因子类型"""
        return "FutureReturn"
    
    @property
    def params(self) -> Dict[str, Any]:
        """获取因子参数"""
        return {
            'period': self._period,
            'field': self._field,
            'log_return': self._log_return
        }


# ==================== 因子注册表 ====================

# 导入CTC因子 - 高低成交量切分
try:
    from .ctc.volume_price_split import (
        HighVolReturnSum, LowVolReturnSum,
        HighVolReturnStd, LowVolReturnStd,
        HighVolAmplitude, LowVolAmplitude
    )
    _CTC_VOLUME_PRICE_FACTORS = {
        'HighVolReturnSum': HighVolReturnSum,
        'LowVolReturnSum': LowVolReturnSum,
        'HighVolReturnStd': HighVolReturnStd,
        'LowVolReturnStd': LowVolReturnStd,
        'HighVolAmplitude': HighVolAmplitude,
        'LowVolAmplitude': LowVolAmplitude,
    }
except ImportError:
    _CTC_VOLUME_PRICE_FACTORS = {}

# 导入CTC因子 - 放量缩量切分
try:
    from .ctc.volume_change_split import (
        HighVolChangeReturnSum, LowVolChangeReturnSum,
        HighVolChangeReturnStd, LowVolChangeReturnStd,
        HighVolChangeAmplitude, LowVolChangeAmplitude
    )
    _CTC_VOLUME_CHANGE_FACTORS = {
        'HighVolChangeReturnSum': HighVolChangeReturnSum,
        'LowVolChangeReturnSum': LowVolChangeReturnSum,
        'HighVolChangeReturnStd': HighVolChangeReturnStd,
        'LowVolChangeReturnStd': LowVolChangeReturnStd,
        'HighVolChangeAmplitude': HighVolChangeAmplitude,
        'LowVolChangeAmplitude': LowVolChangeAmplitude,
    }
except ImportError:
    _CTC_VOLUME_CHANGE_FACTORS = {}

# 导入CTC因子 - 高低价区间切分
try:
    from .ctc.price_volume_split import (
        HighPriceRelativeVolume, LowPriceRelativeVolume,
        HighPriceVolumeChange, LowPriceVolumeChange
    )
    _CTC_PRICE_VOLUME_FACTORS = {
        'HighPriceRelativeVolume': HighPriceRelativeVolume,
        'LowPriceRelativeVolume': LowPriceRelativeVolume,
        'HighPriceVolumeChange': HighPriceVolumeChange,
        'LowPriceVolumeChange': LowPriceVolumeChange,
    }
except ImportError:
    _CTC_PRICE_VOLUME_FACTORS = {}

# CTC不平衡度因子
try:
    from .ctc.volume_price_imbalance import (
        VolAmplitudeImbalance, VolReturnStdImbalance
    )
    _CTC_IMBALANCE_FACTORS = {
        'VolAmplitudeImbalance': VolAmplitudeImbalance,
        'VolReturnStdImbalance': VolReturnStdImbalance,
    }
except ImportError:
    _CTC_IMBALANCE_FACTORS = {}

# CTC量价相关性因子
try:
    from .ctc.price_volume_correlation import (
        PVCorr, DPVCorr, PdVCorr, DPdVCorr
    )
    _CTC_CORRELATION_FACTORS = {
        'PVCorr': PVCorr,
        'DPVCorr': DPVCorr,
        'PdVCorr': PdVCorr,
        'DPdVCorr': DPdVCorr,
    }
except ImportError as e:
    _CTC_CORRELATION_FACTORS = {}

# 合并所有CTC因子
_CTC_FACTORS = {**_CTC_VOLUME_PRICE_FACTORS, **_CTC_VOLUME_CHANGE_FACTORS, **_CTC_PRICE_VOLUME_FACTORS, **_CTC_IMBALANCE_FACTORS, **_CTC_CORRELATION_FACTORS}

# 导入动量因子
try:
    from .momentum.momentum_factors import MomentumFactor
    _MOMENTUM_FACTORS = {'MomentumFactor': MomentumFactor}
except ImportError:
    _MOMENTUM_FACTORS = {}

# 导入ML因子
try:
    from .ml.cross_section_factor import MLCrossSectionalFactor
    _ML_FACTORS = {'MLCrossSectionalFactor': MLCrossSectionalFactor}
except ImportError:
    _ML_FACTORS = {}

# 因子类注册表，用于通过名称动态创建因子实例
_FACTOR_REGISTRY = {
    'CloseOverMA': CloseOverMA,
    'RSI': RSI,
    'Momentum': Momentum,
    'MACD': MACD,
    'BollingerBands': BollingerBands,
    'FutureReturn': FutureReturn,
    **_CTC_FACTORS,
    **_MOMENTUM_FACTORS,
    **_ML_FACTORS,
}


def get_factor_class(factor_name: str):
    """
    根据因子名称获取因子类
    
    Args:
        factor_name: 因子名称
        
    Returns:
        class: 因子类
        
    Raises:
        ValueError: 如果因子名称不存在
    """
    if factor_name not in _FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor: {factor_name}. Available factors: {list(_FACTOR_REGISTRY.keys())}")
    return _FACTOR_REGISTRY[factor_name]


def list_available_factors():
    """
    列出所有可用的因子
    
    Returns:
        list: 因子名称列表
    """
    return list(_FACTOR_REGISTRY.keys())


def create_factor(factor_name: str, **params):
    """
    根据名称和参数创建因子实例
    
    Args:
        factor_name: 因子名称
        **params: 因子参数
        
    Returns:
        OHLCVFactorCalculator: 因子实例
    """
    factor_class = get_factor_class(factor_name)
    return factor_class(**params)
