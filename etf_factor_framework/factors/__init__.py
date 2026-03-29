"""
因子计算模块

提供各类因子计算器的实现。
"""

import warnings

from .ohlcv_calculator import OHLCVFactorCalculator
from .technical_factors import (
    CloseOverMA, RSI, Momentum, MACD, BollingerBands, FutureReturn,
    get_factor_class, list_available_factors, create_factor
)
from .momentum.momentum_factors import MomentumFactor
from .leakage_detector import (
    LeakageDetector,
    LeakageReport,
    FutureDataLeakageError,
    detect_leakage
)

# CTC因子导入
try:
    from .ctc.volume_price_split import (
        HighVolReturnSum,
        LowVolReturnSum,
        HighVolReturnStd,
        LowVolReturnStd,
        HighVolAmplitude,
        LowVolAmplitude,
    )
except ImportError as e:
    warnings.warn(f"Failed to import CTC volume_price_split factors: {e}")
    HighVolReturnSum = None
    LowVolReturnSum = None
    HighVolReturnStd = None
    LowVolReturnStd = None
    HighVolAmplitude = None
    LowVolAmplitude = None

try:
    from .ctc.volume_change_split import (
        HighVolChangeReturnSum,
        LowVolChangeReturnSum,
        HighVolChangeReturnStd,
        LowVolChangeReturnStd,
        HighVolChangeAmplitude,
        LowVolChangeAmplitude,
    )
except ImportError as e:
    warnings.warn(f"Failed to import CTC volume_change_split factors: {e}")
    HighVolChangeReturnSum = None
    LowVolChangeReturnSum = None
    HighVolChangeReturnStd = None
    LowVolChangeReturnStd = None
    HighVolChangeAmplitude = None
    LowVolChangeAmplitude = None

# CTC因子导入 - 高低价区间切分
try:
    from .ctc.price_volume_split import (
        HighPriceRelativeVolume,
        LowPriceRelativeVolume,
        HighPriceVolumeChange,
        LowPriceVolumeChange,
    )
except ImportError as e:
    warnings.warn(f"Failed to import CTC price_volume_split factors: {e}")
    HighPriceRelativeVolume = None
    LowPriceRelativeVolume = None
    HighPriceVolumeChange = None
    LowPriceVolumeChange = None

# CTC因子导入 - 不平衡度因子
try:
    from .ctc.volume_price_imbalance import (
        VolAmplitudeImbalance,
        VolReturnStdImbalance,
    )
except ImportError as e:
    warnings.warn(f"Failed to import CTC volume_price_imbalance factors: {e}")
    VolAmplitudeImbalance = None
    VolReturnStdImbalance = None

# CTC因子导入 - 量价相关性因子
try:
    from .ctc.price_volume_correlation import (
        PVCorr,
        DPVCorr,
        PdVCorr,
        DPdVCorr,
        PriceVolumeCorrelation,
    )
except ImportError as e:
    warnings.warn(f"Failed to import CTC price_volume_correlation factors: {e}")
    PVCorr = None
    DPVCorr = None
    PdVCorr = None
    DPdVCorr = None
    PriceVolumeCorrelation = None

__all__ = [
    # 基类
    'OHLCVFactorCalculator',
    # 技术因子
    'CloseOverMA',
    'RSI',
    'Momentum',
    'MACD',
    'BollingerBands',
    # Baseline因子（前瞻因子）
    'FutureReturn',
    # 因子注册表
    'get_factor_class',
    'list_available_factors',
    'create_factor',
    # 数据泄露检测
    'LeakageDetector',
    'LeakageReport',
    'FutureDataLeakageError',
    'detect_leakage',
    # CTC因子 - 高低成交量切分
    'HighVolReturnSum',
    'LowVolReturnSum',
    'HighVolReturnStd',
    'LowVolReturnStd',
    'HighVolAmplitude',
    'LowVolAmplitude',
    # CTC因子 - 放量缩量切分
    'HighVolChangeReturnSum',
    'LowVolChangeReturnSum',
    'HighVolChangeReturnStd',
    'LowVolChangeReturnStd',
    'HighVolChangeAmplitude',
    'LowVolChangeAmplitude',
    # CTC因子 - 高低价区间切分
    'HighPriceRelativeVolume',
    'LowPriceRelativeVolume',
    'HighPriceVolumeChange',
    'LowPriceVolumeChange',
    # CTC因子 - 不平衡度因子
    'VolAmplitudeImbalance',
    'VolReturnStdImbalance',
    # CTC因子 - 量价相关性因子
    'PVCorr',
    'DPVCorr',
    'PdVCorr',
    'DPdVCorr',
    'PriceVolumeCorrelation',
    # 动量因子
    'MomentumFactor',
]
