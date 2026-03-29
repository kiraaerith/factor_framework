"""
CTC量价因子模块

基于CTC Institute文章改造的量价因子，使用日频OHLCV数据。
"""

# 高低成交量切分因子
from .volume_price_split import (
    HighVolReturnSum,
    LowVolReturnSum,
    HighVolReturnStd,
    LowVolReturnStd,
    HighVolAmplitude,
    LowVolAmplitude,
)

# 放量缩量切分因子
from .volume_change_split import (
    HighVolChangeReturnSum,
    LowVolChangeReturnSum,
    HighVolChangeReturnStd,
    LowVolChangeReturnStd,
    HighVolChangeAmplitude,
    LowVolChangeAmplitude,
)

# 高低价区间切分因子
from .price_volume_split import (
    HighPriceRelativeVolume,
    LowPriceRelativeVolume,
    HighPriceVolumeChange,
    LowPriceVolumeChange,
)

# 不平衡度因子
from .volume_price_imbalance import (
    VolAmplitudeImbalance,
    VolReturnStdImbalance,
)

__all__ = [
    # 高低成交量切分因子
    'HighVolReturnSum',
    'LowVolReturnSum',
    'HighVolReturnStd',
    'LowVolReturnStd',
    'HighVolAmplitude',
    'LowVolAmplitude',
    # 放量缩量切分因子
    'HighVolChangeReturnSum',
    'LowVolChangeReturnSum',
    'HighVolChangeReturnStd',
    'LowVolChangeReturnStd',
    'HighVolChangeAmplitude',
    'LowVolChangeAmplitude',
    # 高低价区间切分因子
    'HighPriceRelativeVolume',
    'LowPriceRelativeVolume',
    'HighPriceVolumeChange',
    'LowPriceVolumeChange',
    # 不平衡度因子
    'VolAmplitudeImbalance',
    'VolReturnStdImbalance',
]
