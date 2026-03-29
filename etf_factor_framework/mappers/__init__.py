"""
仓位映射模块

将因子值转换为可交易仓位的映射器集合。

Available Mappers:
    - RankBasedMapper: 基于排名的Top K选择
    - DirMapper: 直接映射，将因子值作为仓位
    - QuantileMapper: 分位数多空映射
    - ZScoreMapper: Z-Score映射
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # 相对导入（包内使用）
    from .base_mapper import BasePositionMapper
    from .position_mappers import (
        RankBasedMapper,
        DirMapper,
        QuantileMapper,
        ZScoreMapper,
        create_top_k_mapper,
        create_bottom_k_mapper,
        create_equal_weight_mapper,
    )
    from .weight_methods import equal_weight, normalize_weights
    from .position_adjuster import PositionAdjuster
except ImportError:
    # 绝对导入（独立运行）
    from mappers.base_mapper import BasePositionMapper
    from mappers.position_mappers import (
        RankBasedMapper,
        DirMapper,
        QuantileMapper,
        ZScoreMapper,
        create_top_k_mapper,
        create_bottom_k_mapper,
        create_equal_weight_mapper,
    )
    from mappers.weight_methods import equal_weight, normalize_weights
    from mappers.position_adjuster import PositionAdjuster

__all__ = [
    # 映射器基类
    'BasePositionMapper',
    # 具体映射器
    'RankBasedMapper',
    'DirMapper',
    'QuantileMapper',
    'ZScoreMapper',
    # 仓位调整器（A股）
    'PositionAdjuster',
    # 工厂函数
    'create_top_k_mapper',
    'create_bottom_k_mapper',
    'create_equal_weight_mapper',
    # 权重方法
    'equal_weight',
    'normalize_weights',
]
