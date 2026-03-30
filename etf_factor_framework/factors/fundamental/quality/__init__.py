"""
质量因子 (Quality Factors)

包含衡量公司财务质量、盈余质量的因子。
"""

from .piotroski_fscore import PiotroskiFScore
from .accrual_ratio import AccrualRatio

__all__ = ["PiotroskiFScore", "AccrualRatio"]
