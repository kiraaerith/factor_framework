"""
质量因子 (Quality Factors)

包含衡量公司财务质量、盈余质量的因子。
"""

from .piotroski_fscore import PiotroskiFScore
from .accrual_ratio import AccrualRatio
from .earnings_quality import EarningsQuality
from .goodwill_risk import GoodwillRisk
from .interest_coverage import InterestCoverage
from .net_debt_to_equity import NetDebtToEquity

__all__ = ["PiotroskiFScore", "AccrualRatio", "EarningsQuality", "GoodwillRisk", "InterestCoverage", "NetDebtToEquity"]
