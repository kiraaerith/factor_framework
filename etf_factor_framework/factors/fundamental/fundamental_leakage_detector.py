"""
FundamentalLeakageDetector: 基本面因子未来数据泄露检测器

检测原理：
1. 将 FundamentalData 的时间范围按 split_ratio 截断（保留前 split_ratio 比例的交易日）
2. 分别用截断数据和完整数据计算因子值
3. 比较两者在重叠时间段的因子值是否一致
4. 不一致 → 因子使用了未来财务数据（如用了 date 报告期而非 report_date 披露日）
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator


@dataclass
class FundamentalLeakageReport:
    """基本面因子泄露检测报告"""
    factor_name: str
    has_leakage: bool
    split_ratio: float
    short_periods: int
    full_periods: int
    overlap_periods: int
    mismatched_periods: int
    mismatch_ratio: float
    max_absolute_diff: float
    mismatched_dates: List[Any]

    def print_report(self):
        print("=" * 60)
        print(f"基本面因子泄露检测: {self.factor_name}")
        print("=" * 60)
        status = "[LEAKAGE] 存在数据泄露" if self.has_leakage else "[OK] 无数据泄露"
        print(f"检测结果: {status}")
        print(f"切割比例: {self.split_ratio:.1%}")
        print(f"截断数据集长度: {self.short_periods} 个交易日")
        print(f"完整数据集长度: {self.full_periods} 个交易日")
        print(f"重叠时间段: {self.overlap_periods} 个交易日")
        print("-" * 60)
        print(f"不匹配日期数: {self.mismatched_periods}")
        print(f"不匹配比例: {self.mismatch_ratio:.4%}")
        print(f"最大绝对差异: {self.max_absolute_diff:.8f}")
        if self.mismatched_dates:
            print(f"不匹配日期示例: {self.mismatched_dates[:5]}")
        print("=" * 60)


class FundamentalLeakageDetector:
    """
    基本面因子未来数据泄露检测器

    使用示例::

        fd = FundamentalData('2013-01-01', '2025-12-31')
        detector = FundamentalLeakageDetector(split_ratio=0.7)
        report = detector.detect(ROE(), fd)
        assert not report.has_leakage
    """

    def __init__(
        self,
        split_ratio: float = 0.7,
        tolerance: float = 1e-8,
    ):
        """
        Args:
            split_ratio: 截断比例，保留前 split_ratio 比例的交易日，默认 0.7
            tolerance: 数值差异容忍度，默认 1e-8
        """
        if not (0 < split_ratio < 1):
            raise ValueError(f"split_ratio 必须在 (0, 1) 范围内，当前值: {split_ratio}")
        self.split_ratio = split_ratio
        self.tolerance = tolerance

    def detect(
        self,
        calculator: FundamentalFactorCalculator,
        fundamental_data: FundamentalData,
        pricevol_data=None,
    ) -> FundamentalLeakageReport:
        """
        检测因子是否存在未来数据泄露

        Args:
            calculator: 因子计算器实例
            fundamental_data: 完整的 FundamentalData
            pricevol_data: 完整的 PriceVolData（量价因子需要，基本面因子可不传）

        Returns:
            FundamentalLeakageReport
        """
        trading_dates = fundamental_data.trading_dates
        n = len(trading_dates)
        split_point = int(n * self.split_ratio)

        if split_point < 1:
            raise ValueError("split_point 过小，请增大 split_ratio 或回测区间")

        split_date = trading_dates[split_point - 1]

        # 预加载完整原始数据，确保 truncate() 能复用，避免截断运行重新执行 SQL
        # （SQL 按 date 过滤，截断后重新 SQL 会丢失 date > split_date 但已披露的财报）
        fundamental_data._load_raw_data()

        # 先用完整数据计算因子（触发 pricevol_data 加载缓存）
        full_factor = calculator.calculate(fundamental_data, pricevol_data=pricevol_data)

        # 截断数据（只保留 split_date 之前的数据）
        short_fd = fundamental_data.truncate(split_date.strftime("%Y-%m-%d"))
        short_pvd = pricevol_data.truncate(split_date.strftime("%Y-%m-%d")) if pricevol_data else None

        print(f"  - 完整数据集: {n} 个交易日")
        print(f"  - 截断数据集: {len(short_fd.trading_dates)} 个交易日 (截至 {split_date.date()})")

        # 用截断数据计算因子
        short_factor = calculator.calculate(short_fd, pricevol_data=short_pvd)

        # 对比重叠时间段
        return self._compare(full_factor, short_factor, calculator.name)

    def _compare(self, full_factor, short_factor, factor_name: str) -> FundamentalLeakageReport:
        """对比完整因子和截断因子在重叠时间段的值"""
        full_dates = full_factor.dates      # ndarray datetime64
        short_dates = short_factor.dates    # ndarray datetime64
        overlap_dates = np.intersect1d(full_dates, short_dates)

        full_symbols = full_factor.symbols   # ndarray str
        short_symbols = short_factor.symbols  # ndarray str
        overlap_symbols = np.intersect1d(full_symbols, short_symbols)

        if len(overlap_dates) == 0 or len(overlap_symbols) == 0:
            return FundamentalLeakageReport(
                factor_name=factor_name,
                has_leakage=False,
                split_ratio=self.split_ratio,
                short_periods=short_factor.n_periods,
                full_periods=full_factor.n_periods,
                overlap_periods=0,
                mismatched_periods=0,
                mismatch_ratio=0.0,
                max_absolute_diff=0.0,
                mismatched_dates=[],
            )

        # 用 searchsorted 获取索引
        full_sym_idx = np.searchsorted(full_symbols, overlap_symbols)
        short_sym_idx = np.searchsorted(short_symbols, overlap_symbols)
        full_date_idx = np.searchsorted(full_dates, overlap_dates)
        short_date_idx = np.searchsorted(short_dates, overlap_dates)

        full_vals = full_factor.values[np.ix_(full_sym_idx, full_date_idx)]    # (N_overlap, T_overlap)
        short_vals = short_factor.values[np.ix_(short_sym_idx, short_date_idx)]

        diff = np.abs(full_vals - short_vals)

        # 两个都是 NaN → 视为相等
        full_nan = np.isnan(full_vals)
        short_nan = np.isnan(short_vals)
        both_nan = full_nan & short_nan
        mismatch = (diff > self.tolerance) & ~both_nan
        # 一个 NaN 一个不是 → 不相等
        one_nan = full_nan ^ short_nan
        mismatch = mismatch | one_nan

        # 按日期维度（axis=1）判断哪些日期有不匹配
        mismatched_date_mask = mismatch.any(axis=0)
        mismatched_dates = overlap_dates[mismatched_date_mask].tolist()

        total_cells = len(overlap_symbols) * len(overlap_dates)
        mismatched_cells = int(mismatch.sum())
        mismatch_ratio = mismatched_cells / total_cells if total_cells > 0 else 0.0

        valid_diff = diff[~np.isnan(diff)]
        max_abs_diff = float(np.max(valid_diff)) if len(valid_diff) > 0 else 0.0

        return FundamentalLeakageReport(
            factor_name=factor_name,
            has_leakage=len(mismatched_dates) > 0,
            split_ratio=self.split_ratio,
            short_periods=short_factor.n_periods,
            full_periods=full_factor.n_periods,
            overlap_periods=len(overlap_dates),
            mismatched_periods=len(mismatched_dates),
            mismatch_ratio=mismatch_ratio,
            max_absolute_diff=max_abs_diff,
            mismatched_dates=mismatched_dates,
        )
