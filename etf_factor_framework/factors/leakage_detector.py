"""
未来数据泄露检测模块

用于自动检测因子计算函数是否存在未来数据泄露问题。

检测原理：
1. 将完整的数据集沿时间维度切开，构成短数据集
2. 用因子计算函数分别计算短数据集和长数据集的因子值
3. 比较两者在重叠时间段的因子值是否完全一致
4. 如果不一致，说明存在未来数据泄露

示例：
    原始数据: 2024-01-01 到 2024-12-31 (365天)
    短数据:   2024-01-01 到 2024-06-30 (前50%)
    长数据:   2024-01-01 到 2024-12-31 (100%)
    
    如果因子计算器使用了未来数据（如向后看的价格），
    那么短数据和长数据在 2024-01-01 到 2024-06-30 期间的因子值会不一致。
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

try:
    from ..core.ohlcv_data import OHLCVData
    from ..core.factor_data import FactorData
    from .ohlcv_calculator import OHLCVFactorCalculator
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData
    from factors.ohlcv_calculator import OHLCVFactorCalculator


@dataclass
class LeakageReport:
    """
    数据泄露检测报告
    
    Attributes:
        factor_name: 因子名称
        has_leakage: 是否存在数据泄露
        split_ratio: 数据集切割比例（短数据集占原数据集的比例）
        short_data_periods: 短数据集的时间长度
        full_data_periods: 完整数据集的时间长度
        overlap_periods: 重叠时间段长度
        mismatched_periods: 不匹配的时间点数量
        mismatch_ratio: 不匹配比例（mismatched_periods / overlap_periods）
        max_absolute_diff: 最大绝对差异
        max_relative_diff: 最大相对差异（相对于数值本身）
        mismatched_dates: 不匹配的时间点列表
        details: 详细对比信息
    """
    factor_name: str
    has_leakage: bool
    split_ratio: float
    short_data_periods: int
    full_data_periods: int
    overlap_periods: int
    mismatched_periods: int
    mismatch_ratio: float
    max_absolute_diff: float
    max_relative_diff: float
    mismatched_dates: List[Any]
    details: Dict[str, Any]
    
    def __repr__(self) -> str:
        status = "⚠️ 检测到数据泄露" if self.has_leakage else "✅ 无数据泄露"
        return (
            f"LeakageReport({self.factor_name}: {status}, "
            f"mismatch={self.mismatch_ratio:.2%}, "
            f"max_diff={self.max_absolute_diff:.6f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'factor_name': self.factor_name,
            'has_leakage': self.has_leakage,
            'split_ratio': self.split_ratio,
            'short_data_periods': self.short_data_periods,
            'full_data_periods': self.full_data_periods,
            'overlap_periods': self.overlap_periods,
            'mismatched_periods': self.mismatched_periods,
            'mismatch_ratio': self.mismatch_ratio,
            'max_absolute_diff': self.max_absolute_diff,
            'max_relative_diff': self.max_relative_diff,
            'mismatched_dates': self.mismatched_dates,
        }
    
    def print_report(self):
        """打印详细检测报告"""
        print("=" * 60)
        print(f"因子数据泄露检测报告: {self.factor_name}")
        print("=" * 60)
        print(f"检测结果: {'⚠️ 存在数据泄露' if self.has_leakage else '✅ 无数据泄露'}")
        print(f"切割比例: {self.split_ratio:.1%}")
        print(f"短数据集长度: {self.short_data_periods} 期")
        print(f"完整数据集长度: {self.full_data_periods} 期")
        print(f"重叠时间段: {self.overlap_periods} 期")
        print("-" * 60)
        print(f"不匹配时间点数: {self.mismatched_periods}")
        print(f"不匹配比例: {self.mismatch_ratio:.4%}")
        print(f"最大绝对差异: {self.max_absolute_diff:.8f}")
        print(f"最大相对差异: {self.max_relative_diff:.4%}")
        if self.mismatched_dates:
            print(f"不匹配时间点示例: {self.mismatched_dates[:10]}")
            if len(self.mismatched_dates) > 10:
                print(f"  ... 还有 {len(self.mismatched_dates) - 10} 个")
        print("=" * 60)


class LeakageDetector:
    """
    未来数据泄露检测器
    
    用于检测因子计算器是否存在使用未来数据的问题。
    
    Example:
        >>> from core.ohlcv_data import OHLCVData
        >>> from factors.technical_factors import CloseOverMA
        >>> from factors.leakage_detector import LeakageDetector
        >>> 
        >>> # 准备数据
        >>> ohlcv_data = OHLCVData(...)
        >>> 
        >>> # 创建检测器
        >>> detector = LeakageDetector(split_ratio=0.5)
        >>> 
        >>> # 检测因子
        >>> factor = CloseOverMA(period=20)
        >>> report = detector.detect(factor, ohlcv_data)
        >>> 
        >>> # 查看结果
        >>> print(report.has_leakage)  # False 表示无泄露
        >>> report.print_report()  # 打印详细报告
    """
    
    def __init__(
        self,
        split_ratio: float = 0.5,
        tolerance: float = 1e-10,
        relative_tolerance: float = 1e-8,
        verbose: bool = False
    ):
        """
        初始化数据泄露检测器
        
        Args:
            split_ratio: 数据集切割比例，短数据集占原数据集的比例，默认0.5（前50%）
            tolerance: 绝对误差容忍度，用于判断两个值是否相等
            relative_tolerance: 相对误差容忍度
            verbose: 是否打印详细信息
            
        Raises:
            ValueError: 如果split_ratio不在(0, 1)范围内
        """
        if not (0 < split_ratio < 1):
            raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
        
        self.split_ratio = split_ratio
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.verbose = verbose
    
    def detect(
        self,
        calculator: OHLCVFactorCalculator,
        ohlcv_data: OHLCVData
    ) -> LeakageReport:
        """
        检测因子计算器是否存在未来数据泄露
        
        Args:
            calculator: 因子计算器实例
            ohlcv_data: 完整的OHLCV数据
            
        Returns:
            LeakageReport: 检测报告
        """
        if self.verbose:
            print(f"开始检测因子: {calculator.name}")
            print(f"数据形状: {ohlcv_data.shape}")
            print(f"切割比例: {self.split_ratio:.1%}")
        
        # 1. 切割数据集
        short_ohlcv, full_ohlcv = self._split_data(ohlcv_data)
        
        if self.verbose:
            print(f"短数据集: {short_ohlcv.n_periods} 期")
            print(f"完整数据集: {full_ohlcv.n_periods} 期")
        
        # 2. 计算因子值
        short_factor = calculator.calculate(short_ohlcv)
        full_factor = calculator.calculate(full_ohlcv)
        
        # 3. 对比结果
        return self._compare_factors(short_factor, full_factor, calculator.name)
    
    def _split_data(self, ohlcv_data: OHLCVData) -> Tuple[OHLCVData, OHLCVData]:
        """
        沿时间维度切割OHLCV数据
        
        Args:
            ohlcv_data: 原始OHLCV数据
            
        Returns:
            Tuple[OHLCVData, OHLCVData]: (短数据集, 完整数据集)
        """
        n_periods = ohlcv_data.n_periods
        split_point = int(n_periods * self.split_ratio)
        
        if split_point < 1:
            raise ValueError(
                f"Split point {split_point} is too small. "
                f"Need at least 1 period. Consider using larger split_ratio or more data."
            )
        
        # 获取日期列表
        dates = ohlcv_data.dates
        short_dates = dates[:split_point]
        
        # 切割数据
        short_ohlcv = OHLCVData(
            open=ohlcv_data.open[short_dates],
            high=ohlcv_data.high[short_dates],
            low=ohlcv_data.low[short_dates],
            close=ohlcv_data.close[short_dates],
            volume=ohlcv_data.volume[short_dates],
            metadata={'split': 'short', 'split_ratio': self.split_ratio}
        )
        
        # 完整数据保持原样
        full_ohlcv = ohlcv_data
        
        return short_ohlcv, full_ohlcv
    
    def _compare_factors(
        self,
        short_factor: FactorData,
        full_factor: FactorData,
        factor_name: str
    ) -> LeakageReport:
        """
        对比短数据集和长数据集计算出的因子值
        
        Args:
            short_factor: 短数据集计算的因子值
            full_factor: 完整数据集计算的因子值
            factor_name: 因子名称
            
        Returns:
            LeakageReport: 对比报告
        """
        # 获取重叠的时间点
        short_dates = set(short_factor.dates)
        full_dates = set(full_factor.dates)
        overlap_dates = sorted(list(short_dates & full_dates))
        
        # 获取共同的标的
        short_symbols = set(short_factor.symbols)
        full_symbols = set(full_factor.symbols)
        overlap_symbols = sorted(list(short_symbols & full_symbols))
        
        # 提取重叠部分的数据
        short_values = short_factor.values.loc[overlap_symbols, overlap_dates]
        full_values = full_factor.values.loc[overlap_symbols, overlap_dates]
        
        # 计算差异
        diff = short_values - full_values
        
        # 找出不匹配的位置（考虑NaN的情况）
        # 两个都是NaN视为相等
        # 一个是NaN一个不是，视为不相等
        def values_equal(v1, v2):
            """判断两个值是否相等，考虑NaN"""
            if pd.isna(v1) and pd.isna(v2):
                return True
            if pd.isna(v1) or pd.isna(v2):
                return False
            abs_diff = abs(v1 - v2)
            rel_diff = abs_diff / (abs(v1) + 1e-12)
            return abs_diff <= self.tolerance or rel_diff <= self.relative_tolerance
        
        # 逐元素比较
        mismatch_mask = pd.DataFrame(
            [[not values_equal(short_values.iloc[i, j], full_values.iloc[i, j]) 
              for j in range(len(overlap_dates))] 
             for i in range(len(overlap_symbols))],
            index=overlap_symbols,
            columns=overlap_dates
        )
        
        # 统计不匹配
        mismatched_count = mismatch_mask.sum().sum()
        total_values = len(overlap_symbols) * len(overlap_dates)
        mismatch_ratio = mismatched_count / total_values if total_values > 0 else 0
        
        # 找出不匹配的时间点
        mismatched_dates = mismatch_mask.any(axis=0)  # 按列统计，只要有一个symbol不匹配就算
        mismatched_date_list = [d for d, mismatch in mismatched_dates.items() if mismatch]
        
        # 计算最大差异（排除NaN）
        valid_diff = diff.values[~np.isnan(diff.values)]
        max_abs_diff = np.max(np.abs(valid_diff)) if len(valid_diff) > 0 else 0.0
        
        # 计算最大相对差异
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(diff.values) / (np.abs(short_values.values) + 1e-12)
        valid_rel_diff = rel_diff[~np.isnan(rel_diff) & ~np.isinf(rel_diff)]
        max_rel_diff = np.max(valid_rel_diff) if len(valid_rel_diff) > 0 else 0.0
        
        # 判断是否有泄露
        has_leakage = mismatched_count > 0
        
        # 收集详细信息
        details = {
            'short_factor_shape': short_factor.shape,
            'full_factor_shape': full_factor.shape,
            'overlap_symbols': overlap_symbols,
            'overlap_dates': overlap_dates,
            'diff_statistics': {
                'mean_abs_diff': np.mean(np.abs(valid_diff)) if len(valid_diff) > 0 else 0.0,
                'std_abs_diff': np.std(np.abs(valid_diff)) if len(valid_diff) > 0 else 0.0,
                'median_abs_diff': np.median(np.abs(valid_diff)) if len(valid_diff) > 0 else 0.0,
            }
        }
        
        report = LeakageReport(
            factor_name=factor_name,
            has_leakage=has_leakage,
            split_ratio=self.split_ratio,
            short_data_periods=short_factor.n_periods,
            full_data_periods=full_factor.n_periods,
            overlap_periods=len(overlap_dates),
            mismatched_periods=len(mismatched_date_list),
            mismatch_ratio=mismatch_ratio,
            max_absolute_diff=max_abs_diff,
            max_relative_diff=max_rel_diff,
            mismatched_dates=mismatched_date_list,
            details=details
        )
        
        if self.verbose:
            report.print_report()
        
        return report
    
    def detect_all_factors(
        self,
        calculators: List[OHLCVFactorCalculator],
        ohlcv_data: OHLCVData
    ) -> Dict[str, LeakageReport]:
        """
        批量检测多个因子计算器
        
        Args:
            calculators: 因子计算器列表
            ohlcv_data: 完整的OHLCV数据
            
        Returns:
            Dict[str, LeakageReport]: 每个因子的检测报告
        """
        reports = {}
        for calculator in calculators:
            report = self.detect(calculator, ohlcv_data)
            reports[calculator.name] = report
        return reports
    
    def validate_no_leakage(
        self,
        calculator: OHLCVFactorCalculator,
        ohlcv_data: OHLCVData,
        raise_error: bool = True
    ) -> bool:
        """
        验证因子计算器无数据泄露，如有泄露则抛出异常或返回False
        
        Args:
            calculator: 因子计算器实例
            ohlcv_data: 完整的OHLCV数据
            raise_error: 发现泄露时是否抛出异常，默认True
            
        Returns:
            bool: True表示无泄露，False表示有泄露
            
        Raises:
            FutureDataLeakageError: 如果检测到数据泄露且raise_error=True
        """
        report = self.detect(calculator, ohlcv_data)
        
        if report.has_leakage:
            if raise_error:
                raise FutureDataLeakageError(
                    f"检测到未来数据泄露: {report.factor_name}\n"
                    f"不匹配比例: {report.mismatch_ratio:.4%}\n"
                    f"最大差异: {report.max_absolute_diff:.8f}\n"
                    f"不匹配时间点: {report.mismatched_dates[:5]}"
                )
            return False
        return True


class FutureDataLeakageError(Exception):
    """未来数据泄露异常"""
    pass


def detect_leakage(
    calculator: OHLCVFactorCalculator,
    ohlcv_data: OHLCVData,
    split_ratio: float = 0.5,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> LeakageReport:
    """
    便捷的泄露检测函数
    
    Args:
        calculator: 因子计算器实例
        ohlcv_data: 完整的OHLCV数据
        split_ratio: 数据集切割比例
        tolerance: 误差容忍度
        verbose: 是否打印详细信息
        
    Returns:
        LeakageReport: 检测报告
        
    Example:
        >>> from factors.leakage_detector import detect_leakage
        >>> from factors.technical_factors import CloseOverMA
        >>> 
        >>> factor = CloseOverMA(period=20)
        >>> report = detect_leakage(factor, ohlcv_data)
        >>> 
        >>> if report.has_leakage:
        ...     print(f"警告: {report.factor_name} 存在数据泄露!")
        ... else:
        ...     print(f"{report.factor_name} 无数据泄露")
    """
    detector = LeakageDetector(
        split_ratio=split_ratio,
        tolerance=tolerance,
        verbose=verbose
    )
    return detector.detect(calculator, ohlcv_data)


# ==================== 用于测试的有意泄露的因子（用于验证检测器） ====================

class LeakyFactor(OHLCVFactorCalculator):
    """
    故意使用未来数据的因子（用于测试检测器）
    
    这个因子使用了未来价格数据，用于验证检测器能否正确发现问题。
    """
    
    def __init__(self, look_ahead: int = 5):
        """
        初始化有泄露的因子
        
        Args:
            look_ahead: 向前看的周期数
        """
        self._look_ahead = look_ahead
        super().__init__()
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        """
        计算因子值（使用未来数据！）
        
        错误示范：使用了未来的收盘价
        """
        close = ohlcv_data.close
        
        # 这是错误的！使用了未来的价格
        future_close = close.shift(-self._look_ahead, axis=1)
        factor_value = future_close / close - 1  # 使用未来收益率作为因子
        
        return FactorData(
            values=factor_value,
            name=self.name,
            params=self.params
        )
    
    @property
    def name(self) -> str:
        return f"LeakyFactor_{self._look_ahead}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {'look_ahead': self._look_ahead}
