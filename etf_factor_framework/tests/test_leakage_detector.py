"""
数据泄露检测器单元测试模块

测试 LeakageDetector 的各项功能：
1. 正常因子不应被检测出泄露
2. 故意构造的泄露因子应该被检测到
3. 数据切割功能
4. 对比逻辑
5. 批量检测功能
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.ohlcv_data import OHLCVData
from core.factor_data import FactorData
from factors.technical_factors import CloseOverMA, RSI, Momentum
from factors.leakage_detector import (
    LeakageDetector,
    LeakageReport,
    FutureDataLeakageError,
    detect_leakage,
    LeakyFactor
)


class TestLeakageDetectorBasic(unittest.TestCase):
    """测试检测器基本功能"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2', 'ETF3']
        self.dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 生成模拟价格数据
        close = pd.DataFrame(
            np.random.randn(3, 100).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(3, 100)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = LeakageDetector(split_ratio=0.5)
        self.assertEqual(detector.split_ratio, 0.5)
        self.assertEqual(detector.tolerance, 1e-10)
        
        # 测试无效的split_ratio
        with self.assertRaises(ValueError):
            LeakageDetector(split_ratio=0)
        
        with self.assertRaises(ValueError):
            LeakageDetector(split_ratio=1)
        
        with self.assertRaises(ValueError):
            LeakageDetector(split_ratio=-0.5)
    
    def test_data_splitting(self):
        """测试数据切割功能"""
        detector = LeakageDetector(split_ratio=0.5)
        short_ohlcv, full_ohlcv = detector._split_data(self.ohlcv)
        
        # 验证形状
        self.assertEqual(short_ohlcv.n_periods, 50)  # 100 * 0.5
        self.assertEqual(full_ohlcv.n_periods, 100)
        self.assertEqual(short_ohlcv.n_assets, 3)
        self.assertEqual(full_ohlcv.n_assets, 3)
        
        # 验证日期
        self.assertEqual(list(short_ohlcv.dates), list(self.dates[:50]))
        self.assertEqual(list(full_ohlcv.dates), list(self.dates))
    
    def test_data_splitting_different_ratios(self):
        """测试不同切割比例"""
        for ratio in [0.3, 0.5, 0.7, 0.8]:
            detector = LeakageDetector(split_ratio=ratio)
            short_ohlcv, _ = detector._split_data(self.ohlcv)
            
            expected_periods = int(100 * ratio)
            self.assertEqual(short_ohlcv.n_periods, expected_periods)


class TestLeakageDetection(unittest.TestCase):
    """测试数据泄露检测功能"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        close = pd.DataFrame(
            np.random.randn(2, 60).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 60)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_normal_factor_no_leakage(self):
        """测试正常因子无数据泄露"""
        detector = LeakageDetector(split_ratio=0.5)
        
        # 测试 CloseOverMA
        factor = CloseOverMA(period=10)
        report = detector.detect(factor, self.ohlcv)
        
        self.assertFalse(report.has_leakage)
        self.assertEqual(report.mismatched_periods, 0)
        self.assertEqual(report.mismatch_ratio, 0.0)
        self.assertLess(report.max_absolute_diff, 1e-10)
    
    def test_rsi_no_leakage(self):
        """测试RSI因子无数据泄露"""
        detector = LeakageDetector(split_ratio=0.5)
        
        factor = RSI(period=14)
        report = detector.detect(factor, self.ohlcv)
        
        self.assertFalse(report.has_leakage)
        self.assertEqual(report.mismatched_periods, 0)
    
    def test_momentum_no_leakage(self):
        """测试Momentum因子无数据泄露"""
        detector = LeakageDetector(split_ratio=0.5)
        
        factor = Momentum(period=10)
        report = detector.detect(factor, self.ohlcv)
        
        self.assertFalse(report.has_leakage)
        self.assertEqual(report.mismatched_periods, 0)
    
    def test_leaky_factor_detected(self):
        """测试故意构造的泄露因子被检测到"""
        detector = LeakageDetector(split_ratio=0.5)
        
        # 使用故意构造的有泄露的因子
        factor = LeakyFactor(look_ahead=5)
        report = detector.detect(factor, self.ohlcv)
        
        # 应该检测到泄露
        self.assertTrue(report.has_leakage)
        self.assertGreater(report.mismatched_periods, 0)
        self.assertGreater(report.mismatch_ratio, 0)
    
    def test_leaky_factor_with_different_look_ahead(self):
        """测试不同look_ahead的泄露因子"""
        for look_ahead in [1, 3, 5, 10]:
            detector = LeakageDetector(split_ratio=0.6)
            factor = LeakyFactor(look_ahead=look_ahead)
            report = detector.detect(factor, self.ohlcv)
            
            # 都应该检测到泄露
            self.assertTrue(
                report.has_leakage,
                f"LeakyFactor with look_ahead={look_ahead} should be detected"
            )


class TestLeakageReport(unittest.TestCase):
    """测试检测报告"""
    
    def test_report_creation(self):
        """测试报告创建"""
        report = LeakageReport(
            factor_name="TestFactor",
            has_leakage=True,
            split_ratio=0.5,
            short_data_periods=50,
            full_data_periods=100,
            overlap_periods=50,
            mismatched_periods=10,
            mismatch_ratio=0.2,
            max_absolute_diff=0.01,
            max_relative_diff=0.05,
            mismatched_dates=[1, 2, 3],
            details={}
        )
        
        self.assertEqual(report.factor_name, "TestFactor")
        self.assertTrue(report.has_leakage)
        self.assertEqual(report.mismatch_ratio, 0.2)
        
        # 测试字符串表示
        self.assertIn("TestFactor", repr(report))
        self.assertIn("20.00%", repr(report))
    
    def test_report_to_dict(self):
        """测试报告转字典"""
        report = LeakageReport(
            factor_name="TestFactor",
            has_leakage=False,
            split_ratio=0.5,
            short_data_periods=50,
            full_data_periods=100,
            overlap_periods=50,
            mismatched_periods=0,
            mismatch_ratio=0.0,
            max_absolute_diff=0.0,
            max_relative_diff=0.0,
            mismatched_dates=[],
            details={}
        )
        
        d = report.to_dict()
        self.assertEqual(d['factor_name'], "TestFactor")
        self.assertFalse(d['has_leakage'])
        self.assertEqual(d['mismatch_ratio'], 0.0)


class TestConvenienceFunction(unittest.TestCase):
    """测试便捷函数"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1']
        self.dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        close = pd.DataFrame(
            np.random.randn(1, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(1, 50)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_detect_leakage_function(self):
        """测试便捷函数 detect_leakage"""
        factor = CloseOverMA(period=10)
        
        # 使用便捷函数（verbose=False 避免打印输出）
        report = detect_leakage(factor, self.ohlcv, verbose=False)
        
        self.assertIsInstance(report, LeakageReport)
        self.assertFalse(report.has_leakage)
    
    def test_detect_leakage_with_custom_params(self):
        """测试便捷函数自定义参数"""
        factor = CloseOverMA(period=10)
        
        report = detect_leakage(
            factor, 
            self.ohlcv, 
            split_ratio=0.7,
            tolerance=1e-8,
            verbose=False
        )
        
        self.assertEqual(report.split_ratio, 0.7)
        self.assertEqual(report.short_data_periods, 35)  # 50 * 0.7


class TestBatchDetection(unittest.TestCase):
    """测试批量检测功能"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        close = pd.DataFrame(
            np.random.randn(2, 60).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 60)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
        
        self.factors = [
            CloseOverMA(period=10),
            RSI(period=14),
            Momentum(period=10),
        ]
    
    def test_detect_all_factors(self):
        """测试批量检测多个因子"""
        detector = LeakageDetector(split_ratio=0.5)
        reports = detector.detect_all_factors(self.factors, self.ohlcv)
        
        # 验证返回类型
        self.assertIsInstance(reports, dict)
        self.assertEqual(len(reports), 3)
        
        # 验证所有因子都被检测
        for factor in self.factors:
            self.assertIn(factor.name, reports)
        
        # 验证所有正常因子都无泄露
        for name, report in reports.items():
            self.assertFalse(report.has_leakage, f"{name} should not have leakage")


class TestValidateNoLeakage(unittest.TestCase):
    """测试 validate_no_leakage 方法"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1']
        self.dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        close = pd.DataFrame(
            np.random.randn(1, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(1, 50)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_validate_no_leakage_success(self):
        """测试验证通过的情况"""
        detector = LeakageDetector(split_ratio=0.5)
        factor = CloseOverMA(period=10)
        
        # 应该返回 True，不抛出异常
        result = detector.validate_no_leakage(factor, self.ohlcv, raise_error=True)
        self.assertTrue(result)
    
    def test_validate_no_leakage_with_exception(self):
        """测试验证失败时抛出异常"""
        detector = LeakageDetector(split_ratio=0.5)
        factor = LeakyFactor(look_ahead=5)
        
        # 应该抛出 FutureDataLeakageError
        with self.assertRaises(FutureDataLeakageError):
            detector.validate_no_leakage(factor, self.ohlcv, raise_error=True)
    
    def test_validate_no_leakage_without_exception(self):
        """测试验证失败时不抛出异常"""
        detector = LeakageDetector(split_ratio=0.5)
        factor = LeakyFactor(look_ahead=5)
        
        # 应该返回 False，不抛出异常
        result = detector.validate_no_leakage(factor, self.ohlcv, raise_error=False)
        self.assertFalse(result)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1']
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 创建很短的数据
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        close = pd.DataFrame(
            np.random.randn(1, 5).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=dates
        ).abs()
        
        ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(1, 5)) * 10000 + 1000,
                index=self.symbols,
                columns=dates
            )
        )
        
        # 使用很小的split_ratio会导致切割点小于1，应该抛出异常
        detector = LeakageDetector(split_ratio=0.1)
        
        factor = CloseOverMA(period=3)
        
        # 应该抛出 ValueError，因为 split_point 为 0
        with self.assertRaises(ValueError):
            detector.detect(factor, ohlcv)
        
        # 使用更大的split_ratio应该正常工作
        detector_ok = LeakageDetector(split_ratio=0.4)
        report = detector_ok.detect(factor, ohlcv)
        self.assertIsInstance(report, LeakageReport)
    
    def test_data_with_nans(self):
        """测试包含NaN的数据"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        close = pd.DataFrame(
            np.random.randn(1, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=dates
        ).abs()
        
        # 添加一些NaN
        close.iloc[0, 10:15] = np.nan
        
        ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(1, 50)) * 10000 + 1000,
                index=self.symbols,
                columns=dates
            )
        )
        
        detector = LeakageDetector(split_ratio=0.5)
        factor = CloseOverMA(period=10)
        
        # 应该能处理NaN，不抛出异常
        report = detector.detect(factor, ohlcv)
        self.assertIsInstance(report, LeakageReport)


class TestDifferentSplitRatios(unittest.TestCase):
    """测试不同切割比例下的检测效果"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        close = pd.DataFrame(
            np.random.randn(2, 100).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 100)) * 10000 + 1000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_different_split_ratios_no_leakage(self):
        """测试不同切割比例下的无泄露检测"""
        factor = CloseOverMA(period=20)
        
        for ratio in [0.3, 0.5, 0.7, 0.8]:
            detector = LeakageDetector(split_ratio=ratio)
            report = detector.detect(factor, self.ohlcv)
            
            self.assertFalse(
                report.has_leakage,
                f"CloseOverMA should not have leakage with split_ratio={ratio}"
            )
    
    def test_different_split_ratios_with_leakage(self):
        """测试不同切割比例下的泄露检测"""
        factor = LeakyFactor(look_ahead=5)
        
        for ratio in [0.3, 0.5, 0.7, 0.8]:
            detector = LeakageDetector(split_ratio=ratio)
            report = detector.detect(factor, self.ohlcv)
            
            self.assertTrue(
                report.has_leakage,
                f"LeakyFactor should be detected with split_ratio={ratio}"
            )


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
