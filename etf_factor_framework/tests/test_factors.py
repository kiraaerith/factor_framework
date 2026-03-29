"""
因子计算器单元测试模块

测试各类因子计算器的：
1. 输入输出形状验证
2. 参数验证
3. 因子值范围验证
4. 边界条件处理
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
from factors.ohlcv_calculator import OHLCVFactorCalculator
from factors.technical_factors import (
    CloseOverMA, RSI, Momentum, MACD, BollingerBands,
    get_factor_class, list_available_factors, create_factor
)


class TestOHLCVFactorCalculator(unittest.TestCase):
    """测试OHLCV因子计算器基类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2', 'ETF3']
        self.dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 生成模拟价格数据（带趋势）
        close = pd.DataFrame(
            np.random.randn(3, 100).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()  # 确保价格为正
        
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
    
    def test_utility_methods(self):
        """测试工具方法"""
        # 创建测试用的因子计算器
        class TestCalculator(OHLCVFactorCalculator):
            def calculate(self, ohlcv_data):
                return FactorData(ohlcv_data.close, name='test')
            
            @property
            def name(self):
                return 'Test'
            
            @property
            def params(self):
                return {}
        
        calc = TestCalculator()
        
        # 测试滚动均值
        data = self.ohlcv.close
        ma = calc._rolling_mean(data, window=20)
        self.assertEqual(ma.shape, data.shape)
        
        # 测试滚动标准差
        std = calc._rolling_std(data, window=20)
        self.assertEqual(std.shape, data.shape)
        
        # 测试EMA
        ema = calc._ema(data, span=12)
        self.assertEqual(ema.shape, data.shape)
        
        # 测试时间平移
        shifted = calc._shift(data, periods=1)
        self.assertEqual(shifted.shape, data.shape)
        
        # 测试百分比变化
        pct = calc._pct_change(data, periods=5)
        self.assertEqual(pct.shape, data.shape)
        
        # 测试横截面排名
        rank = calc._cross_sectional_rank(data)
        self.assertEqual(rank.shape, data.shape)
        
        # 测试横截面Z-Score
        zscore = calc._cross_sectional_zscore(data)
        self.assertEqual(zscore.shape, data.shape)


class TestCloseOverMA(unittest.TestCase):
    """测试CloseOverMA因子"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=50)
        
        close = pd.DataFrame(
            np.random.randn(2, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 50)) * 10000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_output_shape(self):
        """测试输出形状"""
        factor = CloseOverMA(period=20)
        result = factor.calculate(self.ohlcv)
        
        # 验证输出类型
        self.assertIsInstance(result, FactorData)
        
        # 验证输出形状 (N × T)
        self.assertEqual(result.shape, (2, 50))
        
        # 验证索引和列
        self.assertEqual(list(result.symbols), self.symbols)
        self.assertEqual(list(result.dates), list(self.dates))
    
    def test_factor_name_and_params(self):
        """测试因子名称和参数"""
        factor = CloseOverMA(period=20, field='close')
        
        self.assertEqual(factor.name, 'CloseOverMA_20')
        self.assertEqual(factor.params, {'period': 20, 'field': 'close'})
    
    def test_invalid_params(self):
        """测试无效参数"""
        # 无效周期
        with self.assertRaises(ValueError):
            CloseOverMA(period=0)
        
        with self.assertRaises(ValueError):
            CloseOverMA(period=-5)
        
        # 无效字段
        with self.assertRaises(ValueError):
            CloseOverMA(period=20, field='invalid')
    
    def test_factor_values(self):
        """测试因子值范围"""
        factor = CloseOverMA(period=10)
        result = factor.calculate(self.ohlcv)
        
        values = result.values
        
        # 因子值应该是正数（价格/均线）
        self.assertTrue((values > 0).all().all())
        
        # 前期应该是NaN或有限值（因为窗口不足）
        # 使用min_periods=1，所以不会有NaN


class TestRSI(unittest.TestCase):
    """测试RSI因子"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=50)
        
        close = pd.DataFrame(
            np.random.randn(2, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 50)) * 10000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_output_shape(self):
        """测试输出形状"""
        factor = RSI(period=14)
        result = factor.calculate(self.ohlcv)
        
        # 验证输出类型
        self.assertIsInstance(result, FactorData)
        
        # 验证输出形状 (N × T)
        self.assertEqual(result.shape, (2, 50))
    
    def test_factor_values_range(self):
        """测试RSI值范围在0-100之间"""
        factor = RSI(period=14)
        result = factor.calculate(self.ohlcv)
        
        values = result.values
        
        # RSI应该在0-100范围内
        self.assertTrue((values >= 0).all().all())
        self.assertTrue((values <= 100).all().all())
    
    def test_invalid_params(self):
        """测试无效参数"""
        with self.assertRaises(ValueError):
            RSI(period=0)
        
        with self.assertRaises(ValueError):
            RSI(period=-5)
        
        with self.assertRaises(ValueError):
            RSI(period=14, field='invalid')


class TestMomentum(unittest.TestCase):
    """测试Momentum因子"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=50)
        
        close = pd.DataFrame(
            np.random.randn(2, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 50)) * 10000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_output_shape(self):
        """测试输出形状"""
        factor = Momentum(period=20)
        result = factor.calculate(self.ohlcv)
        
        # 验证输出类型
        self.assertIsInstance(result, FactorData)
        
        # 验证输出形状 (N × T)
        self.assertEqual(result.shape, (2, 50))
    
    def test_simple_return(self):
        """测试简单收益率计算"""
        factor = Momentum(period=10, log_return=False)
        result = factor.calculate(self.ohlcv)
        
        # 验证形状
        self.assertEqual(result.shape, (2, 50))
    
    def test_log_return(self):
        """测试对数收益率计算"""
        factor = Momentum(period=10, log_return=True)
        result = factor.calculate(self.ohlcv)
        
        # 验证形状
        self.assertEqual(result.shape, (2, 50))
    
    def test_invalid_params(self):
        """测试无效参数"""
        with self.assertRaises(ValueError):
            Momentum(period=0)
        
        with self.assertRaises(ValueError):
            Momentum(period=-5)


class TestMACD(unittest.TestCase):
    """测试MACD因子"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=60)
        
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
                np.abs(np.random.randn(2, 60)) * 10000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_output_shape(self):
        """测试输出形状"""
        factor = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = factor.calculate(self.ohlcv)
        
        # 验证输出类型
        self.assertIsInstance(result, FactorData)
        
        # 验证输出形状 (N × T)
        self.assertEqual(result.shape, (2, 60))
    
    def test_invalid_params(self):
        """测试无效参数"""
        # fast_period >= slow_period
        with self.assertRaises(ValueError):
            MACD(fast_period=20, slow_period=10)
        
        # 零或负周期
        with self.assertRaises(ValueError):
            MACD(fast_period=0, slow_period=26)
        
        with self.assertRaises(ValueError):
            MACD(fast_period=12, slow_period=-26)


class TestBollingerBands(unittest.TestCase):
    """测试BollingerBands因子"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.symbols = ['ETF1', 'ETF2']
        self.dates = pd.date_range('2024-01-01', periods=50)
        
        close = pd.DataFrame(
            np.random.randn(2, 50).cumsum(axis=1) + 100,
            index=self.symbols,
            columns=self.dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 50)) * 10000,
                index=self.symbols,
                columns=self.dates
            )
        )
    
    def test_output_shape(self):
        """测试输出形状"""
        factor = BollingerBands(period=20, std_multiplier=2)
        result = factor.calculate(self.ohlcv)
        
        # 验证输出类型
        self.assertIsInstance(result, FactorData)
        
        # 验证输出形状 (N × T)
        self.assertEqual(result.shape, (2, 50))
    
    def test_factor_values_range(self):
        """测试因子值范围在0-1之间"""
        factor = BollingerBands(period=20, std_multiplier=2)
        result = factor.calculate(self.ohlcv)
        
        values = result.values
        
        # 布林带位置应该在0-1范围内
        self.assertTrue((values >= 0).all().all())
        self.assertTrue((values <= 1).all().all())
    
    def test_invalid_params(self):
        """测试无效参数"""
        with self.assertRaises(ValueError):
            BollingerBands(period=0)
        
        with self.assertRaises(ValueError):
            BollingerBands(std_multiplier=-2)


class TestFactorRegistry(unittest.TestCase):
    """测试因子注册表"""
    
    def test_list_available_factors(self):
        """测试列出可用因子"""
        factors = list_available_factors()
        self.assertIn('CloseOverMA', factors)
        self.assertIn('RSI', factors)
        self.assertIn('Momentum', factors)
        self.assertIn('MACD', factors)
        self.assertIn('BollingerBands', factors)
    
    def test_get_factor_class(self):
        """测试获取因子类"""
        self.assertEqual(get_factor_class('RSI'), RSI)
        self.assertEqual(get_factor_class('Momentum'), Momentum)
        
        with self.assertRaises(ValueError):
            get_factor_class('UnknownFactor')
    
    def test_create_factor(self):
        """测试创建因子实例"""
        factor = create_factor('RSI', period=14)
        self.assertIsInstance(factor, RSI)
        self.assertEqual(factor.params['period'], 14)


class TestFactorWithSmallData(unittest.TestCase):
    """使用小规模数据测试边界条件"""
    
    def test_short_time_series(self):
        """测试短时间序列"""
        np.random.seed(42)
        symbols = ['ETF1']
        dates = pd.date_range('2024-01-01', periods=10)
        
        close = pd.DataFrame(
            np.random.randn(1, 10).cumsum(axis=1) + 100,
            index=symbols,
            columns=dates
        ).abs()
        
        ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(1, 10)) * 10000,
                index=symbols,
                columns=dates
            )
        )
        
        # 使用比数据长度更长的周期
        factor = CloseOverMA(period=20)
        result = factor.calculate(ohlcv)
        
        # 仍然应该有输出
        self.assertEqual(result.shape, (1, 10))


class TestFactorCall(unittest.TestCase):
    """测试因子实例的可调用性"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        symbols = ['ETF1', 'ETF2']
        dates = pd.date_range('2024-01-01', periods=30)
        
        close = pd.DataFrame(
            np.random.randn(2, 30).cumsum(axis=1) + 100,
            index=symbols,
            columns=dates
        ).abs()
        
        self.ohlcv = OHLCVData(
            open=close * 0.99,
            high=close * 1.02,
            low=close * 0.98,
            close=close,
            volume=pd.DataFrame(
                np.abs(np.random.randn(2, 30)) * 10000,
                index=symbols,
                columns=dates
            )
        )
    
    def test_callable(self):
        """测试因子实例可以直接调用"""
        factor = CloseOverMA(period=10)
        
        # 使用 __call__ 方法
        result1 = factor(self.ohlcv)
        
        # 使用 calculate 方法
        result2 = factor.calculate(self.ohlcv)
        
        # 结果应该相同
        pd.testing.assert_frame_equal(result1.values, result2.values)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
