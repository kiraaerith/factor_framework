"""
基础框架测试脚本

用于验证阶段一核心数据结构的正确性。
"""

import numpy as np
import pandas as pd

try:
    from .factor_data import FactorData
    from .position_data import PositionData
    from .ohlcv_data import OHLCVData
    from .base_interfaces import FactorCalculator, PositionMapper, Evaluator
except ImportError:
    from factor_data import FactorData
    from position_data import PositionData
    from ohlcv_data import OHLCVData
    from base_interfaces import FactorCalculator, PositionMapper, Evaluator


def test_factor_data():
    """测试 FactorData 类"""
    print("=" * 60)
    print("测试 FactorData 类")
    print("=" * 60)
    
    # 创建测试数据
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    values = pd.DataFrame(
        np.random.randn(5, 10),
        index=symbols,
        columns=dates
    )
    
    # 创建 FactorData
    factor = FactorData(values, name='TestFactor', params={'period': 20})
    
    print(f"FactorData 创建成功")
    print(f"  名称: {factor.name}")
    print(f"  形状: {factor.shape}")
    print(f"  标的数: {factor.n_assets}")
    print(f"  时间长度: {factor.n_periods}")
    print(f"  信息摘要: {factor.info()}")
    
    # 测试方法
    print("\n测试方法:")
    rank_factor = factor.rank()
    print(f"  rank() -> 形状: {rank_factor.shape}")
    
    zscore_factor = factor.zscore()
    print(f"  zscore() -> 形状: {zscore_factor.shape}")
    
    cs_data = factor.get_cross_section(dates[0])
    print(f"  get_cross_section() -> 类型: {type(cs_data)}, 长度: {len(cs_data)}")
    
    ts_data = factor.get_time_series('ETF1')
    print(f"  get_time_series() -> 类型: {type(ts_data)}, 长度: {len(ts_data)}")
    
    print("[OK] FactorData 测试通过\n")


def test_position_data():
    """测试 PositionData 类"""
    print("=" * 60)
    print("测试 PositionData 类")
    print("=" * 60)
    
    # 创建测试数据
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    # 创建权重矩阵（等权持有多头）
    weights_array = np.zeros((5, 10))
    for t in range(10):
        selected = np.random.choice(5, 3, replace=False)  # 随机选3个
        weights_array[selected, t] = 1.0 / 3  # 等权
    
    weights = pd.DataFrame(weights_array, index=symbols, columns=dates)
    
    # 创建 PositionData
    position = PositionData(weights, name='TestPosition', params={'k': 3})
    
    print(f"PositionData 创建成功")
    print(f"  名称: {position.name}")
    print(f"  形状: {position.shape}")
    print(f"  信息摘要: {position.info()}")
    
    # 测试方法
    print("\n测试方法:")
    total_weights = position.get_total_weights()
    print(f"  get_total_weights() -> 平均值: {total_weights.mean():.4f}")
    
    pos_count = position.get_position_count()
    print(f"  get_position_count() -> 平均值: {pos_count.mean():.1f}")
    
    normalized = position.normalize()
    print(f"  normalize() -> 总权重均值: {normalized.get_total_weights().mean():.4f}")
    
    shifted = position.shift(1)
    print(f"  shift(1) -> 形状: {shifted.shape}")
    
    print("[OK] PositionData 测试通过\n")


def test_ohlcv_data():
    """测试 OHLCVData 类"""
    print("=" * 60)
    print("测试 OHLCVData 类")
    print("=" * 60)
    
    # 创建测试数据
    symbols = ['ETF1', 'ETF2', 'ETF3']
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    # 模拟价格数据
    np.random.seed(42)
    close = pd.DataFrame(
        np.cumsum(np.random.randn(3, 10) * 0.01, axis=1) + 3.5,
        index=symbols,
        columns=dates
    )
    
    # 根据close生成OHLC
    open_price = close.shift(1, axis=1).fillna(close.iloc[:, 0])
    high = pd.concat([open_price, close], axis=0).groupby(level=0).max() * (1 + np.abs(np.random.randn(3, 10) * 0.005))
    low = pd.concat([open_price, close], axis=0).groupby(level=0).min() * (1 - np.abs(np.random.randn(3, 10) * 0.005))
    volume = pd.DataFrame(
        np.abs(np.random.randn(3, 10)) * 1000000,
        index=symbols,
        columns=dates
    )
    
    # 确保 high >= low
    high = pd.concat([high, low]).groupby(level=0).max()
    
    # 创建 OHLCVData
    ohlcv = OHLCVData(
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume
    )
    
    print(f"OHLCVData 创建成功")
    print(f"  形状: {ohlcv.shape}")
    print(f"  信息摘要: {ohlcv.info()}")
    
    # 测试方法
    print("\n测试方法:")
    returns = ohlcv.get_returns()
    print(f"  get_returns() -> 形状: {returns.shape}")
    
    log_returns = ohlcv.get_log_returns()
    print(f"  get_log_returns() -> 形状: {log_returns.shape}")
    
    vwap = ohlcv.get_vwap()
    print(f"  get_vwap() -> 形状: {vwap.shape}")
    
    cs_data = ohlcv.get_cross_section(dates[0])
    print(f"  get_cross_section() -> 形状: {cs_data.shape}")
    
    ts_data = ohlcv.get_time_series('ETF1')
    print(f"  get_time_series() -> 形状: {ts_data.shape}")
    
    print("[OK] OHLCVData 测试通过\n")


def test_base_interfaces():
    """测试抽象基类"""
    print("=" * 60)
    print("测试抽象基类")
    print("=" * 60)
    
    # 创建简单的具体实现来测试接口
    class SimpleFactorCalculator(FactorCalculator):
        def __init__(self):
            self._name = "SimpleMA"
            self.period = 5
        
        def calculate(self, ohlcv_data):
            close = ohlcv_data.close
            ma = close.rolling(self.period, axis=1).mean()
            factor_value = close / ma
            return FactorData(factor_value, name=self.name, params=self.get_params())
        
        @property
        def name(self):
            return self._name
        
        def get_params(self):
            return {'period': self.period}
    
    class SimplePositionMapper(PositionMapper):
        def __init__(self):
            self._name = "Top2"
            self.k = 2
        
        def map_to_position(self, factor_data):
            values = factor_data.values
            ranks = values.rank(axis=0, ascending=False)
            selected = (ranks <= self.k).astype(float)
            weights = selected.div(selected.sum(axis=0), axis=1).fillna(0)
            return PositionData(weights, name=self.name)
        
        @property
        def name(self):
            return self._name
    
    class SimpleEvaluator(Evaluator):
        def __init__(self):
            self._name = "SimpleIC"
        
        def evaluate(self, factor_data, ohlcv_data, position_data=None):
            # 简化的IC计算
            return {'IC': 0.05, 'ICIR': 0.3}
        
        @property
        def metrics(self):
            return ['IC', 'ICIR']
        
        @property
        def name(self):
            return self._name
    
    # 测试数据
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    np.random.seed(42)
    close = pd.DataFrame(
        np.cumsum(np.random.randn(5, 10) * 0.01, axis=1) + 3.5,
        index=symbols,
        columns=dates
    )
    open_price = close.shift(1, axis=1).fillna(close.iloc[:, 0])
    high = close * 1.01
    low = close * 0.99
    volume = pd.DataFrame(np.abs(np.random.randn(5, 10)) * 1000000, index=symbols, columns=dates)
    
    ohlcv = OHLCVData(open=open_price, high=high, low=low, close=close, volume=volume)
    
    # 测试 FactorCalculator
    calc = SimpleFactorCalculator()
    print(f"FactorCalculator: {calc}")
    factor = calc(ohlcv)
    print(f"  计算结果: {factor}")
    
    # 测试 PositionMapper
    mapper = SimplePositionMapper()
    print(f"\nPositionMapper: {mapper}")
    position = mapper(factor)
    print(f"  映射结果: {position}")
    
    # 测试 Evaluator
    evaluator = SimpleEvaluator()
    print(f"\nEvaluator: {evaluator}")
    metrics = evaluator(factor, ohlcv, position)
    print(f"  评估结果: {metrics}")
    
    print("[OK] 抽象基类测试通过\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ETF因子框架 - 阶段一测试")
    print("=" * 60 + "\n")
    
    try:
        test_factor_data()
        test_position_data()
        test_ohlcv_data()
        test_base_interfaces()
        
        print("=" * 60)
        print("[SUCCESS] 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
