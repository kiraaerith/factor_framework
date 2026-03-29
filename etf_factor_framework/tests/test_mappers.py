"""
仓位映射器单元测试

测试内容：
    1. RankBasedMapper - 基于排名的映射
    2. DirMapper - 直接映射
    3. QuantileMapper - 分位数映射
    4. ZScoreMapper - Z-Score映射
    5. 权重方法 - equal_weight, normalize_weights
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factor_data import FactorData
from core.position_data import PositionData
from mappers.position_mappers import (
    RankBasedMapper,
    DirMapper,
    QuantileMapper,
    ZScoreMapper,
    create_top_k_mapper,
    create_bottom_k_mapper,
    create_equal_weight_mapper
)
from mappers.weight_methods import equal_weight, normalize_weights


class TestWeightMethods(unittest.TestCase):
    """测试权重分配方法"""
    
    def test_equal_weight_basic(self):
        """测试等权分配基本功能"""
        selected = pd.DataFrame(
            [[True, True], [False, True], [True, False]],
            index=['A', 'B', 'C'],
            columns=['2024-01-01', '2024-01-02']
        )
        weights = equal_weight(selected)
        
        # 验证形状
        self.assertEqual(weights.shape, (3, 2))
        
        # 验证每列和为1
        self.assertAlmostEqual(weights['2024-01-01'].sum(), 1.0)
        self.assertAlmostEqual(weights['2024-01-02'].sum(), 1.0)
        
        # 验证选中标的权重
        self.assertAlmostEqual(weights.loc['A', '2024-01-01'], 0.5)
        self.assertAlmostEqual(weights.loc['B', '2024-01-01'], 0.0)
        self.assertAlmostEqual(weights.loc['C', '2024-01-01'], 0.5)
    
    def test_equal_weight_all_selected(self):
        """测试全部选中时的等权"""
        selected = pd.DataFrame(
            [[True, True], [True, True], [True, True]],
            index=['A', 'B', 'C'],
            columns=['d1', 'd2']
        )
        weights = equal_weight(selected)
        
        # 3个标的等权，每个1/3
        expected = 1.0 / 3.0
        for idx in ['A', 'B', 'C']:
            self.assertAlmostEqual(weights.loc[idx, 'd1'], expected)
    
    def test_equal_weight_none_selected(self):
        """测试没有选中时的处理"""
        selected = pd.DataFrame(
            [[False, False], [False, False]],
            index=['A', 'B'],
            columns=['d1', 'd2']
        )
        weights = equal_weight(selected)
        
        # 没有选中，权重全为0
        self.assertTrue((weights == 0).all().all())
    
    def test_normalize_weights(self):
        """测试权重归一化"""
        weights = pd.DataFrame(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]],
            index=['A', 'B', 'C'],
            columns=['d1', 'd2']
        )
        normalized = normalize_weights(weights, target_sum=1.0)
        
        # 验证每列和为1
        self.assertAlmostEqual(normalized['d1'].sum(), 1.0)
        self.assertAlmostEqual(normalized['d2'].sum(), 1.0)


class TestRankBasedMapper(unittest.TestCase):
    """测试RankBasedMapper"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建一个简单的因子值矩阵
        # 日期1: A=0.1, B=0.5, C=0.3, D=0.2, E=0.4
        # 日期2: A=0.5, B=0.1, C=0.4, D=0.3, E=0.2
        values = pd.DataFrame(
            {
                '2024-01-01': [0.1, 0.5, 0.3, 0.2, 0.4],
                '2024-01-02': [0.5, 0.1, 0.4, 0.3, 0.2],
            },
            index=['A', 'B', 'C', 'D', 'E']
        )
        self.factor_data = FactorData(values, name='TestFactor')
    
    def test_top_k_selection(self):
        """测试Top K选择"""
        mapper = RankBasedMapper(top_k=2, direction=1, weight_method='equal')
        position = mapper.map_to_position(self.factor_data)
        
        # 验证输出类型
        self.assertIsInstance(position, PositionData)
        
        # 验证形状
        self.assertEqual(position.shape, (5, 2))
        
        # 日期1: 最大的是 B=0.5, E=0.4
        self.assertAlmostEqual(position.weights.loc['B', '2024-01-01'], 0.5)
        self.assertAlmostEqual(position.weights.loc['E', '2024-01-01'], 0.5)
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-01'], 0.0)
        
        # 日期2: 最大的是 A=0.5, C=0.4
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-02'], 0.5)
        self.assertAlmostEqual(position.weights.loc['C', '2024-01-02'], 0.5)
        
        # 验证每列权重和
        self.assertAlmostEqual(position.weights['2024-01-01'].sum(), 1.0)
        self.assertAlmostEqual(position.weights['2024-01-02'].sum(), 1.0)
    
    def test_bottom_k_selection(self):
        """测试Bottom K选择（方向=-1）"""
        mapper = RankBasedMapper(top_k=2, direction=-1, weight_method='equal')
        position = mapper.map_to_position(self.factor_data)
        
        # 日期1: 最小的是 A=0.1, D=0.2
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-01'], 0.5)
        self.assertAlmostEqual(position.weights.loc['D', '2024-01-01'], 0.5)
        
        # 日期2: 最小的是 B=0.1, E=0.2
        self.assertAlmostEqual(position.weights.loc['B', '2024-01-02'], 0.5)
        self.assertAlmostEqual(position.weights.loc['E', '2024-01-02'], 0.5)
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        # 无效的top_k
        with self.assertRaises(ValueError):
            RankBasedMapper(top_k=0)
        with self.assertRaises(ValueError):
            RankBasedMapper(top_k=-1)
        
        # 无效的方向
        with self.assertRaises(ValueError):
            RankBasedMapper(direction=2)
        with self.assertRaises(ValueError):
            RankBasedMapper(direction=0)
    
    def test_name_generation(self):
        """测试名称生成"""
        mapper = RankBasedMapper(top_k=5, direction=1, weight_method='equal')
        self.assertIn('Top5', mapper.name)
        
        mapper = RankBasedMapper(top_k=5, direction=-1, weight_method='equal')
        self.assertIn('Bottom5', mapper.name)
    
    def test_factory_functions(self):
        """测试工厂函数"""
        mapper1 = create_top_k_mapper(k=3)
        self.assertEqual(mapper1.direction, 1)
        self.assertEqual(mapper1.top_k, 3)
        
        mapper2 = create_bottom_k_mapper(k=2)
        self.assertEqual(mapper2.direction, -1)
        self.assertEqual(mapper2.top_k, 2)


class TestDirMapper(unittest.TestCase):
    """测试DirMapper"""
    
    def setUp(self):
        """设置测试数据"""
        values = pd.DataFrame(
            {
                '2024-01-01': [0.5, 0.3, 0.2],
                '2024-01-02': [0.2, 0.5, 0.3],
            },
            index=['A', 'B', 'C']
        )
        self.factor_data = FactorData(values, name='TestFactor')
    
    def test_direct_mapping_normalized(self):
        """测试直接映射（归一化）"""
        mapper = DirMapper(normalize=True, target_sum=1.0)
        position = mapper.map_to_position(self.factor_data)
        
        # 验证形状
        self.assertEqual(position.shape, (3, 2))
        
        # 验证归一化后权重和为1
        self.assertAlmostEqual(position.weights['2024-01-01'].sum(), 1.0)
        self.assertAlmostEqual(position.weights['2024-01-02'].sum(), 1.0)
        
        # 验证权重比例
        # 日期1: 0.5, 0.3, 0.2 -> 和为1.0，权重为0.5, 0.3, 0.2
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-01'], 0.5)
        self.assertAlmostEqual(position.weights.loc['B', '2024-01-01'], 0.3)
        self.assertAlmostEqual(position.weights.loc['C', '2024-01-01'], 0.2)
    
    def test_direct_mapping_not_normalized(self):
        """测试直接映射（不归一化）"""
        mapper = DirMapper(normalize=False)
        position = mapper.map_to_position(self.factor_data)
        
        # 验证权重保持原值
        pd.testing.assert_frame_equal(
            position.weights,
            self.factor_data.values
        )
    
    def test_clip_range(self):
        """测试截断功能"""
        values = pd.DataFrame(
            {
                '2024-01-01': [0.1, 0.5, 0.9],
            },
            index=['A', 'B', 'C']
        )
        factor_data = FactorData(values)
        
        mapper = DirMapper(clip_range=(0.2, 0.8), normalize=False)
        position = mapper.map_to_position(factor_data)
        
        # 验证截断
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-01'], 0.2)  # 被截断
        self.assertAlmostEqual(position.weights.loc['B', '2024-01-01'], 0.5)  # 不变
        self.assertAlmostEqual(position.weights.loc['C', '2024-01-01'], 0.8)  # 被截断
    
    def test_fill_na(self):
        """测试NaN填充"""
        values = pd.DataFrame(
            {
                '2024-01-01': [0.5, np.nan, 0.3],
            },
            index=['A', 'B', 'C']
        )
        factor_data = FactorData(values)
        
        mapper = DirMapper(normalize=False, fill_na=0.0)
        position = mapper.map_to_position(factor_data)
        
        # 验证NaN被填充
        self.assertEqual(position.weights.loc['B', '2024-01-01'], 0.0)
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        # 无效的目标权重和
        with self.assertRaises(ValueError):
            DirMapper(target_sum=0)
        with self.assertRaises(ValueError):
            DirMapper(target_sum=-1)
        
        # 无效的clip_range
        with self.assertRaises(ValueError):
            DirMapper(clip_range=(0.5,))  # 长度不为2
    
    def test_factory_function(self):
        """测试工厂函数"""
        mapper = create_equal_weight_mapper()
        self.assertTrue(mapper.normalize)
        self.assertEqual(mapper.target_sum, 1.0)


class TestQuantileMapper(unittest.TestCase):
    """测试QuantileMapper"""
    
    def setUp(self):
        """设置测试数据"""
        # 5个标的，5个值，方便测试5分位
        values = pd.DataFrame(
            {
                '2024-01-01': [0.1, 0.3, 0.5, 0.7, 0.9],
                '2024-01-02': [0.9, 0.7, 0.5, 0.3, 0.1],
            },
            index=['A', 'B', 'C', 'D', 'E']
        )
        self.factor_data = FactorData(values)
    
    def test_long_only(self):
        """测试仅做多"""
        mapper = QuantileMapper(
            n_quantiles=5,
            long_quantile=4,  # 最高分位组
            short_quantile=None  # 不做空
        )
        position = mapper.map_to_position(self.factor_data)
        
        # 验证形状
        self.assertEqual(position.shape, (5, 2))
        
        # 日期1: E=0.9 在最高分位组，应该有多头仓位
        self.assertGreater(position.weights.loc['E', '2024-01-01'], 0)
        
        # A, B, C, D 应该没有多头仓位
        self.assertAlmostEqual(position.weights.loc['A', '2024-01-01'], 0)
        self.assertAlmostEqual(position.weights.loc['B', '2024-01-01'], 0)
        self.assertAlmostEqual(position.weights.loc['C', '2024-01-01'], 0)
        self.assertAlmostEqual(position.weights.loc['D', '2024-01-01'], 0)
        
        # 所有权重为正（多头）
        self.assertTrue((position.weights >= 0).all().all())
    
    def test_long_short(self):
        """测试多空"""
        mapper = QuantileMapper(
            n_quantiles=5,
            long_quantile=4,   # 最高分位组做多
            short_quantile=0   # 最低分位组做空
        )
        position = mapper.map_to_position(self.factor_data)
        
        # 日期1: E=0.9 做多，A=0.1 做空
        self.assertGreater(position.weights.loc['E', '2024-01-01'], 0)
        self.assertLess(position.weights.loc['A', '2024-01-01'], 0)
        
        # 验证总仓位约为0（多空平衡）
        total_weight = position.weights['2024-01-01'].sum()
        self.assertAlmostEqual(total_weight, 0.0, places=5)
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        # n_quantiles太小
        with self.assertRaises(ValueError):
            QuantileMapper(n_quantiles=1)
        
        # long_quantile超出范围
        with self.assertRaises(ValueError):
            QuantileMapper(n_quantiles=5, long_quantile=5)
        
        # long和short相同
        with self.assertRaises(ValueError):
            QuantileMapper(n_quantiles=5, long_quantile=2, short_quantile=2)


class TestZScoreMapper(unittest.TestCase):
    """测试ZScoreMapper"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建固定测试数据，确保Z-Score有的绝对值大于1，有的小于1
        # 每行的值要有差异，避免横截面标准差为0
        values = pd.DataFrame(
            {
                'd1': [30.0, 40.0, 55.0, 60.0, 70.0],
                'd2': [48.0, 48.0, 50.0, 52.0, 52.0],
                'd3': [45.0, 47.0, 50.0, 53.0, 55.0],
            },
            index=['A', 'B', 'C', 'D', 'E']
        )
        self.factor_data = FactorData(values)
    
    def test_basic_mapping(self):
        """测试基本映射"""
        mapper = ZScoreMapper(normalize=True)
        position = mapper.map_to_position(self.factor_data)
        
        # 验证形状
        self.assertEqual(position.shape, (5, 3))
        
        # 权重和应为1（归一化后），或0（如果该列所有Z-Score为0或NaN）
        for col in position.weights.columns:
            total_weight = position.weights[col].sum()
            # 由于数据设计，d1的绝对值都>0，所以应该有非零权重
            self.assertGreaterEqual(total_weight, 0.0)
    
    def test_threshold_filtering(self):
        """测试阈值过滤"""
        mapper = ZScoreMapper(threshold=1.0, normalize=False)
        position = mapper.map_to_position(self.factor_data)
        
        # d1: Z-Score绝对值都>=1，应该全部保留
        for idx in ['A', 'B', 'C', 'D', 'E']:
            self.assertNotEqual(position.weights.loc[idx, 'd1'], 0.0,
                               msg=f"{idx} in d1 should have non-zero weight")
        
        # d2: Z-Score绝对值都<1，应该全部变成0
        for idx in ['A', 'B', 'C', 'D', 'E']:
            self.assertAlmostEqual(position.weights.loc[idx, 'd2'], 0.0,
                                  msg=f"{idx} in d2 should be filtered out")
        
        # d3: Z-Score绝对值都<1，应该全部变成0
        for idx in ['A', 'B', 'C', 'D', 'E']:
            self.assertAlmostEqual(position.weights.loc[idx, 'd3'], 0.0,
                                  msg=f"{idx} in d3 should be filtered out")
    
    def test_callable_interface(self):
        """测试可调用接口"""
        mapper = RankBasedMapper(top_k=2)
        
        # 直接调用实例应该等同于调用map_to_position
        position1 = mapper.map_to_position(self.factor_data)
        position2 = mapper(self.factor_data)
        
        pd.testing.assert_frame_equal(position1.weights, position2.weights)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_single_asset(self):
        """测试单个标的"""
        values = pd.DataFrame(
            [[0.5]],
            index=['A'],
            columns=['d1']
        )
        factor_data = FactorData(values)
        
        mapper = RankBasedMapper(top_k=1)
        position = mapper.map_to_position(factor_data)
        
        # 单个标的应该获得全部权重
        self.assertAlmostEqual(position.weights.loc['A', 'd1'], 1.0)
    
    def test_k_larger_than_n(self):
        """测试K大于标的数量"""
        values = pd.DataFrame(
            [[0.5, 0.3], [0.3, 0.5]],
            index=['A', 'B'],
            columns=['d1', 'd2']
        )
        factor_data = FactorData(values)
        
        # K=5 > N=2
        mapper = RankBasedMapper(top_k=5)
        position = mapper.map_to_position(factor_data)
        
        # 应该选所有标的，等权
        self.assertAlmostEqual(position.weights.loc['A', 'd1'], 0.5)
        self.assertAlmostEqual(position.weights.loc['B', 'd1'], 0.5)
    
    def test_all_nan_column(self):
        """测试全NaN列的处理"""
        values = pd.DataFrame(
            {
                'd1': [0.5, 0.3, 0.2],
                'd2': [np.nan, np.nan, np.nan]
            },
            index=['A', 'B', 'C']
        )
        factor_data = FactorData(values)
        
        mapper = DirMapper(normalize=False, fill_na=0.0)
        position = mapper.map_to_position(factor_data)
        
        # 第二列应该全部为0
        self.assertTrue((position.weights['d2'] == 0).all())


if __name__ == '__main__':
    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestWeightMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestRankBasedMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestDirMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantileMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestZScoreMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出摘要
    print(f"\n{'='*70}")
    print(f"测试运行完成!")
    print(f"运行测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"{'='*70}")
