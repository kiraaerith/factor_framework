"""
阶段4评估模块使用示例

展示如何使用评估模块对因子进行全面评估。
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.factor_data import FactorData
from core.position_data import PositionData
from core.ohlcv_data import OHLCVData
from factors.technical_factors import CloseOverMA, Momentum
from mappers.position_mappers import RankBasedMapper
from evaluation import FactorEvaluator


def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    
    # 标的和时间
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # 生成价格数据（随机游走）
    close_prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(5, 252) * 0.02, axis=1) * 10,
        index=symbols,
        columns=dates
    )
    
    # 生成OHLCV数据
    open_prices = close_prices * (1 + np.random.randn(5, 252) * 0.005)
    
    # High >= max(open, close)
    high_prices = pd.concat([open_prices, close_prices]).groupby(level=0).max()
    high_prices = high_prices * (1 + np.abs(np.random.randn(5, 252) * 0.01))
    
    # Low <= min(open, close)
    low_prices = pd.concat([open_prices, close_prices]).groupby(level=0).min()
    low_prices = low_prices * (1 - np.abs(np.random.randn(5, 252) * 0.01))
    
    volume = pd.DataFrame(
        np.abs(np.random.randn(5, 252)) * 1000000,
        index=symbols,
        columns=dates
    )
    
    ohlcv = OHLCVData(
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume
    )
    
    return ohlcv


def example_basic_evaluation():
    """基础评估示例"""
    print("=" * 60)
    print("示例1: 基础因子评估")
    print("=" * 60)
    
    # 生成数据
    ohlcv = generate_sample_data()
    
    # 计算因子
    factor_calc = CloseOverMA(period=20)
    factor_data = factor_calc.calculate(ohlcv)
    
    print(f"\n因子名称: {factor_data.name}")
    print(f"因子形状: {factor_data.shape}")
    
    # 仓位映射（Top 3等权）
    mapper = RankBasedMapper(top_k=3, weight_method='equal', direction=-1)
    position_data = mapper.map_to_position(factor_data)
    
    # 创建评估器
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=5,  # 5日前瞻收益
        periods_per_year=252,
        risk_free_rate=0.03,
        commission_rate=0.0002,
        slippage_rate=0.0,
        delay=1
    )
    
    # 运行完整评估
    results = evaluator.run_full_evaluation()
    
    # 打印报告
    print("\n" + "-" * 40)
    print("IC类指标")
    print("-" * 40)
    ic = results['ic_metrics']
    print(f"IC Mean:       {ic.get('ic_mean', np.nan):.4f}")
    print(f"IC IR:         {ic.get('ic_ir', np.nan):.4f}")
    print(f"Rank IC Mean:  {ic.get('rank_ic_mean', np.nan):.4f}")
    print(f"Rank IC IR:    {ic.get('rank_ic_ir', np.nan):.4f}")
    
    print("\n" + "-" * 40)
    print("收益类指标")
    print("-" * 40)
    ret = results['returns_metrics']
    print(f"总收益:        {ret.get('total_return', np.nan):.2%}")
    print(f"年化收益:      {ret.get('annualized_return', np.nan):.2%}")
    
    print("\n" + "-" * 40)
    print("风险类指标")
    print("-" * 40)
    risk = results['risk_metrics']
    print(f"最大回撤:      {risk.get('max_drawdown', np.nan):.2%}")
    print(f"年化波动率:    {risk.get('annualized_volatility', np.nan):.2%}")
    
    print("\n" + "-" * 40)
    print("风险调整指标")
    print("-" * 40)
    adj = results['risk_adjusted_metrics']
    print(f"夏普比率:      {adj.get('sharpe_ratio', np.nan):.4f}")
    print(f"卡玛比率:      {adj.get('calmar_ratio', np.nan):.4f}")
    print(f"索提诺比率:    {adj.get('sortino_ratio', np.nan):.4f}")
    
    print("\n" + "-" * 40)
    print("换手率指标")
    print("-" * 40)
    to = results['turnover_metrics']
    print(f"日均换手率:    {to.get('avg_daily_turnover', np.nan):.2%}")
    print(f"年化换手率:    {to.get('annualized_turnover', np.nan):.2%}")
    
    return evaluator


def example_generate_report():
    """生成报告示例"""
    print("\n" + "=" * 60)
    print("示例2: 生成文本报告")
    print("=" * 60)
    
    # 生成数据
    ohlcv = generate_sample_data()
    
    # 计算因子和仓位
    factor_calc = Momentum(period=20, log_return=False)
    factor_data = factor_calc.calculate(ohlcv)
    
    mapper = RankBasedMapper(top_k=3, weight_method='equal', direction=1)
    position_data = mapper.map_to_position(factor_data)
    
    # 创建评估器
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=5,
    )
    
    # 生成并打印报告
    report = evaluator.generate_report()
    print(report)
    
    return evaluator


def example_visualization():
    """可视化示例"""
    print("\n" + "=" * 60)
    print("示例3: 生成可视化图表")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    # 生成数据
    ohlcv = generate_sample_data()
    
    # 计算因子和仓位
    factor_calc = CloseOverMA(period=20)
    factor_data = factor_calc.calculate(ohlcv)
    
    mapper = RankBasedMapper(top_k=3, weight_method='equal')
    position_data = mapper.map_to_position(factor_data)
    
    # 创建评估器
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=5,
    )
    
    # 生成图表
    output_dir = Path(__file__).parent / 'evaluation_output'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n生成图表到目录: {output_dir}")
    figures = evaluator.plot(output_dir=str(output_dir))
    
    print(f"生成图表数量: {len(figures)}")
    for name, fig in figures.items():
        print(f"  - {name}")
        plt.close(fig)  # 关闭图表释放内存
    
    print(f"\n图表已保存到: {output_dir}")
    
    return evaluator


def example_individual_metrics():
    """单独使用指标函数示例"""
    print("\n" + "=" * 60)
    print("示例4: 单独使用指标函数")
    print("=" * 60)
    
    from evaluation.metrics import (
        total_return, annualized_return, max_drawdown,
        sharpe_ratio, calmar_ratio, calculate_ic
    )
    
    # 创建示例收益序列
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    print(f"\n收益序列示例（前5个）: {returns.head().values}")
    print(f"总收益:     {total_return(returns):.4f}")
    print(f"年化收益:   {annualized_return(returns, periods_per_year=252):.4f}")
    print(f"最大回撤:   {max_drawdown(returns):.4f}")
    print(f"夏普比率:   {sharpe_ratio(returns, periods_per_year=252):.4f}")
    print(f"卡玛比率:   {calmar_ratio(returns, periods_per_year=252):.4f}")
    
    # IC计算示例
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    forward_ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = calculate_ic(factor, forward_ret)
    print(f"\nIC示例（完美相关）: {ic:.4f}")


def example_ic_analysis():
    """IC分析示例"""
    print("\n" + "=" * 60)
    print("示例5: IC序列分析")
    print("=" * 60)
    
    from evaluation.metrics import (
        ICMetricsCalculator, calculate_forward_returns
    )
    
    # 生成数据
    np.random.seed(42)
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 生成因子值（有预测能力）
    factor_values = pd.DataFrame(
        np.random.randn(5, 100),
        index=symbols,
        columns=dates
    )
    
    # 生成价格数据
    close_prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(5, 100) * 0.02, axis=1),
        index=symbols,
        columns=dates
    )
    
    # 计算前瞻收益
    forward_returns = calculate_forward_returns(close_prices, periods=5)
    
    # IC计算器
    calc = ICMetricsCalculator(factor_values, forward_returns, periods_per_year=252)
    
    print(f"\nIC序列统计:")
    stats = calc.ic_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nRank IC序列统计:")
    rank_stats = calc.rank_ic_stats()
    for key, value in rank_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    # 运行所有示例
    example_basic_evaluation()
    example_generate_report()
    
    try:
        example_visualization()
    except Exception as e:
        print(f"可视化示例出错: {e}")
    
    example_individual_metrics()
    example_ic_analysis()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
