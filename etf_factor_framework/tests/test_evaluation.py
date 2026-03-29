"""
评估模块单元测试

测试各类评估指标的计算正确性。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import (
    # 收益类
    total_return,
    annualized_return,
    cumulative_returns,
    ReturnsMetricsCalculator,
    calculate_portfolio_returns,
    
    # 风险类
    max_drawdown,
    annualized_volatility,
    downside_volatility,
    RiskMetricsCalculator,
    
    # 风险调整类
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    RiskAdjustedMetricsCalculator,
    
    # 换手率类
    turnover_rate,
    average_turnover,
    TurnoverMetricsCalculator,
    
    # IC类
    calculate_ic,
    calculate_ic_series,
    calculate_rank_ic,
    calculate_icir,
    calculate_ic_statistics,
    calculate_forward_returns,
    ICMetricsCalculator,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_returns():
    """创建样本收益率序列"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    return returns


@pytest.fixture
def sample_factor_data():
    """创建样本因子数据"""
    np.random.seed(42)
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    values = pd.DataFrame(
        np.random.randn(5, 50),
        index=symbols,
        columns=dates
    )
    return values


@pytest.fixture
def sample_forward_returns():
    """创建样本未来收益数据"""
    np.random.seed(42)
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    values = pd.DataFrame(
        np.random.randn(5, 50) * 0.02,
        index=symbols,
        columns=dates
    )
    return values


@pytest.fixture
def sample_position_weights():
    """创建样本仓位权重"""
    symbols = ['ETF1', 'ETF2', 'ETF3']
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    weights = pd.DataFrame(
        [[0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.3, 0.3, 0.4, 0.4],
         [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3],
         [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]],
        index=symbols,
        columns=dates
    )
    return weights


# ==================== 收益类指标测试 ====================

class TestReturnsMetrics:
    """测试收益类指标"""
    
    def test_total_return_basic(self):
        """测试总收益基本计算"""
        returns = pd.Series([0.1, -0.05, 0.08])
        result = total_return(returns)
        expected = (1.1 * 0.95 * 1.08) - 1
        assert np.isclose(result, expected)
    
    def test_total_return_empty(self):
        """测试空序列"""
        returns = pd.Series([], dtype=float)
        result = total_return(returns)
        assert np.isnan(result)
    
    def test_annualized_return(self):
        """测试年化收益"""
        # 252天，每天1%收益
        returns = pd.Series([0.01] * 252)
        result = annualized_return(returns, periods_per_year=252)
        # 年化收益约为 (1.01^252 - 1) = 11.27
        assert result > 10
    
    def test_cumulative_returns(self):
        """测试累积收益序列"""
        returns = pd.Series([0.1, -0.05, 0.08])
        result = cumulative_returns(returns)
        expected = pd.Series([0.1, 0.045, 0.1286])
        assert np.allclose(result.values, expected.values, rtol=0.01)
    
    def test_returns_calculator(self, sample_returns):
        """测试收益计算器类"""
        calc = ReturnsMetricsCalculator(sample_returns, periods_per_year=252)
        metrics = calc.get_all_metrics()
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'average_return' in metrics
        assert isinstance(metrics['total_return'], float)


# ==================== 风险类指标测试 ====================

class TestRiskMetrics:
    """测试风险类指标"""
    
    def test_max_drawdown(self):
        """测试最大回撤计算"""
        # 先涨10%，再跌15%
        returns = pd.Series([0.1, -0.05, -0.05, -0.05, 0.02])
        result = max_drawdown(returns)
        assert result < 0
        # 回撤应该大于0但小于20%
        assert -0.20 < result < 0
    
    def test_max_drawdown_increasing(self):
        """测试一直上涨时的回撤"""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        result = max_drawdown(returns)
        assert result == 0 or np.isclose(result, 0)
    
    def test_annualized_volatility(self, sample_returns):
        """测试年化波动率"""
        result = annualized_volatility(sample_returns, periods_per_year=252)
        assert result > 0
        # 日波动2%，年化约32%
        assert 0.2 < result < 0.5
    
    def test_downside_volatility(self):
        """测试下行波动率"""
        returns = pd.Series([0.02, -0.01, -0.02, 0.01, 0.01])
        result = downside_volatility(returns, periods_per_year=252)
        # 应该小于总波动率
        total_vol = annualized_volatility(returns, periods_per_year=252)
        assert result < total_vol
    
    def test_risk_calculator(self, sample_returns):
        """测试风险计算器类"""
        calc = RiskMetricsCalculator(sample_returns, periods_per_year=252)
        metrics = calc.get_all_metrics()
        
        assert 'max_drawdown' in metrics
        assert 'annualized_volatility' in metrics
        assert metrics['max_drawdown'] <= 0


# ==================== 风险调整类指标测试 ====================

class TestRiskAdjustedMetrics:
    """测试风险调整收益指标"""
    
    def test_sharpe_ratio_positive(self):
        """测试正收益时的夏普比率"""
        returns = pd.Series([0.001] * 100)  # 正收益
        result = sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252)
        assert result > 0
    
    def test_sharpe_ratio_zero_vol(self):
        """测试零波动时的夏普比率"""
        returns = pd.Series([0.001] * 10)  # 完全相同的收益
        # 这种特殊情况，标准差为0或极小
        # 应该返回nan或inf或极大的数
        result = sharpe_ratio(returns)
        # 由于浮点精度，可能得到极大数值而非真正的inf
        assert np.isnan(result) or np.isinf(result) or abs(result) > 1e10
    
    def test_calmar_ratio(self):
        """测试卡玛比率"""
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        result = calmar_ratio(returns, periods_per_year=252)
        # 有回撤时应该为正
        assert isinstance(result, float)
    
    def test_sortino_ratio(self):
        """测试索提诺比率"""
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        result = sortino_ratio(returns, periods_per_year=252)
        assert isinstance(result, float)
    
    def test_risk_adjusted_calculator(self, sample_returns):
        """测试风险调整收益计算器"""
        calc = RiskAdjustedMetricsCalculator(sample_returns)
        metrics = calc.get_all_metrics()
        
        assert 'sharpe_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'sortino_ratio' in metrics


# ==================== 换手率类指标测试 ====================

class TestTurnoverMetrics:
    """测试换手率类指标"""
    
    def test_turnover_rate(self, sample_position_weights):
        """测试换手率序列"""
        result = turnover_rate(sample_position_weights)
        # 应该有9个值（10天产生9个变化）
        assert len(result) == 9
        # 所有值都非负
        assert (result >= 0).all()
    
    def test_average_turnover(self, sample_position_weights):
        """测试平均换手率"""
        result = average_turnover(sample_position_weights)
        assert result >= 0
        # 换手率应该在0-1之间
        assert 0 <= result <= 1
    
    def test_turnover_calculator(self, sample_position_weights):
        """测试换手率计算器"""
        calc = TurnoverMetricsCalculator(sample_position_weights)
        metrics = calc.get_all_metrics()
        
        assert 'avg_daily_turnover' in metrics
        assert 'annualized_turnover' in metrics
        assert metrics['avg_daily_turnover'] >= 0


# ==================== IC类指标测试 ====================

class TestICMetrics:
    """测试IC类指标"""
    
    def test_calculate_ic_perfect_correlation(self):
        """测试完全正相关时的IC"""
        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        result = calculate_ic(factor, returns, method='pearson')
        assert np.isclose(result, 1.0, rtol=0.01)
    
    def test_calculate_ic_no_correlation(self):
        """测试无相关时的IC"""
        np.random.seed(42)
        factor = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))
        result = calculate_ic(factor, returns)
        # 应该接近0
        assert abs(result) < 0.3
    
    def test_calculate_rank_ic(self, sample_factor_data, sample_forward_returns):
        """测试Rank IC序列"""
        result = calculate_rank_ic(sample_factor_data, sample_forward_returns)
        # 结果应该是序列
        assert isinstance(result, pd.Series)
        # 值应该在-1到1之间
        assert (result.abs() <= 1.01).all()
    
    def test_calculate_icir(self):
        """测试ICIR计算"""
        # 稳定的IC序列
        ic_series = pd.Series([0.05, 0.04, 0.06, 0.05, 0.04, 0.06])
        result = calculate_icir(ic_series, annualize=False)
        # ICIR = mean / std
        expected = ic_series.mean() / ic_series.std()
        assert np.isclose(result, expected)
    
    def test_calculate_ic_statistics(self):
        """测试IC统计指标"""
        ic_series = pd.Series([0.05, 0.04, 0.06, 0.05, 0.04, 0.06, -0.01, 0.03])
        result = calculate_ic_statistics(ic_series)
        
        assert 'mean' in result
        assert 'std' in result
        assert 'positive_ratio' in result
        assert 't_stat' in result
        assert 'p_value' in result
    
    def test_calculate_forward_returns(self):
        """测试前瞻收益计算"""
        # 构建正确的DataFrame：index=标的，columns=日期
        dates = pd.date_range('2024-01-01', periods=4)
        close_prices = pd.DataFrame(
            [[100, 102, 104, 106],
             [50, 51, 52, 53]],
            index=['ETF1', 'ETF2'],
            columns=dates
        )
        
        result = calculate_forward_returns(close_prices, periods=1)
        # T日的前瞻收益 = (T+1日价格 / T日价格) - 1
        assert np.isclose(result.iloc[0, 0], 0.02)  # 102/100 - 1 = 0.02
        assert np.isclose(result.iloc[0, 1], 104/102 - 1)  # 104/102 - 1
    
    def test_ic_calculator(self, sample_factor_data, sample_forward_returns):
        """测试IC计算器类"""
        calc = ICMetricsCalculator(
            sample_factor_data,
            sample_forward_returns,
            periods_per_year=252
        )
        metrics = calc.get_all_metrics()
        
        assert 'ic_mean' in metrics
        assert 'rank_ic_mean' in metrics
        assert 'rank_ic_ir' in metrics


# ==================== 组合收益测试 ====================

class TestPortfolioReturns:
    """测试组合收益计算"""
    
    def test_calculate_portfolio_returns(self):
        """测试组合收益计算"""
        weights = pd.DataFrame({
            '2024-01-01': [0.5, 0.5],
            '2024-01-02': [0.6, 0.4],
            '2024-01-03': [0.4, 0.6],
        }, index=['A', 'B']).T
        
        returns = pd.DataFrame({
            '2024-01-01': [0.01, 0.02],
            '2024-01-02': [0.02, 0.01],
            '2024-01-03': [-0.01, 0.03],
        }, index=['A', 'B']).T
        
        result = calculate_portfolio_returns(weights, returns, delay=1)
        
        # 应该有值
        assert len(result) > 0
        # 延迟1天，所以T日的仓位在T+1日产生收益


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_full_evaluation_workflow(self):
        """测试完整评估流程"""
        # 创建测试数据
        np.random.seed(42)
        symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 因子数据
        factor_values = pd.DataFrame(
            np.random.randn(5, 100),
            index=symbols,
            columns=dates
        )
        
        # 收盘价数据（用于计算收益和前瞻收益）
        close_prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(5, 100) * 0.02, axis=1),
            index=symbols,
            columns=dates
        )
        
        # 前瞻收益
        forward_returns = calculate_forward_returns(close_prices, periods=1)
        
        # 计算IC
        ic_calc = ICMetricsCalculator(factor_values, forward_returns)
        ic_metrics = ic_calc.get_all_metrics()
        
        # 验证结果
        assert 'ic_mean' in ic_metrics
        assert 'rank_ic_mean' in ic_metrics
        assert abs(ic_metrics['ic_mean']) < 0.5  # 随机数据的IC应该接近0
    
    def test_end_to_end_with_position(self):
        """测试带仓位的完整流程"""
        np.random.seed(42)
        symbols = ['ETF1', 'ETF2', 'ETF3']
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # 仓位权重（等权）
        position_weights = pd.DataFrame(
            [[1/3, 1/3, 1/3]] * 50,
            index=dates,
            columns=symbols
        ).T
        
        # 资产收益
        asset_returns = pd.DataFrame(
            np.random.randn(3, 50) * 0.02,
            index=symbols,
            columns=dates
        )
        
        # 计算组合收益
        portfolio_returns = calculate_portfolio_returns(
            position_weights, asset_returns, delay=1
        )
        
        # 计算各项指标
        returns_calc = ReturnsMetricsCalculator(portfolio_returns)
        risk_calc = RiskMetricsCalculator(portfolio_returns)
        
        returns_metrics = returns_calc.get_all_metrics()
        risk_metrics = risk_calc.get_all_metrics()
        
        assert 'total_return' in returns_metrics
        assert 'max_drawdown' in risk_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
