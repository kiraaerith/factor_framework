# ETF 因子评估框架

ETF因子评估系统的基础框架 - 基于横截面数据结构的因子计算与评估系统。

## 项目结构

```
etf_factor_framework/
├── core/                       # 核心数据结构
│   ├── __init__.py            # 包初始化，导出主要类
│   ├── factor_data.py         # FactorData 类 (N×T因子值容器)
│   ├── position_data.py       # PositionData 类 (N×T仓位)
│   ├── ohlcv_data.py          # OHLCVData 类 (N×T×5数据)
│   ├── base_interfaces.py     # 抽象基类定义
│   └── test_basic.py          # 基础测试脚本
│
├── factors/                    # 因子计算模块
│   ├── __init__.py            # 包初始化，导出因子类
│   ├── ohlcv_calculator.py    # OHLCV因子计算器基类
│   ├── technical_factors.py   # 技术因子实现
│   └── leakage_detector.py    # 未来数据泄露检测器
│
├── tests/                      # 测试模块
│   ├── __init__.py
│   ├── test_factors.py        # 因子测试
│   └── test_leakage_detector.py  # 数据泄露检测测试
│
├── script/                     # 脚本目录
├── requirements.txt            # Python依赖
└── README.md                   # 本文件
```

## 核心模块说明

### 1. core - 核心数据结构

#### FactorData (因子数据容器)

封装 N × T 维度的因子值数据。

```python
from core import FactorData

factor = FactorData(
    values=pd.DataFrame(...),  # N × T DataFrame
    name='Momentum',
    params={'period': 20}
)

# 常用方法
factor.rank()        # 计算排名
factor.zscore()      # Z-Score标准化
factor.demean()      # 去均值
factor.get_cross_section(date)  # 获取横截面数据
factor.get_time_series(symbol)  # 获取时间序列数据
```

#### PositionData (仓位数据容器)

封装 N × T 维度的仓位权重数据。

```python
from core import PositionData

position = PositionData(
    weights=pd.DataFrame(...),  # N × T DataFrame
    name='EqualWeight',
    params={'k': 5}
)

# 常用方法
position.get_total_weights()     # 获取总权重
position.get_position_count()    # 获取持仓数量
position.normalize()             # 归一化权重
position.shift(1)                # 时间平移（调仓延迟）
```

#### OHLCVData (OHLCV数据容器)

封装 N × T × 5 维度的OHLCV数据。

```python
from core import OHLCVData

ohlcv = OHLCVData(
    open=open_df,      # N × T DataFrame
    high=high_df,      # N × T DataFrame
    low=low_df,        # N × T DataFrame
    close=close_df,    # N × T DataFrame
    volume=volume_df   # N × T DataFrame
)

# 常用方法
ohlcv.get_returns()              # 计算收益率
ohlcv.get_log_returns()          # 计算对数收益率
ohlcv.get_vwap()                 # 计算典型价格
ohlcv.get_true_range()           # 计算真实波幅
ohlcv.get_cross_section(date)    # 获取横截面数据
```

### 2. factors - 因子计算模块

#### 技术因子列表

| 因子类 | 说明 | 参数 |
|--------|------|------|
| `CloseOverMA` | 收盘价与均线比率 | `period`, `field` |
| `RSI` | 相对强弱指标 | `period`, `field` |
| `Momentum` | 动量因子 | `period`, `field`, `log_return` |
| `MACD` | MACD指标 | `fast_period`, `slow_period`, `signal_period` |
| `BollingerBands` | 布林带位置 | `period`, `std_multiplier` |
| `FutureReturn` | 未来收益率 (前瞻因子) | `period`, `field`, `log_return` |

#### 使用示例

```python
from core import OHLCVData
from factors import CloseOverMA, RSI, Momentum

# 准备OHLCV数据
ohlcv = OHLCVData(open=..., high=..., low=..., close=..., volume=...)

# 计算CloseOverMA因子
factor = CloseOverMA(period=20)
result = factor.calculate(ohlcv)
print(result.name)      # "CloseOverMA_20"
print(result.shape)     # (N, T)

# 计算RSI因子
rsi_factor = RSI(period=14)
rsi_result = rsi_factor.calculate(ohlcv)

# 计算动量因子
mom_factor = Momentum(period=20, log_return=True)
mom_result = mom_factor.calculate(ohlcv)
```

#### 因子注册表

通过名称动态创建因子实例：

```python
from factors import list_available_factors, create_factor

# 查看所有可用因子
print(list_available_factors())
# ['CloseOverMA', 'RSI', 'Momentum', 'MACD', 'BollingerBands', 'FutureReturn']

# 通过名称创建因子
factor = create_factor('RSI', period=14)
```

### 3. leakage_detector - 数据泄露检测模块

自动检测因子计算是否存在未来数据泄露问题。

#### 检测原理

1. 将完整数据集沿时间维度切开，构成短数据集（如前50%）
2. 用因子计算函数分别计算短数据集和长数据集的因子值
3. 比较两者在重叠时间段的因子值是否完全一致
4. 如果不一致，说明存在未来数据泄露

#### 使用示例

```python
from factors import CloseOverMA
from factors.leakage_detector import LeakageDetector, detect_leakage

# 创建检测器
detector = LeakageDetector(split_ratio=0.5, verbose=True)

# 检测单个因子
factor = CloseOverMA(period=20)
report = detector.detect(factor, ohlcv_data)

# 查看结果
print(report.has_leakage)    # False 表示无泄露
print(report.mismatch_ratio) # 不匹配比例
report.print_report()        # 打印详细报告

# 便捷检测函数
report = detect_leakage(factor, ohlcv_data, verbose=True)
```

#### LeakageReport 属性

| 属性 | 说明 |
|------|------|
| `has_leakage` | 是否存在数据泄露 |
| `mismatch_ratio` | 不匹配值比例 |
| `max_absolute_diff` | 最大绝对差异 |
| `max_relative_diff` | 最大相对差异 |
| `mismatched_dates` | 不匹配的时间点列表 |

### 4. 抽象基类接口

#### FactorCalculator (因子计算器接口)

```python
from core import FactorCalculator, FactorData, OHLCVData

class MyFactorCalculator(FactorCalculator):
    @property
    def name(self) -> str:
        return "MyFactor"
    
    def get_params(self) -> dict:
        return {'param1': value1}
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        # 实现因子计算逻辑
        close = ohlcv_data.close
        # ... 计算因子值
        return FactorData(factor_values, name=self.name, params=self.get_params())
```

#### PositionMapper (仓位映射器接口)

```python
from core import PositionMapper, FactorData, PositionData

class MyPositionMapper(PositionMapper):
    @property
    def name(self) -> str:
        return "MyMapper"
    
    def map_to_position(self, factor_data: FactorData) -> PositionData:
        # 实现仓位映射逻辑
        values = factor_data.values
        # ... 计算权重
        return PositionData(weights, name=self.name)
```

#### Evaluator (评估器接口)

```python
from core import Evaluator, FactorData, OHLCVData, PositionData

class MyEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "MyEvaluator"
    
    @property
    def metrics(self) -> list:
        return ['metric1', 'metric2']
    
    def evaluate(self, factor_data, ohlcv_data, position_data=None) -> dict:
        # 实现评估逻辑
        return {'metric1': value1, 'metric2': value2}
```

## 测试

### 运行所有测试

```bash
cd etf_factor_framework
python -m pytest tests/ -v
```

### 运行特定测试

```bash
# 测试因子模块
python -m pytest tests/test_factors.py -v

# 测试数据泄露检测
python -m pytest tests/test_leakage_detector.py -v

# 测试核心数据结构
python core/test_basic.py
```

## 依赖

```
numpy >= 1.20.0
pandas >= 1.3.0
pytest >= 7.0.0  # 用于测试
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 开发阶段

### 阶段一 ✅ (已完成)
- [x] 定义 FactorData 类
- [x] 定义 PositionData 类
- [x] 定义 OHLCVData 类
- [x] 定义抽象基类 FactorCalculator
- [x] 定义抽象基类 PositionMapper
- [x] 定义抽象基类 Evaluator

### 阶段二 ✅ (已完成)
- [x] 实现 OHLCVFactorCalculator 基类
- [x] 实现 CloseOverMA 因子
- [x] 实现 RSI 因子
- [x] 实现 Momentum 因子
- [x] 实现 MACD 因子
- [x] 实现 BollingerBands 因子
- [x] 实现 FutureReturn 前瞻因子
- [x] 实现数据泄露检测器

### 阶段三 (待实现)
- [ ] 实现 RankBasedMapper
- [ ] 实现 ThresholdMapper
- [ ] 实现 LongShortMapper

### 阶段四 (待实现)
- [ ] 实现收益类指标
- [ ] 实现风险类指标
- [ ] 实现IC类指标
- [ ] 实现可视化图表

### 阶段五至八 (待实现)
- [ ] 结果存储系统
- [ ] 配置系统
- [ ] 网格搜索
- [ ] 因子库集成

## 完整示例

```python
import pandas as pd
import numpy as np

from core import OHLCVData
from factors import CloseOverMA, RSI, create_factor
from factors.leakage_detector import detect_leakage

# 1. 创建示例数据
symbols = ['ETF1', 'ETF2', 'ETF3']
dates = pd.date_range('2024-01-01', periods=100)
np.random.seed(42)

close = pd.DataFrame(
    np.random.randn(3, 100).cumsum(axis=1) + 100,
    index=symbols, columns=dates
)

ohlcv = OHLCVData(
    open=close * 0.99,
    high=close * 1.02,
    low=close * 0.98,
    close=close,
    volume=pd.DataFrame(np.abs(np.random.randn(3, 100)) * 1000, 
                       index=symbols, columns=dates)
)

# 2. 计算因子
factor1 = CloseOverMA(period=20)
result1 = factor1.calculate(ohlcv)

factor2 = RSI(period=14)
result2 = factor2.calculate(ohlcv)

# 3. 数据泄露检测
report = detect_leakage(factor1, ohlcv, verbose=True)
if not report.has_leakage:
    print(f"✅ {factor1.name} 无数据泄露")

# 4. 获取因子值
latest_rank = result1.get_cross_section(dates[-1]).rank()
print(f"最新排名:\n{latest_rank}")

# 5. 使用因子注册表
factor3 = create_factor('Momentum', period=20)
result3 = factor3.calculate(ohlcv)
```

## 注意事项

1. **数据泄露警告**: `FutureReturn` 因子使用了未来信息，仅用于作为理论极限情况的baseline参考，不应用于实际交易！

2. **因子形状**: 所有因子计算返回的 `FactorData` 形状应与输入的 `OHLCVData` 一致 (N × T)。

3. **NaN处理**: 因子计算器应合理处理NaN值，避免在计算中传播不必要的NaN。

## 许可证

MIT License
