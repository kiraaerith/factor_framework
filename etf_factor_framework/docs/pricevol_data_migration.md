# PriceVolData: calculate() 签名变更与量价因子迁移指南

**日期**: 2026-04-03  
**影响范围**: 所有继承 `FundamentalFactorCalculator` 的因子类

---

## 一、为什么要做这个改动

### 问题

框架的泄露检测器 (`FundamentalLeakageDetector`) 通过 `truncate()` 截断数据来检测因子是否使用了未来数据。它的工作原理是：

1. 将数据截断到某个时间点（如 2022-06-30）
2. 分别用完整数据和截断数据计算因子
3. 比较重叠时间段的因子值——如果不一致，说明因子使用了未来数据

这个机制对基本面因子有效，因为基本面因子通过 `FundamentalData` 获取数据，`truncate()` 能控制其数据范围。

但**量价因子（如 JT 动量）绕过了这个机制**——它在 `calculate()` 内部自行创建 `StockDataLoader` 加载 OHLCV 数据，完全不受 `truncate()` 控制。即使泄露检测器截断了 `FundamentalData`，量价因子仍然加载完整的价格数据，导致泄露检测形同虚设。

### 解决方案

新增 `PriceVolData` 量价数据容器，与 `FundamentalData` 对称设计：

| | FundamentalData | PriceVolData |
|---|---|---|
| **数据源** | 理杏仁 SQLite (lixinger.db) | Tushare SQLite (tushare.db) |
| **数据类型** | 季度财务数据 → 日频 forward-fill | 日频 OHLCV（后复权） |
| **truncate()** | 截断 report_date | 截断交易日 |
| **传递方式** | `calculate(fundamental_data)` | `calculate(..., pricevol_data)` |

量价因子从 `pricevol_data` 获取数据，泄露检测器同时截断两个数据容器，检测机制对所有因子类型统一生效。

---

## 二、改了什么

### 2.1 `calculate()` 签名变更（119 个文件）

基类和所有子类的签名从：

```python
def calculate(self, fundamental_data: FundamentalData) -> FactorData:
```

变为：

```python
def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
```

**纯基本面因子不需要做任何其他修改**——`pricevol_data` 默认 `None`，忽略即可。

### 2.2 新增文件

| 文件 | 说明 |
|------|------|
| `factors/fundamental/pricevol_data.py` | `PriceVolData` 类 |

### 2.3 修改的文件

| 文件 | 说明 |
|------|------|
| `factors/fundamental/fundamental_calculator.py` | 基类 `calculate()` 签名 |
| `factors/fundamental/fundamental_leakage_detector.py` | `detect()` 接受并截断 `pricevol_data` |
| `scripts/run_factor_grid_v3.py` | 创建 `PriceVolData` 并传递给因子和检测器 |
| `factors/momentum/jt_momentum.py` | 改为从 `pricevol_data` 获取 OHLCV |
| 其余 115 个因子文件 | 仅签名变更，逻辑不变 |

---

## 三、Agent 迁移指南

### 场景 1：纯基本面因子（不需要量价数据）

**无需修改逻辑。** 只需确认 `calculate()` 签名包含 `pricevol_data=None`：

```python
class MyFundamentalFactor(FundamentalFactorCalculator):
    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        # pricevol_data 不使用，忽略即可
        values, symbols, dates = fundamental_data.get_daily_panel('q_m_roe_t')
        ...
```

### 场景 2：需要量价数据的因子（新写或迁移已有）

使用 `pricevol_data.get_ohlcv()` 获取 OHLCV 数据：

```python
from factors.fundamental.pricevol_data import PriceVolData

class MyPriceVolFactor(FundamentalFactorCalculator):
    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        # 计算需要的回望天数
        lookback_days = 252 + 30  # 例如 1 年 + 缓冲

        if pricevol_data is not None:
            ohlcv = pricevol_data.get_ohlcv(lookback_extra_days=lookback_days)
        else:
            # Fallback: 直接加载（泄露检测不生效）
            from data.stock_data_loader import StockDataLoader
            loader = StockDataLoader()
            ohlcv = loader.load_ohlcv(
                start_date=str(fundamental_data.start_date.date()),
                end_date=str(fundamental_data.end_date.date()),
                use_adjusted=True,
                lookback_extra_days=lookback_days,
            )

        close = ohlcv.close      # (N, T) ndarray
        symbols = ohlcv.symbols  # (N,)
        dates = ohlcv.dates      # (T,) — 包含回望期，需自行 trim 到 start_date
        ...
```

**要点：**
- `get_ohlcv(lookback_extra_days=N)` 返回的日期范围是 `[start_date - N天, end_date]`，包含回望期
- 因子输出时需要自行 trim 到 `fundamental_data.start_date` 之后（参考 `jt_momentum.py` 的 trim 逻辑）
- 保留 fallback 分支可以让因子在单独测试时也能运行，但 **只有通过 `pricevol_data` 路径才支持泄露检测**

### 场景 3：同时需要基本面和量价数据的因子

两个参数都可以使用：

```python
class MyHybridFactor(FundamentalFactorCalculator):
    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        # 基本面数据
        roe, symbols_f, dates_f = fundamental_data.get_daily_panel('q_m_roe_t')

        # 量价数据
        ohlcv = pricevol_data.get_ohlcv(lookback_extra_days=60)
        close = ohlcv.close

        # 对齐 symbols/dates 后组合计算
        ...
```

---

## 四、PriceVolData API 参考

```python
from factors.fundamental.pricevol_data import PriceVolData

# 创建
pvd = PriceVolData(start_date='2016-01-01', end_date='2025-12-31')

# 获取 OHLCV（懒加载，首次调用触发数据库查询）
ohlcv = pvd.get_ohlcv(lookback_extra_days=300)
# ohlcv.close   — (N, T) ndarray, 后复权收盘价
# ohlcv.open    — (N, T) ndarray, 后复权开盘价
# ohlcv.high    — (N, T) ndarray
# ohlcv.low     — (N, T) ndarray
# ohlcv.volume  — (N, T) ndarray
# ohlcv.symbols — (N,) ndarray
# ohlcv.dates   — (T,) ndarray, datetime64[ns]

# 截断（用于泄露检测，复用已缓存数据）
short_pvd = pvd.truncate('2022-06-30')
short_ohlcv = short_pvd.get_ohlcv(lookback_extra_days=300)
# short_ohlcv.dates[-1] <= 2022-06-30
```

**注意事项：**
- `get_ohlcv()` 内部有缓存，重复调用不会重新查询数据库
- 如果后续调用请求了更大的 `lookback_extra_days`，会自动重新加载
- `truncate()` 从缓存中切片，不触发额外 IO
