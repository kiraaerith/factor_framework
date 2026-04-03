# Rolling Model - 滚动训练模型算子

通过可插拔的模型对多个因子进行滚动训练融合，输出因子值矩阵。

## 架构

```
DataPreparer(ABC)     ModelWrapper(ABC)     RollingTrainer      DiagnosticsTool
加载对齐数据           封装训练/推理          滚动窗口编排         诊断指标(可选)
      ↓                     ↓                    ↓                   ↓
 PreparedData ────→ RollingTrainer.run(data, model) ──→ (T, N) 因子值矩阵
                                │
                                └──→ diagnostics.db（开启时）
```

| 组件 | 类型 | 职责 |
|------|------|------|
| `DataPreparer` | 抽象接口 | 子类决定如何加载和对齐数据 |
| `ModelWrapper` | 抽象接口 | 子类决定如何训练推理、处理 NaN |
| `RollingTrainer` | 固定实现 | 窗口切分、模型实例化、结果拼接 |
| `DiagnosticsTool` | 固定实现 | 每轮即算即存诊断指标到 SQLite |

## 滚动窗口逻辑

训练集从 `train_min_days` 逐渐增长到 `train_max_days`，之后固定长度滑动：

```
Phase 1: 增长期
  模型1: [--train 2y--][val 6m][test 6m]
  模型2: [---train 2.5y---][val 6m][test 6m]
  ...
  模型K: [------train 5y------][val 6m][test 6m]  ← 达到 max

Phase 2: 滑动期
  模型K+1:  [------train 5y------][val 6m][test 6m]
  模型K+2:     [------train 5y------][val 6m][test 6m]
  ...

各模型的测试集拼接 → 最终因子值矩阵，之前的部分为 NaN
```

## 快速使用

```python
import numpy as np
from etf_factor_framework.rolling_model import PreparedData, RollingTrainer
from etf_factor_framework.rolling_model.models.ols import OLSModel

# 1. 准备数据
data = PreparedData(
    features=my_features,          # (T, N, F) ndarray
    labels=my_labels,              # (T, N) ndarray
    dates=my_dates,                # (T,) ndarray
    symbols=my_symbols,            # (N,) ndarray
    feature_names=["PE", "PB"],
)

# 2. 滚动训练
trainer = RollingTrainer(
    train_min_days=504,    # 2年
    train_max_days=1260,   # 5年
    val_days=126,          # 半年
    test_days=126,         # 半年
    step_days=126,         # 半年滚动
)
factor_values = trainer.run(data, OLSModel, {"fit_intercept": True})
# factor_values: (T, N) ndarray
```

## 开启诊断

```python
trainer = RollingTrainer(
    train_min_days=504,
    train_max_days=1260,
    val_days=126,
    test_days=126,
    step_days=126,
    diagnostics_enabled=True,
    diagnostics_db_path="rolling_diagnostics/run_001/diagnostics.db",
)
factor_values = trainer.run(data, OLSModel, {"fit_intercept": True})
```

诊断数据库包含两张表：

**window_metrics** — 每轮一行，训练集和验证集各 7 个标量指标：

| 指标 | 说明 |
|------|------|
| ic / icir | Pearson IC 均值及其 IR |
| rank_ic / rank_icir | Spearman RankIC 均值及其 IR |
| sharpe | 多头组合年化夏普 |
| annual_return | 多头组合年化收益率 |
| max_drawdown | 多头组合最大回撤 |

**window_curves** — 验证集曲线（BLOB 存储）：
- `val_returns`：日频收益曲线 `(T_val,)`
- `val_quantile_returns`：5 分层日频收益曲线 `(T_val, 5)`

读取曲线：
```python
from etf_factor_framework.rolling_model.diagnostics import DiagnosticsTool

curve = DiagnosticsTool.load_curve("diagnostics.db", window_id=0, curve_type="val_returns")
```

## 已有模型

| 模型 | 文件 | 验证集用途 | NaN 处理 |
|------|------|-----------|----------|
| `OLSModel` | `models/ols.py` | 不使用 | 丢弃 NaN 行 |
| `LassoModel` | `models/lasso.py` | 选 alpha（可选） | 丢弃 NaN 行 |

## 自定义模型

继承 `ModelWrapper`，实现 `fit` 和 `predict`：

```python
from etf_factor_framework.rolling_model.base import ModelWrapper

class MyModel(ModelWrapper):
    def __init__(self, **params):
        # 保存超参数
        ...

    def fit(self, train_X, train_y, val_X, val_y):
        # train_X: (T_train, N, F), train_y: (T_train, N)
        # val_X:   (T_val, N, F),   val_y:   (T_val, N)
        # 自行决定如何展平、处理 NaN、使用验证集
        ...

    def predict(self, test_X):
        # test_X: (T_test, N, F) → 返回 (T_test, N)
        ...
```

每轮滚动会创建新实例，无需担心状态污染。

## 自定义数据准备

继承 `DataPreparer`，实现 `prepare`：

```python
from etf_factor_framework.rolling_model.base import DataPreparer, PreparedData

class MyPreparer(DataPreparer):
    def prepare(self) -> PreparedData:
        # 加载数据，对齐时间/股票轴
        # 返回 PreparedData(features, labels, dates, symbols, feature_names)
        ...
```

## 文件结构

```
rolling_model/
├── __init__.py
├── base.py                 # PreparedData, DataPreparer(ABC), ModelWrapper(ABC)
├── rolling_trainer.py      # RollingTrainer
├── diagnostics.py          # DiagnosticsTool
├── models/
│   ├── ols.py              # OLSModel
│   └── lasso.py            # LassoModel
└── preparers/
    └── (按需新增)
```
