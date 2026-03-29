"""
过滤器模块

在因子排名选股之前，对股票池进行逐日过滤（如剔除 ST/退市风险股票）。

目录结构：
  filters/
  ├── base.py              BaseFilter + FilterResult + CompositeFilter
  ├── label_builder.py     从 trade_status 构建 ST/退市标签（测评用）
  ├── filter_evaluator.py  过滤器独立测评（分类指标）
  ├── composite.py         CompositeFilter（也在 base.py 中导出）
  ├── risk/                风险类过滤器（ST、退市等）
  ├── liquidity/           流动性类过滤器（低成交量、频繁停牌等）
  └── compliance/          合规类过滤器（次新股、限售解禁等）
"""

from .base import BaseFilter, FilterResult, CompositeFilter
from .label_builder import LabelBuilder
from .filter_evaluator import FilterEvaluator

__all__ = [
    "BaseFilter",
    "FilterResult",
    "CompositeFilter",
    "LabelBuilder",
    "FilterEvaluator",
]
