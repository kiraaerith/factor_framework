"""
过滤器基础设施

- FilterResult: 过滤器输出容器
- BaseFilter: 过滤器抽象基类
- CompositeFilter: 组合过滤器（串联多个过滤器，OR 逻辑）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.factor_data import FactorData


class FilterResult:
    """
    过滤器输出容器

    核心数据是 exclude_mask (N, T) bool 矩阵，True 表示该 (stock, date) 应被剔除。

    Attributes:
        exclude_mask: (N, T) bool ndarray, True=剔除
        symbols: (N,) str ndarray, 股票代码
        dates: (T,) datetime64 ndarray, 日期
        filter_name: str, 过滤器名称
    """

    def __init__(
        self,
        exclude_mask: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        filter_name: str,
    ):
        if exclude_mask.shape != (len(symbols), len(dates)):
            raise ValueError(
                f"exclude_mask shape {exclude_mask.shape} != "
                f"(len(symbols)={len(symbols)}, len(dates)={len(dates)})"
            )
        self.exclude_mask = exclude_mask.astype(bool)
        self.symbols = np.asarray(symbols)
        self.dates = np.asarray(dates)
        self.filter_name = filter_name

    @property
    def shape(self):
        return self.exclude_mask.shape

    def apply_to_factor(self, factor_data: FactorData) -> FactorData:
        """
        将过滤结果应用到因子数据：被剔除的位置设为 NaN。

        自动对齐 symbols 和 dates（取交集）。

        Args:
            factor_data: 要过滤的因子数据

        Returns:
            FactorData: 过滤后的因子数据（原对象被修改）
        """
        # 建立 symbol/date 映射
        fac_sym_map = {s: i for i, s in enumerate(factor_data.symbols)}
        fac_date_map = {d: i for i, d in enumerate(factor_data.dates)}
        flt_sym_map = {s: i for i, s in enumerate(self.symbols)}
        flt_date_map = {d: i for i, d in enumerate(self.dates)}

        # 取交集
        common_symbols = [s for s in factor_data.symbols if s in flt_sym_map]
        common_dates = [d for d in factor_data.dates if d in flt_date_map]

        if not common_symbols or not common_dates:
            return factor_data

        # 构建索引数组
        fac_sym_idx = np.array([fac_sym_map[s] for s in common_symbols])
        fac_date_idx = np.array([fac_date_map[d] for d in common_dates])
        flt_sym_idx = np.array([flt_sym_map[s] for s in common_symbols])
        flt_date_idx = np.array([flt_date_map[d] for d in common_dates])

        # 提取对齐后的 mask 子矩阵
        aligned_mask = self.exclude_mask[np.ix_(flt_sym_idx, flt_date_idx)]

        # 应用：被剔除的位置设为 NaN
        values = factor_data.values  # (N, T) - 返回拷贝或引用取决于实现
        sub = values[np.ix_(fac_sym_idx, fac_date_idx)]
        sub[aligned_mask] = np.nan
        # 写回（通过 _values 直接修改内部数据）
        factor_data._values[np.ix_(fac_sym_idx, fac_date_idx)] = sub

        return factor_data

    def get_daily_exclude_count(self) -> pd.Series:
        """每天被剔除的股票数量"""
        counts = self.exclude_mask.sum(axis=0)  # (T,)
        return pd.Series(counts, index=pd.DatetimeIndex(self.dates), name="exclude_count")

    def get_daily_exclude_ratio(self) -> pd.Series:
        """每天被剔除的股票占比"""
        N = self.exclude_mask.shape[0]
        ratios = self.exclude_mask.sum(axis=0) / N
        return pd.Series(ratios, index=pd.DatetimeIndex(self.dates), name="exclude_ratio")

    def get_total_exclude_count(self) -> int:
        """被剔除的 (stock, date) 单元格总数"""
        return int(self.exclude_mask.sum())

    def summary(self) -> Dict[str, Any]:
        """返回过滤结果摘要"""
        return {
            "filter_name": self.filter_name,
            "n_symbols": len(self.symbols),
            "n_dates": len(self.dates),
            "total_cells": int(self.exclude_mask.size),
            "excluded_cells": self.get_total_exclude_count(),
            "excluded_ratio": float(self.exclude_mask.mean()),
            "avg_daily_excluded": float(self.exclude_mask.sum(axis=0).mean()),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"FilterResult('{s['filter_name']}', "
            f"{s['n_symbols']} symbols × {s['n_dates']} dates, "
            f"excluded {s['excluded_ratio']:.1%})"
        )


class BaseFilter(ABC):
    """
    过滤器抽象基类

    所有过滤器继承此类，实现 predict() 方法。
    一个过滤器一个 .py 文件，按用途放在对应子目录下。
    """

    @abstractmethod
    def predict(self, **kwargs) -> FilterResult:
        """
        执行过滤，返回 FilterResult。

        不同过滤器接收不同数据输入，通过 kwargs 传入。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """过滤器名称"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """过滤器参数"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.get_params()})"

    def __call__(self, **kwargs) -> FilterResult:
        return self.predict(**kwargs)


class CompositeFilter(BaseFilter):
    """
    组合过滤器：串联多个过滤器，任一命中则剔除（OR 逻辑）。

    用法::

        combined = CompositeFilter([
            STRiskFilter(),
            DelistRiskFilter(),
        ])
        result = combined.predict(ohlcv_data=ohlcv, ...)
    """

    def __init__(self, filters: List[BaseFilter]):
        if not filters:
            raise ValueError("CompositeFilter requires at least one filter")
        self.filters = filters

    @property
    def name(self) -> str:
        names = [f.name for f in self.filters]
        return f"Composite({'+'.join(names)})"

    def get_params(self) -> Dict[str, Any]:
        return {
            "filters": [
                {"name": f.name, "params": f.get_params()}
                for f in self.filters
            ]
        }

    def predict(self, **kwargs) -> FilterResult:
        """
        依次执行所有子过滤器，合并 exclude_mask（OR 逻辑）。

        所有子过滤器必须输出相同的 symbols/dates，
        或者本方法会自动对齐到公共交集。
        """
        results = [f.predict(**kwargs) for f in self.filters]

        if len(results) == 1:
            return results[0]

        # 取公共 symbols/dates
        common_symbols = results[0].symbols
        common_dates = results[0].dates
        for r in results[1:]:
            common_symbols = np.intersect1d(common_symbols, r.symbols)
            common_dates = np.intersect1d(common_dates, r.dates)

        N, T = len(common_symbols), len(common_dates)
        combined_mask = np.zeros((N, T), dtype=bool)

        for r in results:
            sym_map = {s: i for i, s in enumerate(r.symbols)}
            date_map = {d: i for i, d in enumerate(r.dates)}
            sym_idx = np.array([sym_map[s] for s in common_symbols])
            date_idx = np.array([date_map[d] for d in common_dates])
            aligned = r.exclude_mask[np.ix_(sym_idx, date_idx)]
            combined_mask |= aligned

        return FilterResult(
            exclude_mask=combined_mask,
            symbols=common_symbols,
            dates=common_dates,
            filter_name=self.name,
        )
