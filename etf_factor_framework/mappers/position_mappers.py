"""
仓位映射器实现模块

实现具体的仓位映射策略：
    - RankBasedMapper: 基于排名的Top K选择（np.argsort 代替 DataFrame.rank()）
    - DirMapper: 直接映射因子值为仓位（np.where/np.clip 代替 DataFrame 操作）
    - QuantileMapper: 分位数仓位映射（保留逐列循环，消除 pandas 元数据开销）
    - ZScoreMapper: Z-Score 仓位映射（np.where/np.nan_to_num）
"""

from typing import Optional, Literal
import numpy as np
import pandas as pd

import sys
import os

# 处理导入路径
try:
    # 相对导入（包内使用）
    from .base_mapper import BasePositionMapper
    from .weight_methods import equal_weight, normalize_weights
    from ..core.factor_data import FactorData
    from ..core.position_data import PositionData
except ImportError:
    # 绝对导入（独立运行）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from mappers.base_mapper import BasePositionMapper
    from mappers.weight_methods import equal_weight, normalize_weights
    from core.factor_data import FactorData
    from core.position_data import PositionData


class RankBasedMapper(BasePositionMapper):
    """
    基于排名的仓位映射器

    根据因子值的排名选择 Top K 标的，并分配权重。
    排名使用 np.argsort 实现，消除 DataFrame.rank() 开销。

    Attributes:
        top_k: 选择的标的数量
        direction: 因子方向，1=正向（值越大越好），-1=反向（值越小越好）
        weight_method: 权重分配方法
    """

    def __init__(
        self,
        top_k: int = 5,
        direction: int = 1,
        weight_method: Literal['equal', 'softmax'] = 'equal',
        temperature: float = 1.0,
        name: Optional[str] = None
    ):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if direction not in [1, -1]:
            raise ValueError(f"direction must be 1 or -1, got {direction}")
        if weight_method not in ['equal', 'softmax']:
            raise ValueError(f"Unknown weight_method: {weight_method}")

        super().__init__(
            name=name,
            top_k=top_k,
            direction=direction,
            weight_method=weight_method,
            temperature=temperature
        )

        self.top_k = top_k
        self.direction = direction
        self.weight_method = weight_method
        self.temperature = temperature

    @property
    def name(self) -> str:
        if self._custom_name:
            return self._custom_name
        dir_str = "Top" if self.direction == 1 else "Bottom"
        return f"RankBased_{dir_str}{self.top_k}_{self.weight_method}"

    def get_params(self) -> dict:
        return {
            'top_k': self.top_k,
            'direction': self.direction,
            'weight_method': self.weight_method,
            'temperature': self.temperature
        }

    def map_to_position(self, factor_data: FactorData) -> PositionData:
        """
        将因子值映射为仓位

        Args:
            factor_data: 因子数据容器

        Returns:
            PositionData: 仓位数据
        """
        arr = factor_data.values  # (N, T) ndarray

        # 排序键：direction=1 选最大（对 -arr 升序排）；direction=-1 选最小（对 arr 升序排）
        # NaN 替换为 +inf，确保 NaN 排在最后（不会被选入 top_k）
        sort_key = np.where(np.isnan(arr), np.inf, arr * (-self.direction))

        # 双重 argsort 得到 1-based 排名
        ranks = np.argsort(np.argsort(sort_key, axis=0), axis=0) + 1  # (N, T)

        selected = ranks <= self.top_k  # (N, T) bool

        # 整列全为 NaN 时不持仓（所有 +inf 排名仍会触发 top_k 选中，需清零）
        all_nan_cols = np.all(np.isnan(arr), axis=0)  # (T,)
        selected[:, all_nan_cols] = False

        if self.weight_method == 'equal':
            weights = equal_weight(selected)

        elif self.weight_method == 'softmax':
            # 只对选中的标的做 softmax，未选中保持 0
            softmax_input = np.where(selected, arr / self.temperature, -np.inf)
            max_vals = np.max(softmax_input, axis=0, keepdims=True)
            exp_vals = np.exp(softmax_input - max_vals)
            exp_vals = np.where(selected, exp_vals, 0.0)
            exp_sum = exp_vals.sum(axis=0, keepdims=True)
            exp_sum = np.where(exp_sum == 0, np.nan, exp_sum)
            weights = np.nan_to_num(exp_vals / exp_sum, nan=0.0)

        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")

        return PositionData(
            weights,
            symbols=factor_data.symbols,
            dates=factor_data.dates,
            name=self.name,
            params=self.get_params()
        )


class DirMapper(BasePositionMapper):
    """
    直接映射仓位映射器

    直接将因子值作为仓位权重，可选归一化。
    使用 np.where/np.clip 代替 DataFrame 操作。
    """

    def __init__(
        self,
        normalize: bool = True,
        target_sum: float = 1.0,
        clip_range: Optional[tuple] = None,
        fill_na: float = 0.0,
        name: Optional[str] = None
    ):
        if target_sum <= 0:
            raise ValueError(f"target_sum must be positive, got {target_sum}")
        if clip_range is not None and len(clip_range) != 2:
            raise ValueError("clip_range must be a tuple of (lower, upper)")

        super().__init__(
            name=name,
            normalize=normalize,
            target_sum=target_sum,
            clip_range=clip_range,
            fill_na=fill_na
        )

        self.normalize = normalize
        self.target_sum = target_sum
        self.clip_range = clip_range
        self.fill_na = fill_na

    @property
    def name(self) -> str:
        if self._custom_name:
            return self._custom_name
        parts = ["DirMapper"]
        if self.normalize:
            parts.append(f"norm{self.target_sum}")
        if self.clip_range:
            parts.append(f"clip[{self.clip_range[0]},{self.clip_range[1]}]")
        return "_".join(parts)

    def get_params(self) -> dict:
        return {
            'normalize': self.normalize,
            'target_sum': self.target_sum,
            'clip_range': self.clip_range,
            'fill_na': self.fill_na
        }

    def map_to_position(self, factor_data: FactorData) -> PositionData:
        """
        将因子值直接映射为仓位

        Args:
            factor_data: 因子数据容器

        Returns:
            PositionData: 仓位数据
        """
        weights = factor_data.values.copy()  # (N, T) ndarray

        # 填充 NaN
        weights = np.where(np.isnan(weights), self.fill_na, weights)

        # 截断
        if self.clip_range:
            lower, upper = self.clip_range
            weights = np.clip(weights, lower, upper)

        # 归一化
        if self.normalize:
            weights = normalize_weights(weights, self.target_sum)

        return PositionData(
            weights,
            symbols=factor_data.symbols,
            dates=factor_data.dates,
            name=self.name,
            params=self.get_params()
        )


class QuantileMapper(BasePositionMapper):
    """
    分位数映射器

    根据因子值的分位数位置分配仓位。
    保留逐列循环，消除 pandas 元数据开销，仅在 qcut 时使用 pandas。
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        long_quantile: Optional[int] = None,
        short_quantile: Optional[int] = None,
        equal_weight_within_group: bool = True,
        name: Optional[str] = None
    ):
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be at least 2, got {n_quantiles}")

        long_q = long_quantile if long_quantile is not None else n_quantiles - 1
        short_q = short_quantile

        if long_q < 0 or long_q >= n_quantiles:
            raise ValueError(f"long_quantile must be in [0, {n_quantiles-1}], got {long_q}")
        if short_q is not None and (short_q < 0 or short_q >= n_quantiles):
            raise ValueError(f"short_quantile must be in [0, {n_quantiles-1}], got {short_q}")
        if short_q is not None and short_q == long_q:
            raise ValueError("long_quantile and short_quantile cannot be the same")

        super().__init__(
            name=name,
            n_quantiles=n_quantiles,
            long_quantile=long_q,
            short_quantile=short_q,
            equal_weight_within_group=equal_weight_within_group
        )

        self.n_quantiles = n_quantiles
        self.long_quantile = long_q
        self.short_quantile = short_q
        self.equal_weight_within_group = equal_weight_within_group

    @property
    def name(self) -> str:
        if self._custom_name:
            return self._custom_name
        name_parts = [f"Quantile_Q{self.n_quantiles}"]
        name_parts.append(f"L{self.long_quantile}")
        if self.short_quantile is not None:
            name_parts.append(f"S{self.short_quantile}")
        return "_".join(name_parts)

    def get_params(self) -> dict:
        return {
            'n_quantiles': self.n_quantiles,
            'long_quantile': self.long_quantile,
            'short_quantile': self.short_quantile,
            'equal_weight_within_group': self.equal_weight_within_group
        }

    def map_to_position(self, factor_data: FactorData) -> PositionData:
        """
        将因子值映射为分位数仓位

        保留逐列循环，内层用 pd.qcut 计算分位标签（仅用于分桶计算，不持有 DataFrame）。
        """
        arr = factor_data.values  # (N, T) ndarray
        N, T = arr.shape
        result = np.zeros((N, T), dtype=np.float64)

        for t in range(T):
            col = arr[:, t]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                continue

            valid_vals = col[valid_mask]
            try:
                labels = pd.qcut(valid_vals, q=self.n_quantiles, labels=False, duplicates='drop')
                quantile_labels = np.full(N, np.nan)
                quantile_labels[valid_mask] = labels
            except ValueError:
                quantile_labels = np.where(valid_mask, float(self.n_quantiles // 2), np.nan)

            # 多头仓位
            long_mask = (quantile_labels == self.long_quantile)
            long_count = long_mask.sum()
            if long_count > 0:
                if self.equal_weight_within_group:
                    result[long_mask, t] = 0.5 / long_count
                else:
                    long_vals = col[long_mask]
                    long_sum = long_vals.sum()
                    if long_sum != 0:
                        result[long_mask, t] = long_vals / long_sum * 0.5

            # 空头仓位
            if self.short_quantile is not None:
                short_mask = (quantile_labels == self.short_quantile)
                short_count = short_mask.sum()
                if short_count > 0:
                    if self.equal_weight_within_group:
                        result[short_mask, t] = -0.5 / short_count
                    else:
                        short_vals = col[short_mask]
                        val_range = short_vals.max() - short_vals.min() + 1e-10
                        result[short_mask, t] = -(short_vals.max() - short_vals) / val_range / short_count * 0.5

        return PositionData(
            result,
            symbols=factor_data.symbols,
            dates=factor_data.dates,
            name=self.name,
            params=self.get_params()
        )


class ZScoreMapper(BasePositionMapper):
    """
    Z-Score 映射器

    将因子值转换为 Z-Score 后作为仓位权重。
    使用 np.where/np.nan_to_num 代替 DataFrame 操作。
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        normalize: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(
            name=name,
            threshold=threshold,
            normalize=normalize
        )

        self.threshold = threshold
        self.normalize = normalize

    @property
    def name(self) -> str:
        if self._custom_name:
            return self._custom_name
        parts = ["ZScore"]
        if self.threshold:
            parts.append(f"th{self.threshold}")
        if self.normalize:
            parts.append("norm")
        return "_".join(parts)

    def get_params(self) -> dict:
        return {
            'threshold': self.threshold,
            'normalize': self.normalize
        }

    def map_to_position(self, factor_data: FactorData) -> PositionData:
        """
        将因子值映射为 Z-Score 仓位

        Args:
            factor_data: 因子数据容器

        Returns:
            PositionData: 仓位数据
        """
        zscore_data = factor_data.zscore(axis=1)
        weights = zscore_data.values.copy()  # (N, T) ndarray

        # 阈值过滤
        if self.threshold is not None:
            weights = np.where(np.abs(weights) >= self.threshold, weights, 0.0)

        # 填充 NaN 为 0
        weights = np.nan_to_num(weights, nan=0.0)

        # 归一化
        if self.normalize:
            weights = normalize_weights(weights, target_sum=1.0)

        return PositionData(
            weights,
            symbols=factor_data.symbols,
            dates=factor_data.dates,
            name=self.name,
            params=self.get_params()
        )


# 工厂函数
def create_top_k_mapper(k: int = 5, direction: int = 1) -> RankBasedMapper:
    """创建 Top K 映射器"""
    return RankBasedMapper(top_k=k, direction=direction, weight_method='equal')


def create_bottom_k_mapper(k: int = 5) -> RankBasedMapper:
    """创建 Bottom K 映射器"""
    return RankBasedMapper(top_k=k, direction=-1, weight_method='equal')


def create_equal_weight_mapper() -> DirMapper:
    """创建等权映射器"""
    return DirMapper(normalize=True, target_sum=1.0)
