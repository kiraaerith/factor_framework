"""
通用 Label 生成器（Label Generator）

基于股票未来收益率生成训练 label，支持可组合叠加的后处理步骤。

生成流程：
1. 从收盘价计算未来 N 日收益率（N = forward_days，通常与调仓频率一致）
2. 按顺序执行后处理步骤

支持的后处理步骤：
- 'raw'               : 不做处理，使用原始未来收益率
- 'zscore'            : 全市场截面 Z-score 标准化
- 'industry_neutral'  : 一级行业中性化（行业内 z-score）
- 'rank'              : 全市场截面百分位 rank，映射到 [0, 1]

步骤可自由组合叠加，例如：
- ['industry_neutral', 'rank']  : 先行业中性化再 rank
- ['zscore']                    : 仅 z-score
- ['industry_neutral', 'zscore']: 先行业中性化再全市场 z-score
- [] 或 ['raw']                 : 不做处理

使用示例::

    gen = LabelGenerator(forward_days=20, steps=['industry_neutral', 'rank'])
    gen.set_industry_map(fundamental_data.get_industry_map())

    # close_arr: (N, T), symbols: (N,)
    labels = gen.generate(close_arr, symbols)
    # labels: (N, T), 最后 forward_days 列为 NaN
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ALL_LABEL_STEPS = {'raw', 'zscore', 'industry_neutral', 'rank'}


class LabelGenerator:
    """通用 label 生成器。

    Args:
        forward_days: 未来收益率窗口（交易日），通常与调仓频率一致。
        steps: 后处理步骤列表，按顺序执行。
               'raw' 表示不做处理（若与其他步骤共存则被忽略）。
    """

    def __init__(
        self,
        forward_days: int = 20,
        steps: Optional[List[str]] = None,
    ):
        if steps is None:
            steps = ['raw']

        for s in steps:
            if s not in ALL_LABEL_STEPS:
                raise ValueError(
                    f"未知 label 步骤 '{s}'，可选: {sorted(ALL_LABEL_STEPS)}"
                )

        # 过滤掉 'raw'（它只是占位符）
        self._steps = [s for s in steps if s != 'raw']
        self._forward_days = forward_days
        self._industry_map: Optional[Dict[str, str]] = None

    def set_industry_map(self, industry_map: Dict[str, str]) -> 'LabelGenerator':
        """设置行业映射（industry_neutral 步骤需要）。"""
        self._industry_map = industry_map
        return self

    def generate(
        self,
        close_arr: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:
        """生成 label。

        Args:
            close_arr: (N, T) 复权收盘价面板
            symbols: (N,) 股票代码数组

        Returns:
            np.ndarray: (N, T) label 面板，最后 forward_days 列为 NaN
        """
        labels = _compute_forward_returns(close_arr, self._forward_days)

        for step in self._steps:
            if step == 'zscore':
                labels = _cross_sectional_zscore(labels)
            elif step == 'industry_neutral':
                if self._industry_map is None:
                    raise ValueError(
                        "industry_neutral 需要先调用 set_industry_map()"
                    )
                labels = _neutralize_industry(labels, symbols, self._industry_map)
            elif step == 'rank':
                labels = _cross_sectional_rank(labels)

        return labels

    def generate_from_returns(
        self,
        forward_returns: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:
        """从已有的未来收益率生成 label（跳过收益率计算，仅做后处理）。

        适用于已经计算好未来收益率的场景。

        Args:
            forward_returns: (N, T) 未来收益率面板
            symbols: (N,) 股票代码数组

        Returns:
            np.ndarray: (N, T) 处理后的 label
        """
        labels = forward_returns.copy()

        for step in self._steps:
            if step == 'zscore':
                labels = _cross_sectional_zscore(labels)
            elif step == 'industry_neutral':
                if self._industry_map is None:
                    raise ValueError(
                        "industry_neutral 需要先调用 set_industry_map()"
                    )
                labels = _neutralize_industry(labels, symbols, self._industry_map)
            elif step == 'rank':
                labels = _cross_sectional_rank(labels)

        return labels


# ======================================================================
# 底层计算函数
# ======================================================================

def _compute_forward_returns(close: np.ndarray, days: int) -> np.ndarray:
    """计算未来 N 日收益率。

    Args:
        close: (N, T) 复权收盘价
        days: 未来天数

    Returns:
        (N, T) 未来收益率，最后 days 列为 NaN
    """
    N, T = close.shape
    fwd = np.full((N, T), np.nan)
    if T <= days:
        return fwd
    with np.errstate(divide='ignore', invalid='ignore'):
        fwd[:, :-days] = close[:, days:] / close[:, :-days] - 1.0
    return fwd


def _cross_sectional_zscore(arr: np.ndarray) -> np.ndarray:
    """全市场截面 Z-score。

    每日截面 z-score = (x - mean) / std。
    有效样本 < 3 时保留原始值。NaN 保持不变。

    Args:
        arr: (N, T)

    Returns:
        (N, T)
    """
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True, ddof=1)
    n_valid = (~np.isnan(arr)).sum(axis=0, keepdims=True)

    std_safe = np.where(std < 1e-10, 1.0, std)
    zscore = (arr - mean) / std_safe

    zscore = np.where(
        std < 1e-10,
        np.where(np.isnan(arr), np.nan, 0.0),
        zscore,
    )
    zscore = np.where(n_valid < 3, arr, zscore)

    return zscore


def _neutralize_industry(
    arr: np.ndarray,
    symbols: np.ndarray,
    industry_map: Dict[str, str],
) -> np.ndarray:
    """行业内 z-score 中性化。

    外层按行业循环（~30次），内层对全部 T 日向量化。
    行业内股票数 < 3 时保留原始值。

    Args:
        arr: (N, T)
        symbols: (N,)
        industry_map: symbol -> industry

    Returns:
        (N, T)
    """
    industries = np.array(
        [industry_map.get(s) for s in symbols], dtype=object
    )
    result = arr.copy()

    valid_ind_mask = np.array([v is not None for v in industries])
    unique_industries = np.unique(industries[valid_ind_mask].astype(str))

    # Loop: ~30次（一级行业数量）
    for ind in unique_industries:
        mask = (industries == ind)
        group = arr[mask, :]

        n_valid = (~np.isnan(group)).sum(axis=0)
        mean = np.nanmean(group, axis=0)
        std = np.nanstd(group, axis=0, ddof=1)

        std_safe = np.where(std < 1e-10, 1.0, std)
        zscore = (group - mean) / std_safe

        zscore = np.where(
            std < 1e-10,
            np.where(np.isnan(group), np.nan, 0.0),
            zscore,
        )
        zscore = np.where(n_valid < 3, group, zscore)

        result[mask, :] = zscore

    return result


def _cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """全市场截面百分位 rank，映射到 [0, 1]。

    每日截面内对所有股票排序，rank = 排名 / (有效数 - 1)。
    有效样本 < 2 时该日保留 NaN。

    Args:
        arr: (N, T)

    Returns:
        (N, T) 值域 [0, 1]
    """
    N, T = arr.shape
    result = np.full_like(arr, np.nan)

    # Loop: ~T次 (~2430 交易日)
    for t in range(T):
        col = arr[:, t]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid < 2:
            continue

        order = np.argsort(col[valid])
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        ranks /= (n_valid - 1)
        result[valid, t] = ranks

    return result
