"""
因子后处理流水线（Factor Preprocessing Pipeline）

提供可组合叠加的因子后处理步骤，用于合成因子前对子因子进行标准化处理。

支持的步骤（按推荐顺序）：
- 'industry_neutral' : 一级行业中性化（行业内 z-score）
- 'size_neutral'     : 市值中性化（OLS 回归残差）— 市值因子自动跳过
- 'zscore'           : 全市场截面 Z-score 标准化
- 'orthogonalize'    : 对称正交化（Σ^{-1/2}），作用于多因子整体

使用示例::

    pipeline = FactorPreprocessor(
        steps=['industry_neutral', 'size_neutral', 'zscore', 'orthogonalize'],
        size_factor_indices=[1],  # valuation_SIZE 在第1位，跳过市值中性化
    )

    # 准备辅助数据
    pipeline.set_industry_map(fundamental_data.get_industry_map())
    pipeline.set_market_cap(mc_values, mc_symbols)

    # 执行: features_nt (N, T, F) -> (N, T, F) 处理后
    features_processed = pipeline.transform(features_nt, symbols)
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# 单因子级步骤（作用于每个因子的 (N,T) 矩阵）
SINGLE_FACTOR_STEPS = {'industry_neutral', 'size_neutral', 'zscore'}

# 多因子级步骤（作用于 (N,T,F) 整体）
MULTI_FACTOR_STEPS = {'orthogonalize'}

ALL_STEPS = SINGLE_FACTOR_STEPS | MULTI_FACTOR_STEPS


class FactorPreprocessor:
    """可组合叠加的因子后处理流水线。

    Args:
        steps: 后处理步骤列表，按顺序执行。
               单因子步骤在前，多因子步骤在后。
               如果用户混排，会自动将多因子步骤移到最后并打印警告。
        size_factor_indices: 市值因子在 F 维度的索引集合，
                             这些因子跳过 'size_neutral' 步骤。
    """

    def __init__(
        self,
        steps: List[str],
        size_factor_indices: Optional[Set[int]] = None,
    ):
        for s in steps:
            if s not in ALL_STEPS:
                raise ValueError(
                    f"未知步骤 '{s}'，可选: {sorted(ALL_STEPS)}"
                )

        # 分离单因子步骤和多因子步骤，保持各自内部顺序
        self._single_steps = [s for s in steps if s in SINGLE_FACTOR_STEPS]
        self._multi_steps = [s for s in steps if s in MULTI_FACTOR_STEPS]

        # 检查是否有混排
        single_positions = [i for i, s in enumerate(steps) if s in SINGLE_FACTOR_STEPS]
        multi_positions = [i for i, s in enumerate(steps) if s in MULTI_FACTOR_STEPS]
        if single_positions and multi_positions and max(single_positions) > min(multi_positions):
            logger.warning(
                "多因子步骤(orthogonalize)应在单因子步骤之后，已自动调整顺序"
            )

        self._size_factor_indices: Set[int] = size_factor_indices or set()
        self._industry_map: Optional[Dict[str, str]] = None
        self._market_cap_arr: Optional[np.ndarray] = None
        self._market_cap_symbols: Optional[np.ndarray] = None

    def set_industry_map(self, industry_map: Dict[str, str]) -> 'FactorPreprocessor':
        """设置行业映射（industry_neutral 步骤需要）。"""
        self._industry_map = industry_map
        return self

    def set_market_cap(
        self,
        market_cap_arr: np.ndarray,
        market_cap_symbols: np.ndarray,
    ) -> 'FactorPreprocessor':
        """设置市值数据（size_neutral 步骤需要）。

        Args:
            market_cap_arr: (M, T) 市值面板
            market_cap_symbols: (M,) 对应的 symbol 数组
        """
        self._market_cap_arr = market_cap_arr
        self._market_cap_symbols = market_cap_symbols
        return self

    def transform(
        self,
        features_nt: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:
        """执行后处理流水线。

        Args:
            features_nt: (N, T, F) 因子面板，N=股票, T=日期, F=因子数
            symbols: (N,) 股票代码数组

        Returns:
            np.ndarray: (N, T, F) 处理后的因子面板
        """
        N, T, F = features_nt.shape
        result = features_nt.copy()

        # --- 单因子级步骤：逐因子处理 ---
        if self._single_steps:
            for f_idx in range(F):
                factor_slice = result[:, :, f_idx]  # (N, T)

                for step in self._single_steps:
                    if step == 'size_neutral' and f_idx in self._size_factor_indices:
                        logger.info(f"  因子[{f_idx}] 跳过 size_neutral（市值因子）")
                        continue
                    factor_slice = self._apply_single_step(
                        step, factor_slice, symbols, f_idx
                    )

                result[:, :, f_idx] = factor_slice

        # --- 多因子级步骤：整体处理 ---
        for step in self._multi_steps:
            if step == 'orthogonalize':
                result = _orthogonalize(result)

        return result

    def _apply_single_step(
        self,
        step: str,
        factor_arr: np.ndarray,
        symbols: np.ndarray,
        f_idx: int,
    ) -> np.ndarray:
        """执行单个单因子级后处理步骤。

        Args:
            step: 步骤名
            factor_arr: (N, T) 单因子面板
            symbols: (N,) 股票代码
            f_idx: 因子索引（用于日志）

        Returns:
            np.ndarray: (N, T) 处理后
        """
        if step == 'industry_neutral':
            if self._industry_map is None:
                raise ValueError("industry_neutral 需要先调用 set_industry_map()")
            return _neutralize_industry(factor_arr, symbols, self._industry_map)

        elif step == 'size_neutral':
            if self._market_cap_arr is None:
                raise ValueError("size_neutral 需要先调用 set_market_cap()")
            mc_aligned = _align_market_cap(
                self._market_cap_arr, self._market_cap_symbols, symbols
            )
            return _neutralize_size(factor_arr, mc_aligned)

        elif step == 'zscore':
            return _cross_sectional_zscore(factor_arr)

        else:
            raise ValueError(f"未知单因子步骤: {step}")


# ======================================================================
# 单因子级处理函数
# ======================================================================

def _neutralize_industry(
    factor_arr: np.ndarray,
    symbols: np.ndarray,
    industry_map: Dict[str, str],
) -> np.ndarray:
    """行业内 z-score 中性化。

    外层按行业循环（~30次），内层对全部 T 日向量化。
    行业内股票数 < 3 时保留原始值。

    Args:
        factor_arr: (N, T)
        symbols: (N,)
        industry_map: symbol -> industry

    Returns:
        (N, T) 行业中性化后
    """
    industries = np.array(
        [industry_map.get(s) for s in symbols], dtype=object
    )
    result = factor_arr.copy()

    valid_ind_mask = np.array([v is not None for v in industries])
    unique_industries = np.unique(industries[valid_ind_mask].astype(str))

    # Loop: ~30次（一级行业数量）
    for ind in unique_industries:
        mask = (industries == ind)
        group = factor_arr[mask, :]  # (n_group, T)

        n_valid = (~np.isnan(group)).sum(axis=0)  # (T,)
        mean = np.nanmean(group, axis=0)
        std = np.nanstd(group, axis=0, ddof=1)

        std_safe = np.where(std < 1e-10, 1.0, std)
        zscore = (group - mean) / std_safe

        # std 过小：非 NaN 处置 0
        zscore = np.where(
            std < 1e-10,
            np.where(np.isnan(group), np.nan, 0.0),
            zscore,
        )
        # n_valid < 3：保留原始值
        zscore = np.where(n_valid < 3, group, zscore)

        result[mask, :] = zscore

    return result


def _neutralize_size(
    factor_arr: np.ndarray,
    market_cap_arr: np.ndarray,
) -> np.ndarray:
    """市值中性化（OLS 残差）。

    每日截面 OLS: factor ~ log(market_cap)，取残差。
    有效样本 < 5 时不做中性化。

    Args:
        factor_arr: (N, T)
        market_cap_arr: (N, T) 已对齐

    Returns:
        (N, T) 市值中性化后
    """
    T = factor_arr.shape[1]
    result = factor_arr.copy()
    log_mc = np.log(np.where(market_cap_arr > 0, market_cap_arr, np.nan))

    # Loop: ~T次 (~2430 交易日)
    for t in range(T):
        y = factor_arr[:, t]
        mc = log_mc[:, t]

        valid = ~np.isnan(y) & ~np.isnan(mc)
        if valid.sum() < 5:
            continue

        y_valid = y[valid]
        mc_valid = mc[valid]

        X = np.column_stack([np.ones(len(mc_valid)), mc_valid])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
            result[valid, t] = y_valid - X @ coef
        except np.linalg.LinAlgError:
            pass  # OLS 失败时保留原始值

    return result


def _cross_sectional_zscore(factor_arr: np.ndarray) -> np.ndarray:
    """全市场截面 Z-score 标准化。

    每日截面内，对所有股票的因子值做 z-score = (x - mean) / std。
    有效样本 < 3 时不做标准化。

    Args:
        factor_arr: (N, T)

    Returns:
        (N, T) z-score 标准化后
    """
    mean = np.nanmean(factor_arr, axis=0, keepdims=True)   # (1, T)
    std = np.nanstd(factor_arr, axis=0, keepdims=True, ddof=1)  # (1, T)

    n_valid = (~np.isnan(factor_arr)).sum(axis=0, keepdims=True)  # (1, T)

    std_safe = np.where(std < 1e-10, 1.0, std)
    zscore = (factor_arr - mean) / std_safe

    # std 过小：非 NaN 处置 0
    zscore = np.where(
        std < 1e-10,
        np.where(np.isnan(factor_arr), np.nan, 0.0),
        zscore,
    )
    # n_valid < 3：保留原始值
    zscore = np.where(n_valid < 3, factor_arr, zscore)

    return zscore


# ======================================================================
# 多因子级处理函数
# ======================================================================

def _align_market_cap(
    market_cap_arr: np.ndarray,
    market_cap_symbols: np.ndarray,
    target_symbols: np.ndarray,
) -> np.ndarray:
    """将市值面板对齐到 target_symbols，缺失填 NaN。

    Args:
        market_cap_arr: (M, T)
        market_cap_symbols: (M,)
        target_symbols: (N,)

    Returns:
        (N, T) 对齐后
    """
    N = len(target_symbols)
    T = market_cap_arr.shape[1]
    result = np.full((N, T), np.nan, dtype=np.float64)

    mc_sym_map = {s: i for i, s in enumerate(market_cap_symbols)}
    for i, sym in enumerate(target_symbols):
        j = mc_sym_map.get(sym)
        if j is not None:
            result[i, :] = market_cap_arr[j, :]
    return result


def _orthogonalize(features_nt: np.ndarray) -> np.ndarray:
    """对称正交化（逐日截面）。

    每日截面内，对 F 个因子做对称正交化: X* = X @ Σ^{-1/2}。
    Σ 为因子间的相关矩阵，Σ^{-1/2} = P @ diag(1/√λ) @ P^T。

    优点：不偏向任何单个因子（对称性），消除因子间共线性。

    Args:
        features_nt: (N, T, F)

    Returns:
        (N, T, F) 正交化后
    """
    N, T, F = features_nt.shape
    result = features_nt.copy()

    # Loop: ~T次 (~2430 交易日)
    for t in range(T):
        X = features_nt[:, t, :]  # (N, F)
        valid = np.isfinite(X).all(axis=1)
        n_valid = valid.sum()

        if n_valid < F + 1:
            continue

        X_valid = X[valid]  # (n_valid, F)

        # 相关矩阵
        corr = np.corrcoef(X_valid.T)  # (F, F)
        if not np.isfinite(corr).all():
            continue

        # 特征分解: Σ = P D P^T
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)

        # Σ^{-1/2} = P @ diag(1/√λ) @ P^T
        inv_sqrt_D = np.diag(1.0 / np.sqrt(eigvals))
        ortho_matrix = eigvecs @ inv_sqrt_D @ eigvecs.T  # (F, F)

        # 变换
        result[valid, t, :] = X_valid @ ortho_matrix

    return result
