"""
基本面因子中性化模块

提供三种中性化方式：
- 'raw'      : 不做中性化，直接返回原始因子值
- 'industry' : 行业内 z-score（每日截面内，同行业股票单独标准化）
- 'size'     : 市值中性化（每日截面内，OLS 回归剔除 log 市值影响，取残差）

使用示例::

    from neutralization import apply_neutralization

    neutral_arr = apply_neutralization(
        factor_arr=raw_factor_arr,
        symbols=symbols,
        method='industry',
        industry_map=fd.get_industry_map(),
    )
"""

import warnings
from typing import Dict, Optional

import numpy as np


def apply_neutralization(
    factor_arr: np.ndarray,
    symbols: np.ndarray,
    method: str,
    industry_map: Optional[Dict[str, str]] = None,
    market_cap_arr: Optional[np.ndarray] = None,
    market_cap_symbols: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    对因子值做截面中性化处理。

    Args:
        factor_arr: 原始因子面板，shape (N, T), float64
        symbols: 标的数组，shape (N,)，与 factor_arr 行对应
        method: 中性化方式，'raw' / 'industry' / 'size'
        industry_map: symbol -> industry 映射（method='industry' 时必须提供）
        market_cap_arr: 市值面板，shape (M, T)（method='size' 时必须提供）
        market_cap_symbols: 市值面板对应的 symbol 数组，shape (M,)

    Returns:
        np.ndarray: 中性化后的因子面板，shape (N, T)
    """
    if method == 'raw':
        return factor_arr.copy()
    elif method == 'industry':
        if industry_map is None:
            raise ValueError("method='industry' 时必须提供 industry_map")
        return _neutralize_industry_numpy(factor_arr, symbols, industry_map)
    elif method == 'size':
        if market_cap_arr is None:
            raise ValueError("method='size' 时必须提供 market_cap_arr")
        # 如果提供了 market_cap_symbols，先对齐到 factor 的 symbols
        if market_cap_symbols is not None:
            mc_aligned = _align_market_cap(market_cap_arr, market_cap_symbols, symbols)
        else:
            mc_aligned = market_cap_arr
        return _neutralize_size_numpy(factor_arr, mc_aligned)
    else:
        raise ValueError(f"未知中性化方式: {method}，请使用 'raw' / 'industry' / 'size'")


def _align_market_cap(
    market_cap_arr: np.ndarray,
    market_cap_symbols: np.ndarray,
    target_symbols: np.ndarray,
) -> np.ndarray:
    """
    将市值面板对齐到 target_symbols，缺失的 symbol 填 NaN。

    Args:
        market_cap_arr: (M, T)
        market_cap_symbols: (M,)
        target_symbols: (N,)

    Returns:
        np.ndarray: (N, T)，按 target_symbols 顺序对齐
    """
    N = len(target_symbols)
    T = market_cap_arr.shape[1]
    result = np.full((N, T), np.nan, dtype=np.float64)

    mc_sym_set = {s: i for i, s in enumerate(market_cap_symbols)}
    for i, sym in enumerate(target_symbols):
        j = mc_sym_set.get(sym)
        if j is not None:
            result[i, :] = market_cap_arr[j, :]
    return result


# ------------------------------------------------------------------
# 行业中性化（翻转循环：外层 ~30 个行业，内层向量化 T 日）
# ------------------------------------------------------------------

def _neutralize_industry_numpy(
    factor_arr: np.ndarray,
    symbols: np.ndarray,
    industry_map: Dict[str, str],
) -> np.ndarray:
    """
    行业内 z-score 中性化（翻转循环向量化）。

    外层按行业循环（~30次），内层对全部 T 日向量化，
    相比旧版「外层 T 日、内层行业」减少循环次数约 T/30 倍。

    行业内股票数 < 3 时，该行业股票保留原始值（不中性化）。

    Args:
        factor_arr: (N, T) float64
        symbols: (N,)
        industry_map: symbol -> industry 映射

    Returns:
        np.ndarray: (N, T) 行业中性化后的因子值
    """
    industries = np.array([industry_map.get(s) for s in symbols], dtype=object)
    result = factor_arr.copy()

    # 过滤掉 None 行业
    valid_ind_mask = np.array([v is not None for v in industries])
    unique_industries = np.unique(industries[valid_ind_mask].astype(str))

    for ind in unique_industries:
        mask = (industries == ind)          # (N,) bool
        group = factor_arr[mask, :]         # (n_group, T)

        n_valid = (~np.isnan(group)).sum(axis=0)  # (T,) — 每日有效样本数

        mean = np.nanmean(group, axis=0)    # (T,)
        std = np.nanstd(group, axis=0, ddof=1)  # (T,)

        # 保护除零：std 过小时分母设为 1，后续用 where 覆盖结果
        std_safe = np.where(std < 1e-10, 1.0, std)
        zscore = (group - mean) / std_safe  # (n_group, T)，NaN 在 group NaN 处天然保留

        # std < 1e-10：非 NaN 处置 0，NaN 处保留 NaN
        zscore = np.where(
            std < 1e-10,
            np.where(np.isnan(group), np.nan, 0.0),
            zscore,
        )

        # n_valid < 3：该日保留原始值（不中性化）
        zscore = np.where(n_valid < 3, group, zscore)

        result[mask, :] = zscore

    return result


# ------------------------------------------------------------------
# 市值中性化（逐日 OLS，消除 DataFrame 开销）
# ------------------------------------------------------------------

def _neutralize_size_numpy(
    factor_arr: np.ndarray,
    market_cap_arr: np.ndarray,
) -> np.ndarray:
    """
    市值中性化（OLS 回归残差），逐日 OLS，消除 DataFrame 开销。

    每日截面内，以 log(市值) 为自变量，对因子值做 OLS 回归，取残差。
    有效样本数 < 5 时不做中性化。

    Args:
        factor_arr: (N, T) float64
        market_cap_arr: (N, T) float64，已对齐到 factor_arr 的 symbols/dates

    Returns:
        np.ndarray: (N, T) 市值中性化后的因子值
    """
    T = factor_arr.shape[1]
    result = factor_arr.copy()

    log_mc = np.log(np.where(market_cap_arr > 0, market_cap_arr, np.nan))  # (N, T)

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
            residual = y_valid - X @ coef
            result[valid, t] = residual
        except Exception as e:
            warnings.warn(f"市值中性化 OLS 失败 (t={t}): {e}")

    return result
