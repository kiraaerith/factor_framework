"""
权重分配方法模块

提供等权重分配策略（numpy 实现）。
"""

import numpy as np


def equal_weight(selected: np.ndarray) -> np.ndarray:
    """
    等权重分配

    将总权重（默认 1.0）平均分配给所有选中的标的。

    Args:
        selected: bool ndarray (N, T)，True 表示该标的被选中

    Returns:
        np.ndarray: 权重矩阵 (N, T)，每列和为 1.0（无选中时全为 0）
    """
    selected_count = selected.sum(axis=0, keepdims=True).astype(float)  # (1, T)
    selected_count = np.where(selected_count == 0, np.nan, selected_count)
    weights = np.where(selected, 1.0 / selected_count, 0.0)
    return np.nan_to_num(weights, nan=0.0)


def normalize_weights(
    weights: np.ndarray,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    归一化权重

    将权重归一化，使每列绝对值之和为目标值。

    Args:
        weights: 权重矩阵 (N, T)
        target_sum: 目标权重和

    Returns:
        np.ndarray: 归一化后的权重矩阵 (N, T)
    """
    current_sum = np.nansum(np.abs(weights), axis=0, keepdims=True)  # (1, T)
    current_sum = np.where(current_sum == 0, np.nan, current_sum)
    normalized = weights / current_sum * target_sum
    return np.nan_to_num(normalized, nan=0.0)
