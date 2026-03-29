"""
仓位数据容器模块

定义 PositionData 类，用于封装 N × T 维度的仓位数据。
内部使用 NumPy ndarray 存储，消除 pandas 元数据开销。
边界层通过 from_dataframe() 工厂方法转换。
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


class PositionData:
    """
    仓位数据容器类

    封装 N 个标的 × T 时间长度的仓位权重矩阵（N × T）。
    内部使用 numpy ndarray 存储，symbols/dates 以 ndarray 形式持有。

    Attributes:
        weights: 仓位权重矩阵，ndarray shape (N, T), float64
        symbols: 标的数组，ndarray shape (N,)
        dates: 日期数组，ndarray shape (T,)
        name: 仓位策略名称
        params: 仓位参数配置

    Note:
        - 权重总和通常在0到1之间（多头）或-1到0之间（空头）
        - NaN 表示无仓位信息，0 表示明确空仓
        - 权重可以为负，表示做空
    """

    def __init__(
        self,
        weights: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化仓位数据容器

        Args:
            weights: 仓位权重矩阵，shape (N, T), float64
            symbols: 标的数组，shape (N,)
            dates: 日期数组，shape (T,)
            name: 仓位策略名称，可选
            params: 仓位参数配置字典，可选

        Raises:
            TypeError: weights 不是 ndarray
            ValueError: shape 不匹配或 weights 为空
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"weights must be a numpy ndarray, got {type(weights)}")

        weights = weights.astype(np.float64)

        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D, got shape {weights.shape}")

        if weights.size == 0:
            raise ValueError("weights cannot be empty")

        symbols = np.asarray(symbols)
        dates = np.asarray(dates)

        if weights.shape[0] != len(symbols):
            raise ValueError(
                f"weights.shape[0]={weights.shape[0]} != len(symbols)={len(symbols)}"
            )
        if weights.shape[1] != len(dates):
            raise ValueError(
                f"weights.shape[1]={weights.shape[1]} != len(dates)={len(dates)}"
            )

        self._weights = weights
        self._symbols = symbols
        self._dates = dates
        self._name = name or "UnknownPosition"
        self._params = params or {}
        self._n_assets = weights.shape[0]
        self._n_periods = weights.shape[1]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> 'PositionData':
        """
        从 DataFrame 创建 PositionData（边界层使用，内部立即转 numpy）

        Args:
            df: DataFrame，index=symbols, columns=dates
        """
        weights = df.values.astype(np.float64)
        symbols = np.array(df.index.tolist())
        dates = np.asarray(df.columns.tolist())
        return cls(weights, symbols, dates, name=name, params=params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """获取仓位权重矩阵 (N, T)，返回拷贝"""
        return self._weights.copy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def params(self) -> Dict[str, Any]:
        return self._params.copy()

    @property
    def symbols(self) -> np.ndarray:
        """获取标的数组 (N,)"""
        return self._symbols.copy()

    @property
    def dates(self) -> np.ndarray:
        """获取日期数组 (T,)"""
        return self._dates.copy()

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_periods(self) -> int:
        return self._n_periods

    @property
    def shape(self) -> tuple:
        """(N, T)"""
        return self._n_assets, self._n_periods

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _make_new(self, weights: np.ndarray, name: str) -> 'PositionData':
        """用新的 ndarray 构造 PositionData，共享 symbols/dates（不拷贝）"""
        return PositionData(
            weights,
            self._symbols,
            self._dates,
            name=name,
            params=self._params
        )

    # ------------------------------------------------------------------
    # 数据访问
    # ------------------------------------------------------------------

    def get_cross_section(self, date) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一时点的横截面仓位

        Returns:
            (weights (N,), symbols (N,))
        """
        idx_arr = np.where(self._dates == date)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Date {date} not found in position data")
        return self._weights[:, idx_arr[0]].copy(), self._symbols.copy()

    def get_time_series(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一标的的时间序列仓位

        Returns:
            (weights (T,), dates (T,))
        """
        idx_arr = np.where(self._symbols == symbol)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Symbol {symbol} not found in position data")
        return self._weights[idx_arr[0], :].copy(), self._dates.copy()

    def get_total_weights(self) -> np.ndarray:
        """
        获取每个时点的总仓位权重

        Returns:
            ndarray (T,): 每个时点的权重总和
        """
        return np.nansum(self._weights, axis=0)

    def get_active_positions(self) -> np.ndarray:
        """
        获取有仓位的标记矩阵

        Returns:
            ndarray (N, T) bool: True 表示有仓位
        """
        return np.abs(self._weights) > 1e-10

    def get_position_count(self) -> np.ndarray:
        """
        获取每个时点的持仓数量

        Returns:
            ndarray (T,): 每个时点的持仓数量
        """
        return self.get_active_positions().sum(axis=0)

    def to_numpy(self) -> np.ndarray:
        """返回仓位权重 ndarray (N, T) 的拷贝"""
        return self._weights.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame（边界层展示用）"""
        return pd.DataFrame(
            self._weights,
            index=self._symbols,
            columns=self._dates
        )

    # ------------------------------------------------------------------
    # 计算方法（全 numpy 实现）
    # ------------------------------------------------------------------

    def normalize(self, target_sum: float = 1.0) -> 'PositionData':
        """
        按横截面（逐日）归一化仓位权重，使绝对权重之和等于 target_sum

        Args:
            target_sum: 目标权重总和

        Returns:
            PositionData: 归一化后的新对象
        """
        arr = self._weights
        col_sum = np.nansum(np.abs(arr), axis=0, keepdims=True)
        col_sum = np.where(col_sum < 1e-10, np.nan, col_sum)
        result = arr / col_sum * target_sum
        return self._make_new(result, name=f"{self._name}_normalized")

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> 'PositionData':
        """
        截断仓位权重

        Args:
            lower: 仓位下限
            upper: 仓位上限

        Returns:
            PositionData: 截断后的新对象
        """
        result = np.clip(self._weights, lower, upper)
        return self._make_new(result, name=f"{self._name}_clipped")

    def shift(self, periods: int = 1) -> 'PositionData':
        """
        沿时间轴（axis=1）平移（用于实现调仓延迟）

        Args:
            periods: 正数=向后平移，负数=向前平移

        Returns:
            PositionData: 平移后的新对象
        """
        arr = self._weights
        result = np.full_like(arr, np.nan)

        if periods > 0:
            result[:, periods:] = arr[:, :-periods]
        elif periods < 0:
            result[:, :periods] = arr[:, -periods:]
        else:
            result[:] = arr

        return self._make_new(result, name=f"{self._name}_shift{periods}")

    def fillna(self, value: float = 0.0) -> 'PositionData':
        """
        填充缺失值

        Args:
            value: 填充值，默认为0（空仓）

        Returns:
            PositionData: 填充后的新对象
        """
        result = np.where(np.isnan(self._weights), value, self._weights)
        return self._make_new(result, name=self._name)

    def apply_mask(self, mask: np.ndarray, fill_value: float = np.nan) -> 'PositionData':
        """
        应用遮罩过滤仓位

        Args:
            mask: 布尔 ndarray (N, T)，True 表示保留
            fill_value: 被遮罩位置的填充值，默认 NaN

        Returns:
            PositionData: 遮罩后的新对象
        """
        result = np.where(mask, self._weights, fill_value)
        return self._make_new(result, name=f"{self._name}_masked")

    def copy(self) -> 'PositionData':
        """创建深拷贝"""
        return PositionData(
            self._weights.copy(),
            self._symbols.copy(),
            self._dates.copy(),
            name=self._name,
            params=self._params.copy()
        )

    def info(self) -> Dict[str, Any]:
        """获取仓位信息摘要"""
        total_weights = self.get_total_weights()
        active_count = self.get_position_count()

        return {
            'name': self._name,
            'shape': self.shape,
            'n_assets': self._n_assets,
            'n_periods': self._n_periods,
            'date_range': (self._dates[0], self._dates[-1]) if len(self._dates) > 0 else None,
            'params': self._params,
            'avg_position_count': float(active_count.mean()),
            'avg_total_weight': float(np.nanmean(total_weights)),
            'max_total_weight': float(np.nanmax(total_weights)),
            'min_total_weight': float(np.nanmin(total_weights)),
            'missing_ratio': float(np.isnan(self._weights).mean()),
        }

    def __repr__(self) -> str:
        return f"PositionData(name='{self._name}', shape={self.shape})"

    def __getitem__(self, key):
        """支持整数/切片索引访问内部 ndarray"""
        return self._weights[key]
