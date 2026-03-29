"""
因子数据容器模块

定义 FactorData 类，用于封装 N × T 维度的因子值数据。
内部使用 NumPy ndarray 存储，消除 pandas 元数据开销。
边界层通过 from_dataframe() 工厂方法转换。
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


class FactorData:
    """
    因子数据容器类

    封装 N 个标的 × T 时间长度的因子值矩阵（N × T）。
    内部使用 numpy ndarray 存储，symbols/dates 以 ndarray 形式持有。

    Attributes:
        values: 因子值矩阵，ndarray shape (N, T), float64
        symbols: 标的数组，ndarray shape (N,)
        dates: 日期数组，ndarray shape (T,) datetime64[ns]
        name: 因子名称
        factor_type: 因子类型
        params: 因子参数配置
    """

    def __init__(
        self,
        values: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        name: Optional[str] = None,
        factor_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化因子数据容器

        Args:
            values: 因子值矩阵，shape (N, T), float64
            symbols: 标的数组，shape (N,)
            dates: 日期数组，shape (T,), datetime64[ns]
            name: 因子名称，可选
            factor_type: 因子类型，可选
            params: 因子参数配置字典，可选

        Raises:
            TypeError: values 不是 ndarray
            ValueError: shape 不匹配或 values 为空
        """
        if not isinstance(values, np.ndarray):
            raise TypeError(f"values must be a numpy ndarray, got {type(values)}")

        values = values.astype(np.float64)

        if values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {values.shape}")

        if values.size == 0:
            raise ValueError("values cannot be empty")

        symbols = np.asarray(symbols)
        dates = np.asarray(dates)

        if values.shape[0] != len(symbols):
            raise ValueError(
                f"values.shape[0]={values.shape[0]} != len(symbols)={len(symbols)}"
            )
        if values.shape[1] != len(dates):
            raise ValueError(
                f"values.shape[1]={values.shape[1]} != len(dates)={len(dates)}"
            )

        self._values = values
        self._symbols = symbols
        self._dates = dates
        self._name = name or "UnknownFactor"
        self._factor_type = factor_type or name or "UnknownFactor"
        self._params = params or {}
        self._n_assets = values.shape[0]
        self._n_periods = values.shape[1]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: Optional[str] = None,
        factor_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> 'FactorData':
        """
        从 DataFrame 创建 FactorData（边界层使用，内部立即转 numpy）

        Args:
            df: DataFrame，index=symbols, columns=dates
        """
        values = df.values.astype(np.float64)
        symbols = np.array(df.index.tolist())
        dates = np.array(df.columns.tolist(), dtype='datetime64[ns]')
        return cls(values, symbols, dates, name=name, factor_type=factor_type, params=params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """获取因子值矩阵 (N, T)，返回拷贝"""
        return self._values.copy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def factor_type(self) -> str:
        return self._factor_type

    @factor_type.setter
    def factor_type(self, value: str):
        self._factor_type = value

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

    def _make_new(self, values: np.ndarray, name: str, factor_type: str) -> 'FactorData':
        """用新的 ndarray 构造 FactorData，共享 symbols/dates（不拷贝）"""
        return FactorData(
            values,
            self._symbols,
            self._dates,
            name=name,
            factor_type=factor_type,
            params=self._params
        )

    # ------------------------------------------------------------------
    # 数据访问
    # ------------------------------------------------------------------

    def get_cross_section(self, date) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一时点的横截面因子值

        Returns:
            (values (N,), symbols (N,))
        """
        date_np = np.datetime64(date, 'ns')
        idx_arr = np.where(self._dates == date_np)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Date {date} not found in factor data")
        return self._values[:, idx_arr[0]].copy(), self._symbols.copy()

    def get_time_series(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一标的的时间序列因子值

        Returns:
            (values (T,), dates (T,))
        """
        idx_arr = np.where(self._symbols == symbol)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Symbol {symbol} not found in factor data")
        return self._values[idx_arr[0], :].copy(), self._dates.copy()

    def to_numpy(self) -> np.ndarray:
        """返回因子值 ndarray (N, T) 的拷贝"""
        return self._values.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame（边界层展示用）"""
        return pd.DataFrame(
            self._values,
            index=self._symbols,
            columns=self._dates
        )

    # ------------------------------------------------------------------
    # 计算方法（全 numpy 实现）
    # ------------------------------------------------------------------

    def rank(self, axis: int = 1, ascending: bool = True) -> 'FactorData':
        """
        计算因子排名

        Args:
            axis: 0=横截面（每个日期对所有标的排名）
                  1=时间序列（每个标的对所有日期排名）
            ascending: True=升序（小值得低排名），False=降序
        """
        arr = self._values
        nan_mask = np.isnan(arr)

        # 用 inf 填 NaN，使其在 argsort 中排在末尾
        temp = arr.copy()
        temp[nan_mask] = np.inf if ascending else -np.inf

        ranks = np.argsort(np.argsort(temp, axis=axis), axis=axis) + 1
        ranks = ranks.astype(np.float64)
        ranks[nan_mask] = np.nan

        return self._make_new(
            ranks,
            name=f"{self._name}_rank",
            factor_type=f"{self._factor_type}_rank"
        )

    def zscore(self, axis: int = 1) -> 'FactorData':
        """
        Z-Score 标准化

        Args:
            axis: 0=横截面（每个日期标准化），1=时间序列（每个标的标准化）
        """
        arr = self._values
        mean = np.nanmean(arr, axis=axis, keepdims=True)
        std = np.nanstd(arr, axis=axis, keepdims=True)
        std = np.where(std < 1e-10, np.nan, std)
        result = (arr - mean) / std

        return self._make_new(
            result,
            name=f"{self._name}_zscore",
            factor_type=f"{self._factor_type}_zscore"
        )

    def demean(self, axis: int = 1) -> 'FactorData':
        """
        去均值处理

        Args:
            axis: 0=横截面，1=时间序列
        """
        arr = self._values
        mean = np.nanmean(arr, axis=axis, keepdims=True)
        result = arr - mean

        return self._make_new(
            result,
            name=f"{self._name}_demean",
            factor_type=f"{self._factor_type}_demean"
        )

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> 'FactorData':
        """截断异常值"""
        result = np.clip(self._values, lower, upper)

        return self._make_new(
            result,
            name=f"{self._name}_clipped",
            factor_type=f"{self._factor_type}_clipped"
        )

    def shift(self, periods: int = 1) -> 'FactorData':
        """
        沿时间轴（axis=1）平移

        Args:
            periods: 正数=向后平移（获取过去值），负数=向前平移
        """
        arr = self._values
        result = np.full_like(arr, np.nan)

        if periods > 0:
            result[:, periods:] = arr[:, :-periods]
        elif periods < 0:
            result[:, :periods] = arr[:, -periods:]
        else:
            result[:] = arr

        return self._make_new(
            result,
            name=f"{self._name}_shift{periods}",
            factor_type=f"{self._factor_type}_shift{periods}"
        )

    def fillna(self, method: str = 'forward', value: Optional[float] = None) -> 'FactorData':
        """
        填充缺失值

        Args:
            method: 'forward'（沿时间轴前向填充）/ 'backward'/ 'mean'/ 'value'
            value: method='value' 时的填充值
        """
        arr = self._values

        if method == 'forward':
            # 沿 axis=1（时间轴）forward-fill，使用 maximum.accumulate
            mask = np.isnan(arr)
            idx = np.where(~mask, np.arange(arr.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            result = arr[np.arange(arr.shape[0])[:, None], idx]

        elif method == 'backward':
            # 翻转 → forward-fill → 翻转
            arr_flip = arr[:, ::-1]
            mask = np.isnan(arr_flip)
            idx = np.where(~mask, np.arange(arr_flip.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            result = arr_flip[np.arange(arr_flip.shape[0])[:, None], idx][:, ::-1]

        elif method == 'mean':
            # 每个标的用其自身时间序列均值填充
            row_mean = np.nanmean(arr, axis=1, keepdims=True)
            result = np.where(np.isnan(arr), row_mean, arr)

        elif method == 'value':
            result = np.where(np.isnan(arr), value, arr)

        else:
            raise ValueError(f"Unknown fill method: {method}")

        return self._make_new(
            result,
            name=self._name,
            factor_type=self._factor_type
        )

    def copy(self) -> 'FactorData':
        """创建深拷贝"""
        return FactorData(
            self._values.copy(),
            self._symbols.copy(),
            self._dates.copy(),
            name=self._name,
            factor_type=self._factor_type,
            params=self._params.copy()
        )

    def info(self) -> Dict[str, Any]:
        """获取因子信息摘要"""
        return {
            'name': self._name,
            'factor_type': self._factor_type,
            'shape': self.shape,
            'n_assets': self._n_assets,
            'n_periods': self._n_periods,
            'date_range': (self._dates[0], self._dates[-1]) if len(self._dates) > 0 else None,
            'params': self._params,
            'missing_ratio': float(np.isnan(self._values).mean()),
        }

    def __repr__(self) -> str:
        return f"FactorData(name='{self._name}', shape={self.shape})"

    def __getitem__(self, key):
        """支持整数/切片索引访问内部 ndarray"""
        return self._values[key]
