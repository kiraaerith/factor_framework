"""
OHLCV数据容器模块

定义 OHLCVData 类，用于封装 N × T 维度的OHLCV数据。
内部使用 5 个独立的 NumPy ndarray 存储，消除 pandas 元数据开销。
边界层通过 from_dataframe() 工厂方法转换。
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


class OHLCVData:
    """
    OHLCV数据容器类

    封装 N 个标的 × T 时间长度的 OHLCV 数据，5 个字段各存一个独立 ndarray。

    Attributes:
        open/high/low/close/volume: 各字段 ndarray shape (N, T), float64
        symbols: 标的数组，ndarray shape (N,)
        dates: 日期数组，ndarray shape (T,)

    Note:
        - NaN 表示缺失数据
        - volume 可以为 0（停牌）
    """

    COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化OHLCV数据容器

        Args:
            open/high/low/close/volume: shape (N, T), float64
            symbols: 标的数组，shape (N,)
            dates: 日期数组，shape (T,)
            metadata: 元数据字典，可选

        Raises:
            TypeError: 输入不是 ndarray
            ValueError: shape 不匹配或数据为空
        """
        arrays = {'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}
        for field, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{field} must be a numpy ndarray, got {type(arr)}")
            if arr.ndim != 2:
                raise ValueError(f"{field} must be 2D, got shape {arr.shape}")
            if arr.size == 0:
                raise ValueError(f"{field} cannot be empty")

        symbols = np.asarray(symbols)
        dates = np.asarray(dates)

        ref_shape = close.shape
        for field, arr in arrays.items():
            if arr.shape != ref_shape:
                raise ValueError(f"{field}.shape={arr.shape} != close.shape={ref_shape}")

        if ref_shape[0] != len(symbols):
            raise ValueError(f"shape[0]={ref_shape[0]} != len(symbols)={len(symbols)}")
        if ref_shape[1] != len(dates):
            raise ValueError(f"shape[1]={ref_shape[1]} != len(dates)={len(dates)}")

        self._open = open.astype(np.float64)
        self._high = high.astype(np.float64)
        self._low = low.astype(np.float64)
        self._close = close.astype(np.float64)
        self._volume = volume.astype(np.float64)
        self._symbols = symbols
        self._dates = dates
        self._metadata = metadata or {}
        self._n_assets = ref_shape[0]
        self._n_periods = ref_shape[1]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol_col: str = 'symbol',
        date_col: str = 'eob',
        ohlcv_cols: Optional[Dict[str, str]] = None
    ) -> 'OHLCVData':
        """
        从长格式 DataFrame 创建 OHLCVData（边界层使用，pivot 后立即转 numpy）

        输入格式示例：
            symbol      eob         open    high    low     close   volume
            SHSE.510300 2024-01-01  3.5     3.6     3.4     3.55    1000000

        Args:
            df: 长格式 DataFrame
            symbol_col: 标的列名
            date_col: 日期列名
            ohlcv_cols: OHLCV 列名映射
        """
        if ohlcv_cols is None:
            ohlcv_cols = {f: f for f in cls.COLUMNS}

        arrays = {}
        symbols = None
        dates = None

        for field, col_name in ohlcv_cols.items():
            pivoted = df.pivot(index=symbol_col, columns=date_col, values=col_name)
            if symbols is None:
                symbols = np.array(pivoted.index.tolist())
                dates = np.array(pivoted.columns.values, dtype="datetime64[ns]")
            arrays[field] = pivoted.values.astype(np.float64)

        return cls(
            open=arrays['open'],
            high=arrays['high'],
            low=arrays['low'],
            close=arrays['close'],
            volume=arrays['volume'],
            symbols=symbols,
            dates=dates
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def open(self) -> np.ndarray:
        """开盘价 (N, T)，返回拷贝"""
        return self._open.copy()

    @property
    def high(self) -> np.ndarray:
        """最高价 (N, T)，返回拷贝"""
        return self._high.copy()

    @property
    def low(self) -> np.ndarray:
        """最低价 (N, T)，返回拷贝"""
        return self._low.copy()

    @property
    def close(self) -> np.ndarray:
        """收盘价 (N, T)，返回拷贝"""
        return self._close.copy()

    @property
    def volume(self) -> np.ndarray:
        """成交量 (N, T)，返回拷贝"""
        return self._volume.copy()

    @property
    def symbols(self) -> np.ndarray:
        """标的数组 (N,)"""
        return self._symbols.copy()

    @property
    def dates(self) -> np.ndarray:
        """日期数组 (T,)"""
        return self._dates.copy()

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_periods(self) -> int:
        return self._n_periods

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(N, T, 5)"""
        return self._n_assets, self._n_periods, 5

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    # ------------------------------------------------------------------
    # 数据访问
    # ------------------------------------------------------------------

    def get_field(self, field: str) -> np.ndarray:
        """
        获取指定字段的 ndarray (N, T) 拷贝

        Args:
            field: 'open', 'high', 'low', 'close', 'volume' 之一
        """
        if field not in self.COLUMNS:
            raise ValueError(f"Unknown field: {field}, must be one of {self.COLUMNS}")
        return getattr(self, f'_{field}').copy()

    def get_cross_section(self, date) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一时点的横截面数据

        Returns:
            (data (N, 5), symbols (N,))，data 列顺序与 COLUMNS 一致
        """
        idx_arr = np.where(self._dates == date)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Date {date} not found in OHLCV data")
        t = idx_arr[0]
        data = np.column_stack([
            self._open[:, t], self._high[:, t], self._low[:, t],
            self._close[:, t], self._volume[:, t]
        ])
        return data, self._symbols.copy()

    def get_time_series(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取某一标的的时间序列数据

        Returns:
            (data (T, 5), dates (T,))，data 列顺序与 COLUMNS 一致
        """
        idx_arr = np.where(self._symbols == symbol)[0]
        if len(idx_arr) == 0:
            raise KeyError(f"Symbol {symbol} not found in OHLCV data")
        n = idx_arr[0]
        data = np.column_stack([
            self._open[n, :], self._high[n, :], self._low[n, :],
            self._close[n, :], self._volume[n, :]
        ])
        return data, self._dates.copy()

    # ------------------------------------------------------------------
    # 收益率计算（全 numpy 实现）
    # ------------------------------------------------------------------

    def get_returns(self, field: str = 'close', periods: int = 1) -> np.ndarray:
        """
        计算收益率（T 处存储 price[T] / price[T-periods] - 1）

        Args:
            field: 价格字段，默认 'close'
            periods: 计算周期，默认 1

        Returns:
            ndarray (N, T)，前 periods 列为 NaN
        """
        price = self.get_field(field)
        result = np.full_like(price, np.nan)
        result[:, periods:] = (price[:, periods:] - price[:, :-periods]) / price[:, :-periods]
        return result

    def get_open_returns(self, periods: int = 1) -> np.ndarray:
        """
        计算开盘价收益率，用于 A 股执行假设。

        列 T 存储 open[T+periods] / open[T] - 1，即在 T 日开盘买入、
        T+periods 日开盘卖出所实现的收益率。

        配合 delay=1 使用：position.shift(1) * get_open_returns()
        即为"T-1日信号 → T日开盘买入 → T+periods日开盘卖出"的组合收益。

        Args:
            periods: 持仓周期，默认 1（下一日开盘卖出）

        Returns:
            ndarray (N, T)，后 periods 列为 NaN
        """
        o = self._open
        result = np.full_like(o, np.nan)
        result[:, :-periods] = (o[:, periods:] - o[:, :-periods]) / o[:, :-periods]
        return result

    def get_log_returns(self, field: str = 'close', periods: int = 1) -> np.ndarray:
        """
        计算对数收益率

        Args:
            field: 价格字段，默认 'close'
            periods: 计算周期，默认 1

        Returns:
            ndarray (N, T)，前 periods 列为 NaN
        """
        price = self.get_field(field)
        result = np.full_like(price, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            result[:, periods:] = np.log(price[:, periods:] / price[:, :-periods])
        return result

    def get_vwap(self) -> np.ndarray:
        """典型价格 (High + Low + Close) / 3，shape (N, T)"""
        return (self._high + self._low + self._close) / 3.0

    def get_ohlc4(self) -> np.ndarray:
        """OHLC 均价 (Open + High + Low + Close) / 4，shape (N, T)"""
        return (self._open + self._high + self._low + self._close) / 4.0

    def get_true_range(self) -> np.ndarray:
        """
        真实波幅 True Range = max(high-low, |high-close_prev|, |low-close_prev|)

        Returns:
            ndarray (N, T)，第 0 列为 NaN（无前一收盘价）
        """
        h = self._high
        l = self._low
        c = self._close

        tr1 = h - l                                   # (N, T)
        tr2 = np.abs(h[:, 1:] - c[:, :-1])           # (N, T-1)
        tr3 = np.abs(l[:, 1:] - c[:, :-1])           # (N, T-1)

        result = np.full_like(h, np.nan)
        result[:, 1:] = np.maximum(tr1[:, 1:], np.maximum(tr2, tr3))
        return result

    # ------------------------------------------------------------------
    # 对齐与工具
    # ------------------------------------------------------------------

    def align_with(self, other: 'OHLCVData') -> 'OHLCVData':
        """
        与另一个 OHLCVData 对齐（取 symbols/dates 交集），用 np.searchsorted

        Args:
            other: 要对齐的 OHLCVData

        Returns:
            OHLCVData: 对齐后的新对象
        """
        common_symbols = np.intersect1d(self._symbols, other._symbols)
        common_dates = np.intersect1d(self._dates, other._dates)

        if len(common_symbols) == 0 or len(common_dates) == 0:
            raise ValueError("No common symbols or dates found for alignment")

        sym_idx = np.searchsorted(self._symbols, common_symbols)
        date_idx = np.searchsorted(self._dates, common_dates)

        def _slice(arr):
            return arr[sym_idx, :][:, date_idx]

        return OHLCVData(
            open=_slice(self._open),
            high=_slice(self._high),
            low=_slice(self._low),
            close=_slice(self._close),
            volume=_slice(self._volume),
            symbols=common_symbols,
            dates=common_dates,
            metadata=self._metadata.copy()
        )

    def copy(self) -> 'OHLCVData':
        """创建深拷贝"""
        return OHLCVData(
            open=self._open.copy(),
            high=self._high.copy(),
            low=self._low.copy(),
            close=self._close.copy(),
            volume=self._volume.copy(),
            symbols=self._symbols.copy(),
            dates=self._dates.copy(),
            metadata=self._metadata.copy()
        )

    def to_3d_array(self) -> np.ndarray:
        """
        转换为 3D numpy 数组 (N, T, 5)，维度顺序为 (symbol, date, field)
        field 顺序与 COLUMNS 一致：open/high/low/close/volume
        """
        return np.stack(
            [self._open, self._high, self._low, self._close, self._volume],
            axis=-1
        )

    def info(self) -> Dict[str, Any]:
        """获取 OHLCV 信息摘要"""
        syms = self._symbols.tolist()
        return {
            'shape': self.shape,
            'n_assets': self._n_assets,
            'n_periods': self._n_periods,
            'date_range': (self._dates[0], self._dates[-1]) if len(self._dates) > 0 else None,
            'symbols_preview': syms[:5] + ['...'] if len(syms) > 5 else syms,
            'metadata': self._metadata,
            'missing_ratio': {
                field: float(np.isnan(getattr(self, f'_{field}')).mean())
                for field in self.COLUMNS
            }
        }

    def __repr__(self) -> str:
        return f"OHLCVData(shape={self.shape})"
