"""
A股数据加载模块

从Tushare SQLite数据库加载A股行情数据，返回框架所需的OHLCVData和TradeContext。

数据库路径：
  - Tushare SQLite: ../china_stock_data/tushare.db

数据表：
  - daily_hfq: 后复权日频OHLCV
  - daily: 不复权日频OHLCV + 日期×股票网格
  - suspend_d: 每日停复牌信息
  - stock_basic: 股票基础信息含上市/退市日期
  - stk_limit: 涨跌停价
  - stock_st: ST标记
"""

import warnings
import sqlite3
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_TUSHARE_DB_PATH = str(Path(__file__).resolve().parents[2].parent / "china_stock_data" / "tushare.db")

from core.ohlcv_data import OHLCVData


class StockDataLoader:
    """
    A股数据加载器

    从 Tushare SQLite 数据库加载 A 股行情及交易状态数据。
    所有返回的矩阵均为 numpy ndarray，pandas 仅用于 SQL 查询边界层。

    使用示例::

        loader = StockDataLoader()
        ohlcv = loader.load_ohlcv('2020-01-01', '2024-12-31', use_adjusted=True)
        raw_open_arr, symbols, dates = loader.load_raw_open('2020-01-01', '2024-12-31')
        trade_ctx = loader.load_trade_context(
            '2020-01-01', '2024-12-31',
            raw_open_arr=raw_open_arr, symbols=symbols, dates=dates
        )
    """

    DEFAULT_TUSHARE_DB_PATH = _DEFAULT_TUSHARE_DB_PATH

    def __init__(self, tushare_db_path: str = None):
        """
        Args:
            tushare_db_path: Tushare SQLite数据库路径，默认使用 DEFAULT_TUSHARE_DB_PATH
        """
        self.tushare_db_path = tushare_db_path or self.DEFAULT_TUSHARE_DB_PATH

    def close(self):
        """No-op, kept for backward compatibility (SQLite connections are per-query now)"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def load_ohlcv(
        self,
        start_date: str,
        end_date: str,
        use_adjusted: bool = True,
        symbols: Optional[List[str]] = None,
        lookback_extra_days: int = 0,
    ) -> OHLCVData:
        """
        加载OHLCV数据，返回OHLCVData对象（内部 ndarray）

        数据源: tushare daily_hfq (后复权) 或 daily (不复权)

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_adjusted: 是否使用后复权数据，默认True
            symbols: 股票代码列表（None=全市场，juejin 格式如 SHSE.600000）
            lookback_extra_days: 为因子回望窗口额外预留的日历天数

        Returns:
            OHLCVData: OHLCV数据容器（内部 ndarray）
        """
        table = "daily_hfq" if use_adjusted else "daily"

        if lookback_extra_days > 0:
            actual_start = (
                pd.Timestamp(start_date) - pd.Timedelta(days=lookback_extra_days)
            ).strftime("%Y%m%d")
        else:
            actual_start = pd.Timestamp(start_date).strftime("%Y%m%d")

        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date, open, high, low, close, vol AS volume
            FROM {table}
            WHERE trade_date BETWEEN '{actual_start}' AND '{end_str}'
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            raise ValueError(
                f"No data found for date range {actual_start} to {end_str} in tushare {table}"
            )

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        if symbols is not None:
            symbol_set = set(symbols)
            df = df[df["symbol"].isin(symbol_set)]

        # tushare daily_hfq 停牌日有行但 OHLCV 为 NULL，过滤掉
        df = df.dropna(subset=["open", "close"])

        # tushare vol 单位是"手"(1手=100股)，转换为"股"以保持与原掘金数据一致
        df["volume"] = df["volume"] * 100

        n_symbols = df["symbol"].nunique()
        n_dates = df["trade_date"].nunique()
        print(f"  - 加载OHLCV({'后复权' if use_adjusted else '不复权'}): "
              f"{n_symbols} 只股票 × {n_dates} 个交易日 (数据源: tushare {table})")
        self._estimate_memory(n_symbols, n_dates)

        ohlcv = OHLCVData.from_dataframe(
            df,
            symbol_col="symbol",
            date_col="trade_date",
            ohlcv_cols={
                "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            },
        )
        return ohlcv

    def load_raw_open(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载不复权开盘价（专用于涨跌停判断），数据源: tushare daily

        Returns:
            (raw_open: ndarray (N, T), symbols: ndarray (N,), dates: ndarray (T,))
        """
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date, open AS raw_open
            FROM daily
            WHERE trade_date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        if symbols is not None:
            symbol_set = set(symbols)
            df = df[df["symbol"].isin(symbol_set)]

        pivoted = df.pivot(index="symbol", columns="trade_date", values="raw_open")
        symbols_arr = np.array(pivoted.index.tolist())
        dates_arr = np.array(pivoted.columns.values, dtype="datetime64[ns]")
        raw_open_arr = pivoted.values.astype(np.float64)

        return raw_open_arr, symbols_arr, dates_arr

    @staticmethod
    def _jq_to_ts(symbol: str) -> str:
        """juejin code (SHSE.600000) -> tushare code (600000.SH)"""
        prefix, code = symbol.split(".")
        market = "SH" if prefix == "SHSE" else "SZ"
        return f"{code}.{market}"

    @staticmethod
    def _ts_to_jq(ts_code: str) -> str:
        """tushare code (600000.SH) -> juejin code (SHSE.600000)"""
        code, market = ts_code.split(".")
        prefix = "SHSE" if market == "SH" else "SZSE"
        return f"{prefix}.{code}"

    def _load_tushare_suspend(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        从 tushare suspend_d 加载停牌数据

        Returns:
            DataFrame: columns=[symbol, trade_date, is_suspended]
            symbol 为 juejin 格式，trade_date 为 datetime，is_suspended 全为 True
        """
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date
            FROM suspend_d
            WHERE suspend_type = 'S'
              AND trade_date BETWEEN '{start_str}' AND '{end_str}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "is_suspended"])

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["is_suspended"] = True
        return df[["symbol", "trade_date", "is_suspended"]]

    def _load_tushare_delist(self) -> pd.DataFrame:
        """
        从 tushare stock_basic 加载退市日期

        Returns:
            DataFrame: columns=[symbol, delisted_date]
            symbol 为 juejin 格式，delisted_date 为 datetime
        """
        conn = sqlite3.connect(self.tushare_db_path)
        query = """
            SELECT ts_code, delist_date
            FROM stock_basic
            WHERE delist_date IS NOT NULL AND delist_date != ''
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=["symbol", "delisted_date"])

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["delisted_date"] = pd.to_datetime(df["delist_date"])
        return df[["symbol", "delisted_date"]]

    def _load_tushare_list_date(self) -> pd.DataFrame:
        """
        从 tushare stock_basic 加载上市日期

        Returns:
            DataFrame: columns=[symbol, listed_date]
            symbol 为 juejin 格式，listed_date 为 datetime
        """
        conn = sqlite3.connect(self.tushare_db_path)
        query = """
            SELECT ts_code, list_date
            FROM stock_basic
            WHERE list_date IS NOT NULL AND list_date != ''
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=["symbol", "listed_date"])

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["listed_date"] = pd.to_datetime(df["list_date"])
        return df[["symbol", "listed_date"]]

    def _load_tushare_st(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        从 tushare stock_st 加载每日 ST 股票列表

        Returns:
            DataFrame: columns=[symbol, trade_date, is_st]
            symbol 为 juejin 格式，trade_date 为 datetime，is_st 全为 True
        """
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date
            FROM stock_st
            WHERE trade_date BETWEEN '{start_str}' AND '{end_str}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "is_st"])

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["is_st"] = True
        return df[["symbol", "trade_date", "is_st"]]

    def _load_tushare_stk_limit(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        从 tushare stk_limit 加载涨跌停价

        Returns:
            DataFrame: columns=[symbol, trade_date, upper_limit, lower_limit]
            symbol 为 juejin 格式，trade_date 为 datetime
        """
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date, up_limit, down_limit
            FROM stk_limit
            WHERE trade_date BETWEEN '{start_str}' AND '{end_str}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(
                columns=["symbol", "trade_date", "upper_limit", "lower_limit"]
            )

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.rename(
            columns={"up_limit": "upper_limit", "down_limit": "lower_limit"},
            inplace=True,
        )
        return df[["symbol", "trade_date", "upper_limit", "lower_limit"]]

    def load_trade_status(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        加载交易状态数据（停牌、涨跌停价、ST、上市退市日期等）

        数据源：
          - 停牌(is_suspended): tushare suspend_d
          - 退市(delisted_date): tushare stock_basic
          - 上市日期(listed_date): tushare stock_basic
          - 涨跌停价(upper_limit, lower_limit): tushare stk_limit
          - ST标记(is_st): tushare stock_st
          - 日期×股票网格: tushare daily (symbol, trade_date)

        Returns:
            DataFrame: 长格式，包含 symbol/trade_date 及可用的状态字段
        """
        # ── 1. 从 tushare daily 加载日期×股票网格 ──
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")

        conn_ts = sqlite3.connect(self.tushare_db_path)
        query = f"""
            SELECT ts_code, trade_date
            FROM daily
            WHERE trade_date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql_query(query, conn_ts)
        conn_ts.close()

        df["symbol"] = df["ts_code"].apply(self._ts_to_jq)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[["symbol", "trade_date"]]

        if symbols is not None:
            symbol_set = set(symbols)
            df = df[df["symbol"].isin(symbol_set)]

        # ── 2. 从 tushare 加载停牌数据，合并到 df ──
        suspend_df = self._load_tushare_suspend(start_date, end_date)
        if not suspend_df.empty:
            suspend_keys = set(
                zip(suspend_df["symbol"], suspend_df["trade_date"])
            )
            df["is_suspended"] = [
                (sym, dt) in suspend_keys
                for sym, dt in zip(df["symbol"], df["trade_date"])
            ]
        else:
            df["is_suspended"] = False

        # ── 3. 从 tushare 加载 ST 数据，合并到 df ──
        st_df = self._load_tushare_st(start_date, end_date)
        if not st_df.empty:
            st_keys = set(
                zip(st_df["symbol"], st_df["trade_date"])
            )
            df["is_st"] = [
                (sym, dt) in st_keys
                for sym, dt in zip(df["symbol"], df["trade_date"])
            ]
        else:
            df["is_st"] = False

        # ── 4. 从 tushare 加载退市日期，合并到 df ──
        delist_df = self._load_tushare_delist()
        if not delist_df.empty:
            delist_map = delist_df.set_index("symbol")["delisted_date"]
            df["delisted_date"] = df["symbol"].map(delist_map)
        else:
            df["delisted_date"] = pd.NaT

        # ── 5. 从 tushare 加载上市日期，合并到 df ──
        list_date_df = self._load_tushare_list_date()
        if not list_date_df.empty:
            list_date_map = list_date_df.set_index("symbol")["listed_date"]
            df["listed_date"] = df["symbol"].map(list_date_map)
        else:
            df["listed_date"] = pd.NaT

        # ── 6. 从 tushare 加载涨跌停价，合并到 df ──
        limit_df = self._load_tushare_stk_limit(start_date, end_date)
        if not limit_df.empty:
            limit_indexed = limit_df.set_index(["symbol", "trade_date"])
            df_indexed = df.set_index(["symbol", "trade_date"])
            df_indexed["upper_limit"] = limit_indexed["upper_limit"]
            df_indexed["lower_limit"] = limit_indexed["lower_limit"]
            df = df_indexed.reset_index()
            df["upper_limit"] = df["upper_limit"].fillna(0.0)
            df["lower_limit"] = df["lower_limit"].fillna(0.0)
        else:
            df["upper_limit"] = 0.0
            df["lower_limit"] = 0.0

        # ── 6. 填充缺失列的默认值 ──
        defaults = {
            "is_suspended": False,
            "upper_limit": 0.0,
            "lower_limit": 0.0,
            "is_st": False,
            "listed_date": pd.NaT,
            "delisted_date": pd.NaT,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        for date_col in ["listed_date", "delisted_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        n_suspended = df["is_suspended"].sum()
        n_st = df["is_st"].sum()
        n_delist = df["delisted_date"].notna().any() and delist_df.shape[0] or 0
        n_limit = (df["upper_limit"] > 0).sum()
        print(f"  - 交易状态: 停牌 {n_suspended:,}, ST {n_st:,}, "
              f"涨跌停价 {n_limit:,}, "
              f"退市 {n_delist} 只 (数据源: tushare)")

        return df

    def load_trade_context(
        self,
        start_date: str,
        end_date: str,
        raw_open_arr: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        new_stock_filter_days: int = 365,
        suspended_value_mode: str = "freeze",
    ):
        """
        加载并构建 TradeContext（交易上下文）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            raw_open_arr: 不复权开盘价矩阵 (N, T) ndarray
            symbols: 股票代码数组 (N,)
            dates: 日期数组 (T,)
            new_stock_filter_days: 新股过滤天数（上市后N日历天不参与交易）
            suspended_value_mode: 停牌价值处理模式 'freeze' 或 'zero'

        Returns:
            TradeContext: 交易上下文对象（内部 ndarray）
        """
        from core.trade_context import TradeContext

        trade_status_df = self.load_trade_status(start_date, end_date)

        def pivot_bool(col_name, default=False) -> np.ndarray:
            if col_name not in trade_status_df.columns:
                return np.full((len(symbols), len(dates)), default, dtype=bool)
            pivoted = trade_status_df.pivot(
                index="symbol", columns="trade_date", values=col_name
            )
            arr = pivoted.reindex(index=symbols, columns=dates).to_numpy().copy()
            mask = pd.isna(arr)
            arr[mask] = default
            return arr.astype(bool)

        def pivot_float(col_name, default=0.0) -> np.ndarray:
            if col_name not in trade_status_df.columns:
                return np.full((len(symbols), len(dates)), default, dtype=np.float64)
            pivoted = trade_status_df.pivot(
                index="symbol", columns="trade_date", values=col_name
            )
            return (
                pivoted.reindex(index=symbols, columns=dates)
                .fillna(default)
                .values.astype(np.float64)
            )

        is_suspended = pivot_bool("is_suspended", False)
        upper_limit = pivot_float("upper_limit", 0.0)
        lower_limit = pivot_float("lower_limit", 0.0)
        is_st = pivot_bool("is_st", False)

        listed_dates = None
        if "listed_date" in trade_status_df.columns:
            listed_dates = (
                trade_status_df.dropna(subset=["listed_date"])
                .groupby("symbol")["listed_date"]
                .first()
                .reindex(symbols)
            )

        delisted_dates = None
        if "delisted_date" in trade_status_df.columns:
            delisted_dates = (
                trade_status_df.dropna(subset=["delisted_date"])
                .groupby("symbol")["delisted_date"]
                .first()
                .reindex(symbols)
            )

        return TradeContext(
            symbols=symbols,
            dates=dates,
            is_suspended=is_suspended,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            raw_open=raw_open_arr,
            is_st=is_st,
            listed_dates=listed_dates,
            delisted_dates=delisted_dates,
            new_stock_filter_days=new_stock_filter_days,
            suspended_value_mode=suspended_value_mode,
        )

    def _estimate_memory(self, n_symbols: int, n_dates: int) -> float:
        """估算基础层内存占用（MB）"""
        ohlcv_adj_mb = n_symbols * n_dates * 5 * 8 / 1024 / 1024
        raw_open_mb = n_symbols * n_dates * 8 / 1024 / 1024
        masks_mb = n_symbols * n_dates * 4 / 1024 / 1024
        total_mb = ohlcv_adj_mb + raw_open_mb + masks_mb
        print(f"  - 预估内存: {total_mb:.0f} MB "
              f"({n_symbols} 只 × {n_dates} 天)")
        if total_mb > 2048:
            warnings.warn(
                f"预估内存 {total_mb:.0f} MB 较大，建议缩小日期范围或股票池"
            )
        return total_mb
