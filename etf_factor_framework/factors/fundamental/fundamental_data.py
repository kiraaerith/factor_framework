"""
FundamentalData: lixinger 财务数据容器

从 lixinger SQLite 数据库加载季度财务数据（financial_statements 表），
以 report_date（披露日期）为基准，展开为日频面板（stock × date）。

关键设计：
- 用 report_date 而非 date（报告期），保证无未来数据泄露
- report_date 为 NULL 时用兜底策略估计：报告期 + 固定延迟
- forward-fill：每条季报数据从 report_date 起生效，直到下次披露
- 只在交易日填充（对齐交易日历）
"""

import sqlite3
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # etf_cross_ml-master
LIXINGER_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "lixinger.db")
TUSHARE_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "tushare.db")


def lixinger_code_to_symbol(stock_code: str) -> str:
    """将 lixinger 6位股票代码转换为 DuckDB symbol 格式（SHSE.XXXXXX / SZSE.XXXXXX）"""
    code = str(stock_code).strip()
    if code.startswith('6') or code.startswith('9'):
        return f"SHSE.{code}"
    elif code.startswith(('0', '2', '3')):
        return f"SZSE.{code}"
    elif code.startswith(('4', '8')):
        return f"BSE.{code}"
    return f"SHSE.{code}"


class FundamentalData:
    """
    基本面数据容器

    从 lixinger SQLite 加载季度财务数据，以 report_date 展开为日频面板。

    使用示例::

        fd = FundamentalData('2013-01-01', '2025-12-31')
        values, symbols, dates = fd.get_daily_panel('q_m_roe_t')  # (ndarray N×T, symbols, dates)

    Args:
        start_date: 回测起始日期（交易日面板的起始）
        end_date: 回测结束日期
        stock_codes: 要加载的 lixinger 股票代码列表（None=全市场）
        lixinger_db: lixinger SQLite 数据库路径
        tushare_db: tushare SQLite 数据库路径（交易日历）
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]] = None,
        lixinger_db: str = LIXINGER_DB,
        tushare_db: str = TUSHARE_DB,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self._stock_codes = stock_codes
        self._lixinger_db = lixinger_db
        self._tushare_db = tushare_db

        self._raw_data: Optional[pd.DataFrame] = None
        self._trading_dates: Optional[pd.DatetimeIndex] = None
        self._panel_cache: dict = {}

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load_raw_data(self) -> None:
        """从 lixinger 加载 financial_statements，并做预处理（一次性）"""
        if self._raw_data is not None:
            return

        # 向前多加载 2 年，确保回测起始日前有足够的 forward-fill 数据
        load_start = (self.start_date - pd.DateOffset(years=2)).date()
        load_end = self.end_date.date()

        conn = sqlite3.connect(self._lixinger_db)

        if self._stock_codes:
            codes_str = "','".join(str(c) for c in self._stock_codes)
            codes_filter = f"AND stock_code IN ('{codes_str}')"
        else:
            codes_filter = ""

        query = f"""
            SELECT *
            FROM financial_statements
            WHERE date BETWEEN '{load_start}' AND '{load_end}'
            {codes_filter}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            warnings.warn("FundamentalData: financial_statements 查询结果为空")
            self._raw_data = df
            return

        # 转换日期类型（去掉时区信息，统一为 tz-naive）
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.tz_localize(None)

        # 兜底：对 report_date 为 NULL 的记录用报告期+固定延迟估计
        null_mask = df["report_date"].isna()
        if null_mask.any():
            df.loc[null_mask, "report_date"] = df.loc[null_mask].apply(
                self._impute_report_date, axis=1
            )
            n_imputed = null_mask.sum()
            if n_imputed > 0:
                print(f"  - report_date 兜底估计：{n_imputed} 条记录")

        # 去重：相同 (stock_code, report_date) 只保留 date 最大的记录（最新季报）
        df = df.sort_values(["stock_code", "report_date", "date"])
        df = df.groupby(["stock_code", "report_date"], as_index=False).last()

        # 添加 DuckDB symbol 列
        df["symbol"] = df["stock_code"].apply(lixinger_code_to_symbol)

        self._raw_data = df
        print(
            f"  - 加载理杏仁财务数据: {df['stock_code'].nunique()} 只股票 × {len(df)} 条记录"
        )

    def _get_trading_dates(self) -> pd.DatetimeIndex:
        """从 tushare trade_cal 加载交易日历（懒加载）"""
        if self._trading_dates is not None:
            return self._trading_dates

        start_str = self.start_date.strftime("%Y%m%d")
        end_str = self.end_date.strftime("%Y%m%d")

        conn = sqlite3.connect(self._tushare_db)
        query = f"""
            SELECT cal_date
            FROM trade_cal
            WHERE exchange = 'SSE'
              AND is_open = 1
              AND cal_date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY cal_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        self._trading_dates = pd.DatetimeIndex(pd.to_datetime(df["cal_date"]))
        print(f"  - 交易日历: {len(self._trading_dates)} 个交易日 "
              f"({self.start_date.date()} ~ {self.end_date.date()}) (数据源: tushare)")
        return self._trading_dates

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def get_daily_panel(self, field: str):
        """
        返回指定字段的日频面板 (N股 × T日)

        以 report_date 为信号生效日，forward-fill 到下一次披露日。

        Args:
            field: lixinger financial_statements 中的字段名，如 'q_m_roe_t'

        Returns:
            tuple: (values: np.ndarray (N, T) float64, symbols: np.ndarray (N,), dates: np.ndarray (T,) datetime64[ns])
        """
        if field in self._panel_cache:
            return self._panel_cache[field]

        self._load_raw_data()
        trading_dates = self._get_trading_dates()

        if self._raw_data.empty:
            empty = (np.empty((0, len(trading_dates)), dtype=np.float64),
                     np.array([], dtype=str),
                     np.array(trading_dates, dtype='datetime64[ns]'))
            self._panel_cache[field] = empty
            return empty

        # 1. 提取有效数据（field 非 NaN）
        df = self._raw_data[["symbol", "report_date", field]].dropna(
            subset=[field]
        ).copy()

        # 2. 仅保留 report_date <= end_date 的记录（防止未来数据渗入）
        df = df[df["report_date"] <= self.end_date]

        if df.empty:
            empty = (np.empty((0, len(trading_dates)), dtype=np.float64),
                     np.array([], dtype=str),
                     np.array(trading_dates, dtype='datetime64[ns]'))
            self._panel_cache[field] = empty
            return empty

        # 3. pivot: 行=symbol, 列=report_date, 值=field
        pivot = df.pivot_table(
            index="symbol",
            columns="report_date",
            values=field,
            aggfunc="last",
        )

        # 4. 将 report_date 合并到 trading_dates，reindex，forward-fill
        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1)

        # 5. 只保留交易日列，立即转 ndarray
        panel = panel.reindex(columns=trading_dates)
        values = panel.values.astype(np.float64)
        symbols_arr = np.array(panel.index.tolist())
        dates_arr = np.array(panel.columns.tolist(), dtype='datetime64[ns]')

        result = (values, symbols_arr, dates_arr)
        self._panel_cache[field] = result
        return result

    def truncate(self, end_date: str) -> "FundamentalData":
        """
        返回截断到 end_date 的 FundamentalData（用于泄露检测）

        截断逻辑：仅保留 report_date <= end_date 的记录，
        并将交易日历收缩到 [start_date, end_date]。
        """
        end_ts = pd.Timestamp(end_date)
        new_fd = FundamentalData(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=end_date,
            stock_codes=self._stock_codes,
            lixinger_db=self._lixinger_db,
            tushare_db=self._tushare_db,
        )

        # 复用已加载的原始数据，避免重复 IO
        if self._raw_data is not None:
            new_fd._raw_data = self._raw_data[
                self._raw_data["report_date"] <= end_ts
            ].copy()

        # 复用交易日历并截断
        if self._trading_dates is not None:
            new_fd._trading_dates = self._trading_dates[
                self._trading_dates <= end_ts
            ]

        return new_fd

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        self._load_raw_data()
        return sorted(self._raw_data["symbol"].unique())

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        return self._get_trading_dates()

    @property
    def n_trading_days(self) -> int:
        return len(self._get_trading_dates())

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def get_industry_map(self) -> dict:
        """
        返回 symbol -> industry 的静态映射（基于 lixinger company_list.industry）。

        symbol 格式与 DuckDB 对齐（如 SHSE.600519）。
        industry 字段为理杏仁行业分类字符串，部分股票可能为 None。

        Returns:
            dict: {symbol: industry_str}
        """
        conn = sqlite3.connect(self._lixinger_db)
        df = pd.read_sql_query(
            "SELECT stock_code, industry FROM company_list WHERE industry IS NOT NULL",
            conn,
        )
        conn.close()

        result = {}
        for _, row in df.iterrows():
            symbol = lixinger_code_to_symbol(str(row["stock_code"]).strip())
            result[symbol] = row["industry"]
        return result

    def get_market_cap_panel(self):
        """
        返回日频市值面板 (N股 × T日)，字段为 lixinger fundamental.mc（总市值，亿元）。

        - 以日历日为基准，forward-fill 到下个数据点
        - 最终只保留交易日列（对齐掘金日历）

        Returns:
            tuple: (values: np.ndarray (N, T) float64, symbols: np.ndarray (N,), dates: np.ndarray (T,) datetime64[ns])
        """
        if "_market_cap_panel" in self._panel_cache:
            return self._panel_cache["_market_cap_panel"]

        load_start = (self.start_date - pd.DateOffset(days=5)).date()
        load_end = self.end_date.date()

        conn = sqlite3.connect(self._lixinger_db)
        df = pd.read_sql_query(
            f"""
            SELECT stock_code, date, mc
            FROM fundamental
            WHERE substr(date, 1, 10) BETWEEN '{load_start}' AND '{load_end}'
              AND mc IS NOT NULL
            """,
            conn,
        )
        conn.close()

        trading_dates = self._get_trading_dates()

        if df.empty:
            empty = (np.empty((0, len(trading_dates)), dtype=np.float64),
                     np.array([], dtype=str),
                     np.array(trading_dates, dtype='datetime64[ns]'))
            self._panel_cache["_market_cap_panel"] = empty
            return empty

        _dates = pd.to_datetime(df["date"])
        df["date"] = _dates.dt.tz_localize(None) if _dates.dt.tz is not None else _dates
        df["symbol"] = df["stock_code"].apply(lixinger_code_to_symbol)

        if trading_dates.tz is not None:
            trading_dates = trading_dates.tz_localize(None)

        # pivot: 行=symbol, 列=date
        pivot = df.pivot_table(index="symbol", columns="date", values="mc", aggfunc="last")

        # 合并交易日，forward-fill，只保留交易日列，立即转 ndarray
        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1).reindex(columns=trading_dates)
        values = panel.values.astype(np.float64)
        symbols_arr = np.array(panel.index.tolist())
        dates_arr = np.array(panel.columns.tolist(), dtype='datetime64[ns]')

        print(f"  - 市值面板: {values.shape[0]} 只股票 × {values.shape[1]} 个交易日")
        result = (values, symbols_arr, dates_arr)
        self._panel_cache["_market_cap_panel"] = result
        return result

    def get_valuation_panel(self, field: str):
        """
        返回日频估值面板 (N股 × T日)，从 lixinger fundamental 表加载任意估值字段。

        适用于 pb / pe_ttm / ps_ttm / dyr 等日频更新的估值指标。
        无季报延迟问题（T 日数据 T 日已知），直接 forward-fill 到下个数据点。

        Args:
            field: lixinger fundamental 表中的字段名，如 'pb'、'pe_ttm'

        Returns:
            tuple: (values: np.ndarray (N, T) float64, symbols: np.ndarray (N,), dates: np.ndarray (T,) datetime64[ns])
        """
        cache_key = f"_valuation_{field}"
        if cache_key in self._panel_cache:
            return self._panel_cache[cache_key]

        load_start = (self.start_date - pd.DateOffset(days=5)).date()
        load_end = self.end_date.date()

        # substr(date, 1, 10) extracts the YYYY-MM-DD portion from the ISO8601+tz string
        # stored in lixinger (e.g. '2023-05-29T00:00:00+08:00'). Plain string comparison
        # against '2023-05-29' would fail to include the boundary date because
        # '2023-05-29T...' > '2023-05-29' lexicographically.
        conn = sqlite3.connect(self._lixinger_db)
        df = pd.read_sql_query(
            f"""
            SELECT stock_code, date, {field}
            FROM fundamental
            WHERE substr(date, 1, 10) BETWEEN '{load_start}' AND '{load_end}'
              AND {field} IS NOT NULL
            """,
            conn,
        )
        conn.close()

        trading_dates = self._get_trading_dates()

        if df.empty:
            empty = (np.empty((0, len(trading_dates)), dtype=np.float64),
                     np.array([], dtype=str),
                     np.array(trading_dates, dtype='datetime64[ns]'))
            self._panel_cache[cache_key] = empty
            return empty

        _dates = pd.to_datetime(df["date"])
        df["date"] = _dates.dt.tz_localize(None) if _dates.dt.tz is not None else _dates
        df["symbol"] = df["stock_code"].apply(lixinger_code_to_symbol)

        if trading_dates.tz is not None:
            trading_dates = trading_dates.tz_localize(None)

        pivot = df.pivot_table(index="symbol", columns="date", values=field, aggfunc="last")

        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1).reindex(columns=trading_dates)
        values = panel.values.astype(np.float64)
        symbols_arr = np.array(panel.index.tolist())
        dates_arr = np.array(panel.columns.tolist(), dtype='datetime64[ns]')

        print(f"  - 估值面板({field}): {values.shape[0]} 只股票 × {values.shape[1]} 个交易日")
        result = (values, symbols_arr, dates_arr)
        self._panel_cache[cache_key] = result
        return result

    @staticmethod
    def _impute_report_date(row) -> pd.Timestamp:
        """对 report_date 为空的记录，用报告期+固定延迟估计披露日"""
        report_date = row["date"]
        month = report_date.month
        if month == 3:    # Q1，最晚 4月30日
            return report_date + pd.DateOffset(months=1)
        elif month == 6:  # 半年报，最晚 8月31日
            return report_date + pd.DateOffset(months=2)
        elif month == 9:  # Q3，最晚 10月31日
            return report_date + pd.DateOffset(months=1)
        else:             # 年报，最晚次年 4月30日
            return report_date + pd.DateOffset(months=4)
