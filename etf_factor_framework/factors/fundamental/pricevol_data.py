"""
PriceVolData: 量价数据容器

从 Tushare SQLite 数据库加载后复权日频 OHLCV 数据，缓存为 (N, T) 面板。
支持 truncate(end_date) 用于泄露检测——截断后只保留 end_date 之前的数据。

设计与 FundamentalData 对称：
- FundamentalData 管理基本面数据（理杏仁），truncate 截断 report_date
- PriceVolData 管理量价数据（Tushare），truncate 截断交易日

使用示例::

    pvd = PriceVolData('2016-01-01', '2025-12-31')
    ohlcv = pvd.get_ohlcv(lookback_extra_days=300)  # 回望期向前 300 天
    close = ohlcv.close   # (N, T_with_lookback)
    symbols = ohlcv.symbols
    dates = ohlcv.dates

    # 泄露检测：截断到 2022-06-30
    short_pvd = pvd.truncate('2022-06-30')
    short_ohlcv = short_pvd.get_ohlcv(lookback_extra_days=300)
    # short_ohlcv.dates[-1] <= 2022-06-30
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.ohlcv_data import OHLCVData
from data.stock_data_loader import StockDataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # etf_cross_ml-master
TUSHARE_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "tushare.db")

logger = logging.getLogger(__name__)


class PriceVolData:
    """
    量价数据容器

    从 Tushare SQLite 加载后复权 OHLCV，懒加载并缓存。
    支持 truncate() 截断时间范围，用于泄露检测。

    Args:
        start_date: 回测起始日期（因子输出的起始日期）
        end_date: 回测结束日期
        tushare_db: Tushare SQLite 数据库路径
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tushare_db: str = TUSHARE_DB,
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self._tushare_db = tushare_db

        # Cache: keyed by lookback_extra_days
        self._ohlcv_cache: Optional[OHLCVData] = None
        self._cached_lookback: int = -1  # lookback used for current cache

    def get_ohlcv(
        self,
        lookback_extra_days: int = 0,
        use_adjusted: bool = True,
    ) -> OHLCVData:
        """
        获取 OHLCV 数据，包含 lookback 回望期。

        首次调用触发数据库加载，后续调用复用缓存。
        若请求的 lookback_extra_days 大于缓存中的，重新加载。

        Args:
            lookback_extra_days: 为因子回望窗口额外预留的日历天数
            use_adjusted: 是否使用后复权数据，默认 True

        Returns:
            OHLCVData: OHLCV 数据容器 (N, T)
        """
        if self._ohlcv_cache is not None and lookback_extra_days <= self._cached_lookback:
            return self._ohlcv_cache

        logger.info(
            f"PriceVolData: loading OHLCV "
            f"({self.start_date.date()} ~ {self.end_date.date()}, "
            f"lookback={lookback_extra_days} days)"
        )

        loader = StockDataLoader(tushare_db_path=self._tushare_db)
        ohlcv = loader.load_ohlcv(
            start_date=str(self.start_date.date()),
            end_date=str(self.end_date.date()),
            use_adjusted=use_adjusted,
            lookback_extra_days=lookback_extra_days,
        )

        self._ohlcv_cache = ohlcv
        self._cached_lookback = lookback_extra_days
        return ohlcv

    def truncate(self, end_date: str) -> "PriceVolData":
        """
        返回截断到 end_date 的 PriceVolData（用于泄露检测）。

        截断逻辑：只保留 dates <= end_date 的数据。
        回望期（start_date 之前的数据）不受影响。

        如果当前实例已有缓存的 OHLCV，直接从缓存中切片，避免重复 IO。

        Args:
            end_date: 截断日期（含）

        Returns:
            PriceVolData: 新实例，end_date 被截断
        """
        end_ts = pd.Timestamp(end_date)

        new_pvd = PriceVolData(
            start_date=str(self.start_date.date()),
            end_date=end_date,
            tushare_db=self._tushare_db,
        )

        # Reuse cached OHLCV if available — slice date dimension
        if self._ohlcv_cache is not None:
            end_dt64 = np.datetime64(end_ts)
            dates = self._ohlcv_cache._dates
            mask = dates <= end_dt64

            if mask.any():
                idx = np.where(mask)[0]
                new_pvd._ohlcv_cache = OHLCVData(
                    open=self._ohlcv_cache._open[:, idx].copy(),
                    high=self._ohlcv_cache._high[:, idx].copy(),
                    low=self._ohlcv_cache._low[:, idx].copy(),
                    close=self._ohlcv_cache._close[:, idx].copy(),
                    volume=self._ohlcv_cache._volume[:, idx].copy(),
                    symbols=self._ohlcv_cache._symbols.copy(),
                    dates=dates[idx].copy(),
                )
                new_pvd._cached_lookback = self._cached_lookback

                logger.info(
                    f"PriceVolData.truncate: "
                    f"{len(dates)} -> {int(mask.sum())} dates "
                    f"(end_date={end_date})"
                )

        return new_pvd

    @property
    def n_trading_days(self) -> int:
        """缓存中的交易日数（不含回望期），未加载时返回 0"""
        if self._ohlcv_cache is None:
            return 0
        target_start = np.datetime64(self.start_date)
        return int((self._ohlcv_cache._dates >= target_start).sum())

    def __repr__(self) -> str:
        loaded = "loaded" if self._ohlcv_cache is not None else "not loaded"
        return (
            f"PriceVolData("
            f"{self.start_date.date()} ~ {self.end_date.date()}, "
            f"{loaded})"
        )
