"""
标签构建模块

从 DuckDB trade_status 表构建 ST/退市标签，用于测评过滤器的准确性。
标签本身不参与过滤器的运行逻辑。

标签定义：
  - target='st': 当前非 ST，且在 [date+1, date+horizon] 内变为 ST → label=True
  - target='delist': 在 [date+1, date+horizon] 内退市 → label=True
  - target='st_or_delist': 上述任一成立 → label=True
"""

from typing import Optional
import numpy as np
import pandas as pd


class LabelBuilder:
    """
    从 trade_status 数据构建 ST/退市事件标签。

    输出 (N, T) bool 矩阵，True 表示该 (stock, date) 在未来 horizon 个交易日内
    发生了目标事件。
    """

    VALID_TARGETS = ("st", "delist", "st_or_delist")

    @staticmethod
    def build_labels(
        is_st: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        target: str = "st_or_delist",
        horizon: int = 252,
        delisted_dates: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        构建事件标签矩阵。

        Args:
            is_st: (N, T) bool, 每个 (stock, date) 是否为 ST 状态。
                   来自 TradeContext.is_st 或 trade_status 表的 is_st 字段。
            symbols: (N,) str, 股票代码数组
            dates: (T,) datetime64, 日期数组
            target: 标签类型 'st' / 'delist' / 'st_or_delist'
            horizon: 预测窗口（交易日数）
            delisted_dates: pd.Series, index=symbol, value=退市日期 (datetime64)。
                           如果 target 包含 'delist'，必须提供。

        Returns:
            (N, T) bool ndarray, True=未来 horizon 内发生了目标事件
        """
        if target not in LabelBuilder.VALID_TARGETS:
            raise ValueError(
                f"target must be one of {LabelBuilder.VALID_TARGETS}, got '{target}'"
            )

        N, T = is_st.shape
        labels = np.zeros((N, T), dtype=bool)

        if target in ("st", "st_or_delist"):
            st_labels = LabelBuilder._build_st_labels(is_st, horizon)
            labels |= st_labels

        if target in ("delist", "st_or_delist"):
            if delisted_dates is None:
                raise ValueError(
                    "delisted_dates is required when target includes 'delist'"
                )
            delist_labels = LabelBuilder._build_delist_labels(
                symbols, dates, delisted_dates, horizon
            )
            labels |= delist_labels

        return labels

    @staticmethod
    def _build_st_labels(is_st: np.ndarray, horizon: int) -> np.ndarray:
        """
        构建 ST 事件标签。

        对于每个 (stock, date)，如果：
          1. 当前不是 ST (is_st[i, t] == False)
          2. 在 [t+1, t+horizon] 范围内存在 is_st[i, t'] == True
        则 label[i, t] = True。

        已经是 ST 的股票 label = False（不需要预测 "继续 ST"）。
        """
        N, T = is_st.shape
        labels = np.zeros((N, T), dtype=bool)

        # 对每只股票，找到所有 "从非 ST 变为 ST" 的转变点
        # 转变点定义：is_st[i, t-1] == False and is_st[i, t] == True
        # 即 is_st[:, 1:] & ~is_st[:, :-1]
        if T < 2:
            return labels

        # (N, T-1) bool: t 位置为 True 表示在 t 时刻发生了 ST 转变
        st_onset = is_st[:, 1:] & ~is_st[:, :-1]  # onset at index 1..T-1

        for i in range(N):
            # 找到股票 i 的所有 ST 转变日的列索引 (在 dates 中的位置)
            onset_cols = np.where(st_onset[i])[0] + 1  # +1 因为 st_onset 从 index 1 开始
            for col in onset_cols:
                # 在 [col - horizon, col - 1] 范围内，且当时不是 ST 的日期，标记为 True
                start = max(0, col - horizon)
                end = col  # 不含 col 本身（col 当天已经是 ST 了）
                # 只标记当时不是 ST 的日期
                non_st_mask = ~is_st[i, start:end]
                labels[i, start:end] |= non_st_mask

        return labels

    @staticmethod
    def _build_delist_labels(
        symbols: np.ndarray,
        dates: np.ndarray,
        delisted_dates: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        """
        构建退市事件标签。

        对于每个 (stock, date)，如果该股票在 [date+1, date+horizon] 内退市，
        则 label[i, t] = True。
        """
        N, T = len(symbols), len(dates)
        labels = np.zeros((N, T), dtype=bool)

        dates_ts = pd.DatetimeIndex(dates)

        for i, sym in enumerate(symbols):
            if sym not in delisted_dates.index:
                continue
            delist_dt = delisted_dates[sym]
            if pd.isna(delist_dt):
                continue

            delist_dt = pd.Timestamp(delist_dt)

            # 在 [delist_dt - horizon 个交易日, delist_dt) 范围内的日期标记为 True
            # 找到 delist_dt 在 dates 中的位置
            delist_pos = np.searchsorted(dates_ts, delist_dt, side="right")
            # delist_pos 是第一个 > delist_dt 的位置，所以 delist_pos - 1 是最后一个 <= delist_dt 的

            start = max(0, delist_pos - horizon)
            end = delist_pos  # 不含 delist_pos（退市日之后无数据）
            if start < end:
                labels[i, start:end] = True

        return labels

    @staticmethod
    def build_labels_from_trade_status_df(
        trade_status_df: pd.DataFrame,
        symbols: np.ndarray,
        dates: np.ndarray,
        target: str = "st_or_delist",
        horizon: int = 252,
    ) -> np.ndarray:
        """
        便捷方法：直接从 trade_status DataFrame 构建标签。

        Args:
            trade_status_df: StockDataLoader.load_trade_status() 返回的 DataFrame
            symbols: (N,) 股票代码
            dates: (T,) 日期
            target: 标签类型
            horizon: 预测窗口

        Returns:
            (N, T) bool ndarray
        """
        # Pivot is_st
        if "is_st" in trade_status_df.columns:
            pivoted_st = trade_status_df.pivot(
                index="symbol", columns="trade_date", values="is_st"
            )
            is_st = (
                pivoted_st.reindex(index=symbols, columns=dates)
                .fillna(False)
                .values.astype(bool)
            )
        else:
            is_st = np.zeros((len(symbols), len(dates)), dtype=bool)

        # Extract delisted_dates
        delisted_dates = None
        if target in ("delist", "st_or_delist"):
            if "delisted_date" in trade_status_df.columns:
                delist_df = trade_status_df[["symbol", "delisted_date"]].drop_duplicates("symbol")
                delist_df = delist_df.set_index("symbol")["delisted_date"]
                # 过滤掉无效值（1970-01-01 或 NaT 通常表示未退市）
                delist_df = delist_df[delist_df > pd.Timestamp("2000-01-01")]
                delisted_dates = delist_df
            else:
                delisted_dates = pd.Series(dtype="datetime64[ns]")

        return LabelBuilder.build_labels(
            is_st=is_st,
            symbols=symbols,
            dates=dates,
            target=target,
            horizon=horizon,
            delisted_dates=delisted_dates,
        )
