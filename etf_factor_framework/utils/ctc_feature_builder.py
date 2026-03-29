"""
CTC因子特征构建工具

将MultiIndex DataFrame数据转为OHLCVData，计算CTC因子，
返回flat panel格式的特征DataFrame，供rf_backtest使用。
"""

import numpy as np
import pandas as pd
from itertools import product as iter_product

from etf_factor_framework.core.ohlcv_data import OHLCVData
from etf_factor_framework.factors.ctc import (
    HighVolReturnSum, LowVolReturnSum,
    HighVolReturnStd, LowVolReturnStd,
    HighVolAmplitude, LowVolAmplitude,
    HighVolChangeReturnSum, LowVolChangeReturnSum,
    HighVolChangeReturnStd, LowVolChangeReturnStd,
    HighVolChangeAmplitude, LowVolChangeAmplitude,
    HighPriceRelativeVolume, LowPriceRelativeVolume,
    HighPriceVolumeChange, LowPriceVolumeChange,
    VolAmplitudeImbalance, VolReturnStdImbalance,
)
from etf_factor_framework.factors.ctc.price_volume_correlation import PVCorr


# 因子名 -> 类的映射
FACTOR_CLASS_MAP = {
    'HighVolReturnSum': HighVolReturnSum,
    'LowVolReturnSum': LowVolReturnSum,
    'HighVolReturnStd': HighVolReturnStd,
    'LowVolReturnStd': LowVolReturnStd,
    'HighVolAmplitude': HighVolAmplitude,
    'LowVolAmplitude': LowVolAmplitude,
    'HighVolChangeReturnSum': HighVolChangeReturnSum,
    'LowVolChangeReturnSum': LowVolChangeReturnSum,
    'HighVolChangeReturnStd': HighVolChangeReturnStd,
    'LowVolChangeReturnStd': LowVolChangeReturnStd,
    'HighVolChangeAmplitude': HighVolChangeAmplitude,
    'LowVolChangeAmplitude': LowVolChangeAmplitude,
    'HighPriceRelativeVolume': HighPriceRelativeVolume,
    'LowPriceRelativeVolume': LowPriceRelativeVolume,
    'HighPriceVolumeChange': HighPriceVolumeChange,
    'LowPriceVolumeChange': LowPriceVolumeChange,
    'VolAmplitudeImbalance': VolAmplitudeImbalance,
    'VolReturnStdImbalance': VolReturnStdImbalance,
    'PVCorr': PVCorr,
}

# 因子参数网格
# 有 window + top_pct 参数的因子
FACTORS_WITH_TOP_PCT = [
    'HighVolReturnSum', 'LowVolReturnSum',
    'HighVolReturnStd', 'LowVolReturnStd',
    'HighVolAmplitude', 'LowVolAmplitude',
    'HighVolChangeReturnSum', 'LowVolChangeReturnSum',
    'HighVolChangeReturnStd', 'LowVolChangeReturnStd',
    'HighVolChangeAmplitude', 'LowVolChangeAmplitude',
    'HighPriceRelativeVolume', 'LowPriceRelativeVolume',
    'HighPriceVolumeChange', 'LowPriceVolumeChange',
    'VolAmplitudeImbalance', 'VolReturnStdImbalance',
]

# 只有 window 参数的因子
FACTORS_WINDOW_ONLY = ['PVCorr']

WINDOWS = [20, 40, 60]
TOP_PCTS = [0.1, 0.2, 0.3]


def get_param_grid(factor_name):
    """获取因子的超参组合列表"""
    if factor_name in FACTORS_WITH_TOP_PCT:
        return [{'window': w, 'top_pct': p} for w, p in iter_product(WINDOWS, TOP_PCTS)]
    elif factor_name in FACTORS_WINDOW_ONLY:
        return [{'window': w} for w in WINDOWS]
    else:
        raise ValueError(f"Unknown factor: {factor_name}")


def data_to_ohlcv(data):
    """
    将MultiIndex DataFrame (symbol, eob) 转为 OHLCVData (N×T pivot)。

    Parameters
    ----------
    data : pd.DataFrame
        MultiIndex (symbol, eob) with columns: open, high, low, close, volume

    Returns
    -------
    OHLCVData
    """
    pivots = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        pivots[col] = data[col].unstack(level='symbol').T  # N×T (index=symbol, columns=eob)
    return OHLCVData(
        open=pivots['open'],
        high=pivots['high'],
        low=pivots['low'],
        close=pivots['close'],
        volume=pivots['volume'],
    )


def build_ctc_features(data, factor_name):
    """
    构建指定CTC因子的所有超参变体特征。

    Parameters
    ----------
    data : pd.DataFrame
        MultiIndex (symbol, eob) with OHLCV columns
    factor_name : str
        CTC因子类名

    Returns
    -------
    pd.DataFrame
        flat panel: (eob, symbol, feat1, feat2, ...)
    list
        feature column names
    """
    factor_class = FACTOR_CLASS_MAP[factor_name]
    param_grid = get_param_grid(factor_name)
    ohlcv_data = data_to_ohlcv(data)

    feature_cols = []
    all_feat_dfs = []

    for params in param_grid:
        # 构建列名
        if 'top_pct' in params:
            col_name = f"{factor_name}_w{params['window']}_p{params['top_pct']}"
        else:
            col_name = f"{factor_name}_w{params['window']}"

        # 实例化并计算
        factor = factor_class(**params)
        factor_data = factor.calculate(ohlcv_data)

        # factor_data.values: N×T DataFrame (index=symbol, columns=date)
        # 转为 long format: (eob, symbol, value)
        stacked = factor_data.values.stack()
        stacked.index.names = ['symbol', 'eob']
        feat_series = stacked.rename(col_name)
        all_feat_dfs.append(feat_series)
        feature_cols.append(col_name)

    # 合并所有特征
    feat_panel = pd.concat(all_feat_dfs, axis=1).reset_index()

    return feat_panel, feature_cols
