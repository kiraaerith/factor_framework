"""
Shared base class for rolling-model composite factors.

Handles sub-factor computation, alignment, ranking, forward return labels,
and close price loading. Subclasses only need to specify the model class,
model params, and trainer params.
"""

import os
import sys
import logging
import importlib
from typing import Type, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator
from rolling_model import PreparedData, RollingTrainer
from rolling_model.base import ModelWrapper

logger = logging.getLogger(__name__)

SUB_FACTORS = [
    ("factors.fundamental.value.dyr", "DYR", 1),
    ("factors.fundamental.valuation.valuation_SIZE", "valuation_SIZE", -1),
    ("factors.fundamental.growth.ni_st_ft", "NI_ST_FT", 1),
    ("factors.fundamental.value.rp_ep", "RP_EP", 1),
    ("factors.fundamental.growth.na_growth_comp", "NA_GROWTH_COMP", 1),
]

FORWARD_DAYS = 20


def _cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per column. NaN stays NaN. Result in [0,1].
    arr: (N, T) -> returns (N, T)
    """
    result = np.full_like(arr, np.nan)
    N, T = arr.shape
    # Loop: ~T iterations (~2430 days)
    for t in range(T):
        col = arr[:, t]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid < 2:
            continue
        order = np.argsort(col[valid])
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        ranks /= (n_valid - 1)
        result[valid, t] = ranks
    return result


def _compute_forward_returns(close: np.ndarray, days: int) -> np.ndarray:
    """Compute forward N-day returns from close prices.
    close: (N, T) -> returns (N, T), last `days` columns are NaN.
    """
    N, T = close.shape
    fwd = np.full((N, T), np.nan)
    if T <= days:
        return fwd
    with np.errstate(divide='ignore', invalid='ignore'):
        fwd[:, :-days] = close[:, days:] / close[:, :-days] - 1.0
    return fwd


class RollingCompositeBase(FundamentalFactorCalculator):
    """Base class for rolling-model composite factors.

    Subclasses must set class attributes:
        FACTOR_NAME: str
        MODEL_CLASS: Type[ModelWrapper]
        MODEL_PARAMS: dict
        TRAINER_PARAMS: dict  (train_min_days, train_max_days, val_days, test_days, step_days)
    """

    FACTOR_NAME: str = "ROLLING_BASE"
    FACTOR_DIRECTION: int = 1
    MODEL_CLASS: Type[ModelWrapper] = None
    MODEL_PARAMS: dict = {}
    TRAINER_PARAMS: dict = {}
    RANK_LABELS: bool = False  # if True, transform labels to cross-sectional rank
    CUSTOM_SUB_FACTORS: List[Tuple] = None  # override SUB_FACTORS if set

    @property
    def name(self) -> str:
        return self.FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return self.FACTOR_NAME

    @property
    def _sub_factors(self):
        return self.CUSTOM_SUB_FACTORS if self.CUSTOM_SUB_FACTORS is not None else SUB_FACTORS

    @property
    def params(self) -> dict:
        return {
            "sub_factors": [sf[1] for sf in self._sub_factors],
            "model": self.MODEL_CLASS.__name__ if self.MODEL_CLASS else "unknown",
            "forward_days": FORWARD_DAYS,
            "direction": self.FACTOR_DIRECTION,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        # --- Step 1: compute sub-factors and align ---
        sub_results = []
        for mod_path, cls_name, direction in self._sub_factors:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            calculator = cls()
            fd = calculator.calculate(fundamental_data)
            sub_results.append((fd, direction, cls_name))

        common_symbols_set = None
        common_dates_set = None
        for fd, _, _ in sub_results:
            syms = set(fd.symbols.tolist())
            dts = set(fd.dates.astype('int64').tolist())
            common_symbols_set = syms if common_symbols_set is None else common_symbols_set & syms
            common_dates_set = dts if common_dates_set is None else common_dates_set & dts

        common_symbols = np.array(sorted(common_symbols_set))
        common_dates = np.array(sorted(common_dates_set), dtype='datetime64[ns]')
        N = len(common_symbols)
        T = len(common_dates)
        F = len(self._sub_factors)

        logger.info(f"{self.FACTOR_NAME}: common N={N}, T={T}, F={F}")

        if N == 0 or T == 0:
            raise ValueError(f"{self.FACTOR_NAME}: no common data. N={N}, T={T}")

        # Build features (N, T, F) then transpose to (T, N, F)
        features_nt = np.full((N, T, F), np.nan)

        for f_idx, (fd, direction, _) in enumerate(sub_results):
            sym_to_idx = {s: i for i, s in enumerate(fd.symbols.tolist())}
            date_to_idx = {int(d): i for i, d in enumerate(fd.dates.astype('int64'))}

            sym_indices = np.array([sym_to_idx[s] for s in common_symbols.tolist()])
            date_indices = np.array([date_to_idx[int(d)] for d in common_dates.astype('int64')])
            aligned = fd.values[np.ix_(sym_indices, date_indices)]

            ranked = _cross_sectional_rank(aligned)
            if direction == -1:
                ranked = np.where(np.isnan(ranked), np.nan, 1.0 - ranked)

            features_nt[:, :, f_idx] = ranked

        features = np.transpose(features_nt, (1, 0, 2))  # (T, N, F)

        # --- Step 2: forward return labels ---
        from data.stock_data_loader import StockDataLoader

        loader = StockDataLoader()
        ohlcv = loader.load_ohlcv(
            start_date=str(common_dates[0])[:10],
            end_date=str(common_dates[-1])[:10],
            use_adjusted=True,
        )

        ohlcv_sym_to_idx = {s: i for i, s in enumerate(ohlcv.symbols.tolist())}
        ohlcv_date_to_idx = {int(d): i for i, d in enumerate(ohlcv.dates.astype('int64'))}

        valid_syms = [s for s in common_symbols if s in ohlcv_sym_to_idx]
        valid_sym_indices_ohlcv = np.array([ohlcv_sym_to_idx[s] for s in valid_syms])
        valid_sym_indices_common = np.array([
            np.searchsorted(common_symbols, s) for s in valid_syms
        ])

        common_dates_int = common_dates.astype('int64')
        valid_date_mask = np.array([int(d) in ohlcv_date_to_idx for d in common_dates_int])
        valid_date_indices_ohlcv = np.array([
            ohlcv_date_to_idx[int(d)] for d in common_dates_int[valid_date_mask]
        ])

        close_aligned = np.full((N, T), np.nan)
        close_sub = ohlcv.close[np.ix_(valid_sym_indices_ohlcv, valid_date_indices_ohlcv)]
        close_aligned[np.ix_(valid_sym_indices_common, np.where(valid_date_mask)[0])] = close_sub

        fwd_returns = _compute_forward_returns(close_aligned, FORWARD_DAYS)

        if self.RANK_LABELS:
            # Transform to cross-sectional rank (N, T) -> (N, T)
            fwd_returns = _cross_sectional_rank(fwd_returns)

        labels = fwd_returns.T  # (T, N)

        # --- Step 3: rolling model ---
        feature_names = [name for _, name, _ in self._sub_factors]
        prepared = PreparedData(
            features=features,
            labels=labels,
            dates=common_dates,
            symbols=common_symbols,
            feature_names=feature_names,
        )

        trainer = RollingTrainer(**self.TRAINER_PARAMS)
        factor_values_tn = trainer.run(prepared, self.MODEL_CLASS, self.MODEL_PARAMS)

        factor_values = factor_values_tn.T  # (N, T)

        return FactorData(
            values=factor_values,
            symbols=common_symbols,
            dates=common_dates,
            name=self.name,
            params=self.params,
        )
