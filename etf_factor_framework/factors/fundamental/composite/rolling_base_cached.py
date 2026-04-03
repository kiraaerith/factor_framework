"""
Cached rolling composite base class.

Same pipeline as rolling_base.py but loads sub-factors from NPZ cache
instead of computing them from scratch. Saves ~10 minutes per run.

Subclasses set class attributes:
    FACTOR_NAME, MODEL_CLASS, MODEL_PARAMS, TRAINER_PARAMS, RANK_LABELS
    CACHE_PATH, FACTOR_INDICES, PREPROCESS_STEPS, SIZE_FACTOR_INDICES
"""

import os
import sys
import logging
from typing import List, Optional, Set, Type

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator
from factors.fundamental.composite.preprocessing import FactorPreprocessor
from rolling_model import PreparedData, RollingTrainer
from rolling_model.base import ModelWrapper

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'agent_draft', '202604', '20260403_新因子合成', 'pool18_cache.npz'
)

FORWARD_DAYS = 20


def _cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per column. NaN stays NaN. Result in [0,1]."""
    result = np.full_like(arr, np.nan)
    N, T = arr.shape
    # Loop: ~T iterations (~2430 trading days)
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
    """Forward N-day returns. Last `days` columns are NaN."""
    N, T = close.shape
    fwd = np.full((N, T), np.nan)
    if T <= days:
        return fwd
    with np.errstate(divide='ignore', invalid='ignore'):
        fwd[:, :-days] = close[:, days:] / close[:, :-days] - 1.0
    return fwd


class RollingCompositeCached(FundamentalFactorCalculator):
    """Cached rolling composite base. Subclasses configure via class attributes."""

    FACTOR_NAME: str = "ROLLING_CACHED"
    FACTOR_DIRECTION: int = 1
    MODEL_CLASS: Type[ModelWrapper] = None
    MODEL_PARAMS: dict = {}
    TRAINER_PARAMS: dict = {}
    RANK_LABELS: bool = False
    CACHE_PATH: str = DEFAULT_CACHE_PATH
    FACTOR_INDICES: Optional[List[int]] = None
    PREPROCESS_STEPS: List[str] = []
    SIZE_FACTOR_INDICES: Set[int] = set()
    SKIP_RANK: bool = True  # default: skip rank, feed raw/preprocessed values to model

    @property
    def name(self) -> str:
        return self.FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return self.FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "model": self.MODEL_CLASS.__name__ if self.MODEL_CLASS else "unknown",
            "rank_labels": self.RANK_LABELS,
            "preprocessing": self.PREPROCESS_STEPS,
            "forward_days": FORWARD_DAYS,
            "direction": self.FACTOR_DIRECTION,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        cache_path = os.path.abspath(self.CACHE_PATH)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\nRun precompute_pool18.py first."
            )

        # --- Step 1: load from cache ---
        logger.info(f"{self.FACTOR_NAME}: loading cache")
        data = np.load(cache_path, allow_pickle=True)

        symbols = data['symbols']
        dates = data['dates_int64'].astype('datetime64[ns]')
        all_values = data['values']          # (N, T, F_all)
        all_names = data['names']
        all_directions = data['directions']
        industry_keys = data['industry_keys']
        market_cap = data['market_cap']

        N, T, F_all = all_values.shape

        # Select factor subset
        indices = self.FACTOR_INDICES if self.FACTOR_INDICES is not None else list(range(F_all))
        values = all_values[:, :, indices]
        names = all_names[indices]
        directions = all_directions[indices]
        F = len(indices)

        logger.info(f"{self.FACTOR_NAME}: N={N}, T={T}, F={F}")

        # --- Step 2: optional preprocessing ---
        if self.PREPROCESS_STEPS:
            preprocessor = FactorPreprocessor(
                steps=self.PREPROCESS_STEPS,
                size_factor_indices=self.SIZE_FACTOR_INDICES,
            )
            if 'industry_neutral' in self.PREPROCESS_STEPS:
                ind_map = {s: ind for s, ind in zip(symbols, industry_keys) if ind}
                preprocessor.set_industry_map(ind_map)
            if 'size_neutral' in self.PREPROCESS_STEPS:
                preprocessor.set_market_cap(market_cap, symbols)
            values = preprocessor.transform(values, symbols)

        # --- Step 3: rank and flip (optional) ---
        features_nt = np.full((N, T, F), np.nan)
        if self.SKIP_RANK:
            # Feed raw/preprocessed values directly to model, only flip direction
            for f_idx in range(F):
                v = values[:, :, f_idx]
                if directions[f_idx] == -1:
                    v = np.where(np.isnan(v), np.nan, -v)
                features_nt[:, :, f_idx] = v
        else:
            # Cross-sectional rank then flip
            for f_idx in range(F):
                ranked = _cross_sectional_rank(values[:, :, f_idx])
                if directions[f_idx] == -1:
                    ranked = np.where(np.isnan(ranked), np.nan, 1.0 - ranked)
                features_nt[:, :, f_idx] = ranked

        features = np.transpose(features_nt, (1, 0, 2))  # (T, N, F)

        # --- Step 4: forward return labels ---
        from data.stock_data_loader import StockDataLoader

        loader = StockDataLoader()
        ohlcv = loader.load_ohlcv(
            start_date=str(dates[0])[:10],
            end_date=str(dates[-1])[:10],
            use_adjusted=True,
        )

        ohlcv_sym_to_idx = {s: i for i, s in enumerate(ohlcv.symbols.tolist())}
        ohlcv_date_to_idx = {int(d): i for i, d in enumerate(ohlcv.dates.astype('int64'))}

        valid_syms = [s for s in symbols if s in ohlcv_sym_to_idx]
        valid_sym_indices_ohlcv = np.array([ohlcv_sym_to_idx[s] for s in valid_syms])
        valid_sym_indices_common = np.array([
            np.searchsorted(symbols, s) for s in valid_syms
        ])

        dates_int = dates.astype('int64')
        valid_date_mask = np.array([int(d) in ohlcv_date_to_idx for d in dates_int])
        valid_date_indices_ohlcv = np.array([
            ohlcv_date_to_idx[int(d)] for d in dates_int[valid_date_mask]
        ])

        close_aligned = np.full((N, T), np.nan)
        close_sub = ohlcv.close[np.ix_(valid_sym_indices_ohlcv, valid_date_indices_ohlcv)]
        close_aligned[np.ix_(valid_sym_indices_common, np.where(valid_date_mask)[0])] = close_sub

        fwd_returns = _compute_forward_returns(close_aligned, FORWARD_DAYS)

        if self.RANK_LABELS:
            fwd_returns = _cross_sectional_rank(fwd_returns)

        labels = fwd_returns.T  # (T, N)

        # --- Step 5: rolling model ---
        feature_names = list(names)
        prepared = PreparedData(
            features=features,
            labels=labels,
            dates=dates,
            symbols=symbols,
            feature_names=feature_names,
        )

        trainer = RollingTrainer(**self.TRAINER_PARAMS)
        factor_values_tn = trainer.run(prepared, self.MODEL_CLASS, self.MODEL_PARAMS)

        factor_values = factor_values_tn.T  # (N, T)

        return FactorData(
            values=factor_values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
