"""
Cached equal-weight rank composite factor.

Loads pre-computed sub-factors from NPZ cache file, applies configurable
preprocessing steps, then computes equal-weight rank composite.

Cache file format (produced by precompute_pool18.py):
  - symbols     : (N,)    stock codes
  - dates_int64 : (T,)    trading dates as int64
  - values      : (N,T,F) raw factor values
  - names       : (F,)    factor names
  - directions  : (F,)    factor directions (+1/-1)
  - industry_keys   : (N,) industry strings
  - market_cap      : (N,T) daily market cap

Subclasses set class attributes to configure behavior:
  - CACHE_PATH: path to NPZ cache file
  - FACTOR_INDICES: which factors to use (default: all)
  - PREPROCESS_STEPS: list of preprocessing steps
  - MIN_VALID_RATIO: minimum fraction of factors required (default: 1/3)
"""

import os
import sys
import logging
from typing import List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator
from factors.fundamental.composite.preprocessing import FactorPreprocessor

logger = logging.getLogger(__name__)

# Default cache path (workspace)
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'agent_draft', '202604', '20260403_新因子合成', 'pool18_cache.npz'
)


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


class EqualRankCached(FundamentalFactorCalculator):
    """Cached equal-weight rank composite, configurable via class attributes.

    Subclass and override these attributes:
        FACTOR_NAME: str          - factor identifier
        CACHE_PATH: str           - path to NPZ cache file
        FACTOR_INDICES: list[int] - which factor columns to use (None=all)
        PREPROCESS_STEPS: list    - e.g. ['industry_neutral', 'size_neutral', 'zscore']
        SIZE_FACTOR_INDICES: set  - indices to skip size_neutral (relative to FACTOR_INDICES)
        MIN_VALID_RATIO: float    - min fraction of factors needed (default 1/3)
    """

    FACTOR_NAME: str = "EQUAL_RANK_CACHED"
    CACHE_PATH: str = DEFAULT_CACHE_PATH
    FACTOR_INDICES: Optional[List[int]] = None  # None = use all
    PREPROCESS_STEPS: List[str] = []
    SIZE_FACTOR_INDICES: Set[int] = set()
    MIN_VALID_RATIO: float = 1.0 / 3.0
    SKIP_RANK: bool = True  # default: skip rank, use zscore for normalization

    @property
    def name(self) -> str:
        return self.FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return self.FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "method": "equal_rank_cached",
            "preprocessing": self.PREPROCESS_STEPS,
            "direction": 1,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        cache_path = os.path.abspath(self.CACHE_PATH)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                f"Run precompute_pool18.py first."
            )

        # Load cache
        logger.info(f"{self.FACTOR_NAME}: loading cache from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        symbols = data['symbols']
        dates = data['dates_int64'].astype('datetime64[ns]')
        all_values = data['values']       # (N, T, F)
        all_names = data['names']         # (F,)
        all_directions = data['directions']  # (F,)
        industry_keys = data['industry_keys']  # (N,)
        market_cap = data['market_cap']   # (N, T)

        N, T, F_all = all_values.shape

        # Select factor subset
        if self.FACTOR_INDICES is not None:
            indices = self.FACTOR_INDICES
        else:
            indices = list(range(F_all))

        values = all_values[:, :, indices]        # (N, T, F)
        names = all_names[indices]
        directions = all_directions[indices]
        F = len(indices)

        logger.info(f"{self.FACTOR_NAME}: N={N}, T={T}, F={F}, "
                    f"factors={list(names)}")

        # Apply preprocessing
        if self.PREPROCESS_STEPS:
            logger.info(f"{self.FACTOR_NAME}: preprocessing={self.PREPROCESS_STEPS}")

            preprocessor = FactorPreprocessor(
                steps=self.PREPROCESS_STEPS,
                size_factor_indices=self.SIZE_FACTOR_INDICES,
            )

            # Set auxiliary data if needed
            if 'industry_neutral' in self.PREPROCESS_STEPS:
                ind_map = {s: ind for s, ind in zip(symbols, industry_keys) if ind}
                preprocessor.set_industry_map(ind_map)

            if 'size_neutral' in self.PREPROCESS_STEPS:
                preprocessor.set_market_cap(market_cap, symbols)

            values = preprocessor.transform(values, symbols)
            logger.info(f"{self.FACTOR_NAME}: preprocessing done")

        # Normalize, flip, accumulate
        min_valid = max(1, int(F * self.MIN_VALID_RATIO))
        val_sum = np.zeros((N, T), dtype=np.float64)
        valid_count = np.zeros((N, T), dtype=np.float64)

        for f_idx in range(F):
            if self.SKIP_RANK:
                # Raw values, only flip direction
                v = values[:, :, f_idx].copy()
                if directions[f_idx] == -1:
                    normed = np.where(np.isnan(v), np.nan, -v)
                else:
                    normed = v
            else:
                normed = _cross_sectional_rank(values[:, :, f_idx])
                if directions[f_idx] == -1:
                    normed = np.where(np.isnan(normed), np.nan, 1.0 - normed)

            valid_mask = ~np.isnan(normed)
            val_sum = np.where(valid_mask, val_sum + normed, val_sum)
            valid_count = np.where(valid_mask, valid_count + 1.0, valid_count)

        composite = np.where(
            valid_count >= min_valid,
            val_sum / valid_count,
            np.nan,
        )

        logger.info(f"{self.FACTOR_NAME}: nan_ratio={np.isnan(composite).mean():.1%}, "
                    f"min_valid={min_valid}")

        return FactorData(
            values=composite,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )


# ======================================================================
# Pre-defined configurations
# ======================================================================

class EQUAL_RANK_POOL18_CACHED(EqualRankCached):
    """18-factor equal-weight rank, no preprocessing."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_CACHED"


class EQUAL_RANK_POOL18_INDNEU_CACHED(EqualRankCached):
    """18-factor equal-weight rank, industry neutralized."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_INDNEU_CACHED"
    PREPROCESS_STEPS = ['industry_neutral']


class EQUAL_RANK_POOL18_SIZENEU_CACHED(EqualRankCached):
    """18-factor equal-weight rank, size neutralized (SIZE factor skipped)."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_SIZENEU_CACHED"
    PREPROCESS_STEPS = ['size_neutral']
    SIZE_FACTOR_INDICES = {1}  # valuation_SIZE at index 1


class EQUAL_RANK_POOL18_INDSIZENEU_CACHED(EqualRankCached):
    """18-factor equal-weight rank, industry + size neutralized."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_INDSIZENEU_CACHED"
    PREPROCESS_STEPS = ['industry_neutral', 'size_neutral']
    SIZE_FACTOR_INDICES = {1}


class EQUAL_RANK_POOL18_INDNEU_ZSCORE_CACHED(EqualRankCached):
    """18-factor equal-weight rank, industry neutralized + zscore."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_INDNEU_ZSCORE_CACHED"
    PREPROCESS_STEPS = ['industry_neutral', 'zscore']


class EQUAL_RANK_POOL18_FULL_CACHED(EqualRankCached):
    """18-factor equal-weight rank, industry + size + zscore + orthogonalize."""
    FACTOR_NAME = "EQUAL_RANK_POOL18_FULL_CACHED"
    PREPROCESS_STEPS = ['industry_neutral', 'size_neutral', 'zscore', 'orthogonalize']
    SIZE_FACTOR_INDICES = {1}


# --- Strong 4 factors only ---

STRONG_4_INDICES = [0, 1, 2, 3]

class EQUAL_RANK_STRONG4_CACHED(EqualRankCached):
    """4 strong factors only, no preprocessing."""
    FACTOR_NAME = "EQUAL_RANK_STRONG4_CACHED"
    FACTOR_INDICES = STRONG_4_INDICES


class EQUAL_RANK_STRONG4_INDNEU_CACHED(EqualRankCached):
    """4 strong factors, industry neutralized."""
    FACTOR_NAME = "EQUAL_RANK_STRONG4_INDNEU_CACHED"
    FACTOR_INDICES = STRONG_4_INDICES
    PREPROCESS_STEPS = ['industry_neutral']
