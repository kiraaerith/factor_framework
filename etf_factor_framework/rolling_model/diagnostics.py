"""
DiagnosticsTool: compute and store per-window training/validation metrics.

Called by RollingTrainer after each window. Computes metrics immediately
and writes to SQLite, then releases intermediate data. No accumulation.
"""

import io
import os
import sqlite3
from typing import Optional

import numpy as np


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS window_metrics (
    window_id           INTEGER PRIMARY KEY,
    train_start         TEXT,
    train_end           TEXT,
    val_start           TEXT,
    val_end             TEXT,
    train_ic            REAL,
    train_icir          REAL,
    train_rank_ic       REAL,
    train_rank_icir     REAL,
    train_sharpe        REAL,
    train_annual_return REAL,
    train_max_drawdown  REAL,
    val_ic              REAL,
    val_icir            REAL,
    val_rank_ic         REAL,
    val_rank_icir       REAL,
    val_sharpe          REAL,
    val_annual_return   REAL,
    val_max_drawdown    REAL
);

CREATE TABLE IF NOT EXISTS window_curves (
    window_id     INTEGER,
    curve_type    TEXT,
    data          BLOB,
    PRIMARY KEY (window_id, curve_type),
    FOREIGN KEY (window_id) REFERENCES window_metrics(window_id)
);
"""


def _ndarray_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_ndarray(blob: bytes) -> np.ndarray:
    buf = io.BytesIO(blob)
    return np.load(buf)


def _safe_nanmean(arr):
    if len(arr) == 0:
        return np.nan
    return np.nanmean(arr)


def _safe_nanstd(arr):
    if len(arr) < 2:
        return np.nan
    return np.nanstd(arr, ddof=1)


def _compute_ic_metrics(pred: np.ndarray, actual: np.ndarray):
    """Compute IC and RankIC metrics from (T, N) prediction and actual arrays.

    Returns dict with ic, icir, rank_ic, rank_icir.
    """
    T, N = pred.shape
    ics = []
    rank_ics = []

    # Loop: ~T次 (typically 126 for half-year validation)
    for t in range(T):
        p = pred[t]
        a = actual[t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() < 5:
            continue

        p_valid = p[mask]
        a_valid = a[mask]

        # Pearson IC
        corr = np.corrcoef(p_valid, a_valid)
        if np.isfinite(corr[0, 1]):
            ics.append(corr[0, 1])

        # Spearman RankIC (use scipy-free rank approach)
        p_rank = p_valid.argsort().argsort().astype(float)
        a_rank = a_valid.argsort().argsort().astype(float)
        rank_corr = np.corrcoef(p_rank, a_rank)
        if np.isfinite(rank_corr[0, 1]):
            rank_ics.append(rank_corr[0, 1])

    ics = np.array(ics) if ics else np.array([])
    rank_ics = np.array(rank_ics) if rank_ics else np.array([])

    ic_mean = _safe_nanmean(ics)
    ic_std = _safe_nanstd(ics)
    rank_ic_mean = _safe_nanmean(rank_ics)
    rank_ic_std = _safe_nanstd(rank_ics)

    return {
        'ic': ic_mean,
        'icir': ic_mean / ic_std if ic_std > 0 else np.nan,
        'rank_ic': rank_ic_mean,
        'rank_icir': rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan,
    }


def _compute_portfolio_metrics(pred: np.ndarray, actual: np.ndarray,
                               top_k: int = 50, annual_factor: float = 252.0):
    """Compute long portfolio metrics from (T, N) prediction and actual arrays.

    Selects top_k stocks by prediction each day, computes equal-weight returns.
    Returns dict with sharpe, annual_return, max_drawdown, and daily returns array.
    """
    T, N = pred.shape
    daily_returns = np.full(T, np.nan)

    k = min(top_k, N)

    # Loop: ~T次
    for t in range(T):
        p = pred[t]
        a = actual[t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() < k:
            continue

        indices = np.where(mask)[0]
        p_valid = p[indices]
        top_indices = indices[np.argsort(p_valid)[-k:]]
        daily_returns[t] = np.nanmean(a[top_indices])

    valid_returns = daily_returns[np.isfinite(daily_returns)]
    if len(valid_returns) < 2:
        return {
            'sharpe': np.nan,
            'annual_return': np.nan,
            'max_drawdown': np.nan,
            'daily_returns': daily_returns,
        }

    mean_ret = np.mean(valid_returns)
    std_ret = np.std(valid_returns, ddof=1)
    sharpe = mean_ret / std_ret * np.sqrt(annual_factor) if std_ret > 0 else np.nan
    annual_return = mean_ret * annual_factor

    cumulative = np.cumprod(1 + valid_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = np.min(drawdowns)

    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'daily_returns': daily_returns,
    }


def _compute_quantile_returns(pred: np.ndarray, actual: np.ndarray,
                              n_quantiles: int = 5):
    """Compute daily returns for n quantile portfolios.

    Returns (T, n_quantiles) array.
    """
    T, N = pred.shape
    quantile_returns = np.full((T, n_quantiles), np.nan)

    # Loop: ~T次
    for t in range(T):
        p = pred[t]
        a = actual[t]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() < n_quantiles * 2:
            continue

        indices = np.where(mask)[0]
        p_valid = p[indices]
        a_valid = a[indices]
        sorted_idx = np.argsort(p_valid)
        splits = np.array_split(sorted_idx, n_quantiles)

        for q, group in enumerate(splits):
            if len(group) > 0:
                quantile_returns[t, q] = np.mean(a_valid[group])

    return quantile_returns


class DiagnosticsTool:
    """Compute and store per-window diagnostics to SQLite.

    Usage:
        diag = DiagnosticsTool("path/to/diagnostics.db")
        diag.compute_and_save(window_id, window_info,
                              train_pred, train_y, val_pred, val_y)
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_CREATE_SQL)
        self.conn.commit()

    def compute_and_save(self,
                         window_id: int,
                         window_info: dict,
                         train_pred: np.ndarray,   # (T_train, N)
                         train_y: np.ndarray,       # (T_train, N)
                         val_pred: np.ndarray,       # (T_val, N)
                         val_y: np.ndarray           # (T_val, N)
                         ) -> None:
        """Compute all metrics and write to DB. Releases data after saving."""

        train_ic = _compute_ic_metrics(train_pred, train_y)
        train_port = _compute_portfolio_metrics(train_pred, train_y)

        val_ic = _compute_ic_metrics(val_pred, val_y)
        val_port = _compute_portfolio_metrics(val_pred, val_y)

        val_quantile = _compute_quantile_returns(val_pred, val_y)

        self.conn.execute(
            """INSERT INTO window_metrics VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )""",
            (
                window_id,
                window_info.get('train_start', ''),
                window_info.get('train_end', ''),
                window_info.get('val_start', ''),
                window_info.get('val_end', ''),
                train_ic['ic'], train_ic['icir'],
                train_ic['rank_ic'], train_ic['rank_icir'],
                train_port['sharpe'], train_port['annual_return'],
                train_port['max_drawdown'],
                val_ic['ic'], val_ic['icir'],
                val_ic['rank_ic'], val_ic['rank_icir'],
                val_port['sharpe'], val_port['annual_return'],
                val_port['max_drawdown'],
            )
        )

        # Store validation curves
        self.conn.execute(
            "INSERT INTO window_curves VALUES (?, ?, ?)",
            (window_id, 'val_returns', _ndarray_to_blob(val_port['daily_returns']))
        )
        self.conn.execute(
            "INSERT INTO window_curves VALUES (?, ?, ?)",
            (window_id, 'val_quantile_returns', _ndarray_to_blob(val_quantile))
        )

        self.conn.commit()

    def close(self):
        self.conn.close()

    @staticmethod
    def load_curve(db_path: str, window_id: int, curve_type: str) -> Optional[np.ndarray]:
        """Utility to read a stored curve back from the database."""
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT data FROM window_curves WHERE window_id=? AND curve_type=?",
            (window_id, curve_type)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return _blob_to_ndarray(row[0])
