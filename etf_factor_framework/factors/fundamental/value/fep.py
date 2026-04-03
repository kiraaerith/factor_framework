"""
FEP (Fundamental Equity Price ratio) factor.

FEP = FE / mc_daily

FE (Fundamental Equity) is the intrinsic value estimated via a VAR(1) model
using the Clean Surplus Accounting framework (Edwards-Bell-Ohlson extended):

  FE_t = BE_t * (1 + sum_{h=1}^{10} (exp(CSprof_{t+h} - BEg_{t+h}) - 1)
                                   * exp(cum_BEg_{t+h} - h*dr))

The 12-variable VAR state vector is re-fitted each year using all available
historical annual data.  Factor becomes effective at Q4 annual report
disclosure date (report_date), then FE_t is divided by current-day mc
(Method A: dynamic denominator).

Data sources:
  - lixinger.financial_statements : annual financial data (Q4 year-end +
                                     annual flow sums from quarterly data)
  - lixinger.fundamental           : daily mc

Factor direction: positive (higher FEP = more undervalued)
Mainboard filter: SHSE.60xxxx, SZSE.00xxxx only

Notes:
  - Annual data is NOT directly available (db has fs_type='q' only).
    Q4 point-in-time fields: filter date LIKE '%-12-31%'.
    Annual flow fields: sum Q1-Q4 within each year.
  - Leakage prevention: VAR uses data up to year t; FEP_t effective
    at report_date_t (April of year t+1).
  - If > 6 of 12 VAR variables are NaN: FEP = NaN for that stock/year.
  - Missing VAR variables imputed with cross-sectional mean.
  - CSprof clipped to [-5, 5]; cum_BEg clipped to [-5, 5].
  - VAR Gamma eigenvalue check: scale down if max modulus > 1.5.
"""

import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData, lixinger_code_to_symbol
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "FEP"
FACTOR_DIRECTION = 1       # positive: higher FEP (more undervalued) is better
DR = 0.10                  # fixed annual discount rate
N_FORECAST = 10            # forecast horizon (years)
MAX_MISSING_VARS = 6       # max NaN variables allowed in state vector
CSPROFIT_CLIP = 5.0        # clip CSprof predictions to [-clip, clip]
CUMBEG_CLIP = 5.0          # clip cumulative BEg to [-clip, clip]
BEG_CLIP = 5.0             # clip individual BEg predictions
MIN_MC = 1.0               # min mc (yuan) below which FEP = NaN
MIN_TRAIN_PAIRS = 50       # min training pairs to fit Gamma
GAMMA_EIGVAL_THRESH = 1.5  # scale Gamma if max eigenvalue modulus > this
HISTORY_YEARS = 15         # years of extra history to load for VAR training

VAR_COLS = [
    'bm', 'POy', 'Yy', 'BEg', 'Ag', 'Yg',
    'CSprof', 'Roe', 'Gprof', 'Mlev', 'Blev', 'Cash',
]


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _impute_report_date(row) -> pd.Timestamp:
    d = row['date']
    if d.month == 3:
        return d + pd.DateOffset(months=1)
    elif d.month == 6:
        return d + pd.DateOffset(months=2)
    elif d.month == 9:
        return d + pd.DateOffset(months=1)
    else:
        return d + pd.DateOffset(months=4)


class FEP(FundamentalFactorCalculator):
    """
    Fundamental Equity Price ratio.

    FEP = FE / mc_daily  where FE is from a 12-variable VAR(1) model.
    Annual FE is computed from year-end financial data; the daily factor
    divides this fixed FE by the current day's total market cap (Method A).
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {'direction': FACTOR_DIRECTION}

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_fs_data(self, fd: FundamentalData) -> pd.DataFrame:
        """Load financial_statements with extended history."""
        load_start = (fd.start_date - pd.DateOffset(years=HISTORY_YEARS)).date()
        load_end = fd.end_date.date()

        if fd._stock_codes:
            codes_str = "','".join(str(c) for c in fd._stock_codes)
            codes_filter = f"AND stock_code IN ('{codes_str}')"
        else:
            codes_filter = ''

        conn = sqlite3.connect(fd._lixinger_db)
        df = pd.read_sql_query(
            f"""
            SELECT stock_code, date, report_date,
                   q_bs_tetoshopc_t, q_bs_ta_t, q_bs_tl_t,
                   q_m_roe_t, q_ps_gp_m_t, q_bs_cabb_t,
                   q_ps_npatoshopc_c, q_ps_toi_c, q_cfs_cpfdapdoi_c
            FROM financial_statements
            WHERE date BETWEEN '{load_start}' AND '{load_end}'
            {codes_filter}
            """,
            conn,
        )
        conn.close()
        return df

    def _load_year_end_mc(self, fd: FundamentalData) -> pd.DataFrame:
        """Load year-end (last December trading day) mc per stock per year."""
        load_start = (fd.start_date - pd.DateOffset(years=HISTORY_YEARS)).date()
        load_end = fd.end_date.date()

        if fd._stock_codes:
            codes_str = "','".join(str(c) for c in fd._stock_codes)
            codes_filter = f"AND stock_code IN ('{codes_str}')"
        else:
            codes_filter = ''

        conn = sqlite3.connect(fd._lixinger_db)
        df = pd.read_sql_query(
            f"""
            SELECT stock_code, substr(date, 1, 10) AS date_str, mc
            FROM fundamental
            WHERE substr(date, 6, 2) = '12'
              AND substr(date, 1, 10) BETWEEN '{load_start}' AND '{load_end}'
              AND mc IS NOT NULL
            {codes_filter}
            """,
            conn,
        )
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=['stock_code', 'year', 'mc_ye'])

        df['year'] = df['date_str'].str[:4].astype(int)
        df = df.sort_values(['stock_code', 'date_str'])
        df_ye = df.groupby(['stock_code', 'year'], as_index=False)['mc'].last()
        df_ye.rename(columns={'mc': 'mc_ye'}, inplace=True)
        return df_ye

    # ------------------------------------------------------------------
    # Annual data preparation
    # ------------------------------------------------------------------

    def _build_annual_data(self, df_fs: pd.DataFrame) -> pd.DataFrame:
        """
        Build annual dataset from quarterly financial_statements.

        Returns DataFrame with one row per (stock_code, year) containing:
          Q4 point-in-time fields: BE, TA, TL, Roe, GPM, Cash_raw, report_date
          Annual flow sums:        NP, Rev, Div
        """
        df = df_fs.copy()

        # Parse dates
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce').dt.tz_localize(None)

        # Impute missing report_dates
        null_mask = df['report_date'].isna()
        if null_mask.any():
            df.loc[null_mask, 'report_date'] = df.loc[null_mask].apply(
                _impute_report_date, axis=1
            )

        # Remove records where period-end > report_date (data anomaly)
        df = df[df['date'] <= df['report_date']].copy()

        df['year'] = df['date'].dt.year

        # ---- Q4 point-in-time fields (period-end month == 12) ----
        df_q4 = df[df['date'].dt.month == 12].copy()
        # Keep latest report per (stock_code, year)
        df_q4 = df_q4.sort_values(['stock_code', 'year', 'report_date'])
        df_q4 = df_q4.groupby(['stock_code', 'year'], as_index=False).last()

        df_annual = df_q4[[
            'stock_code', 'year', 'report_date',
            'q_bs_tetoshopc_t', 'q_bs_ta_t', 'q_bs_tl_t',
            'q_m_roe_t', 'q_ps_gp_m_t', 'q_bs_cabb_t',
        ]].rename(columns={
            'q_bs_tetoshopc_t': 'BE',
            'q_bs_ta_t':        'TA',
            'q_bs_tl_t':        'TL',
            'q_m_roe_t':        'Roe',
            'q_ps_gp_m_t':      'GPM',
            'q_bs_cabb_t':      'Cash_raw',
        })

        # ---- Annual flow sums (all quarters in the year) ----
        df_flow = (
            df.groupby(['stock_code', 'year'], as_index=False)
            .agg(
                NP=('q_ps_npatoshopc_c', 'sum'),
                Rev=('q_ps_toi_c', 'sum'),
                Div=('q_cfs_cpfdapdoi_c', 'sum'),
            )
        )

        # Merge
        df_annual = df_annual.merge(df_flow, on=['stock_code', 'year'], how='left')

        return df_annual

    # ------------------------------------------------------------------
    # VAR variable computation
    # ------------------------------------------------------------------

    def _compute_var_variables(
        self, df_annual: pd.DataFrame, df_mc_ye: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute 12 VAR state variables for each (stock, year) observation.
        All monetary values in yuan (native lixinger units).
        """
        df = df_annual.merge(df_mc_ye, on=['stock_code', 'year'], how='left')

        # Add symbol + mainboard filter
        df['symbol'] = df['stock_code'].apply(lixinger_code_to_symbol)
        mb_mask = np.array([_is_mainboard(s) for s in df['symbol']])
        df = df[mb_mask].reset_index(drop=True)

        # Sort for groupby lag operations
        df = df.sort_values(['symbol', 'year']).reset_index(drop=True)

        # ---- Lagged values (previous year) ----
        df['BE_lag'] = df.groupby('symbol')['BE'].shift(1)
        df['TA_lag'] = df.groupby('symbol')['TA'].shift(1)
        df['NP_lag'] = df.groupby('symbol')['NP'].shift(1)

        # ---- Log growth rates ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            df['BEg'] = np.where(
                (df['BE'] > 0) & (df['BE_lag'] > 0),
                np.log(df['BE'] / df['BE_lag']),
                np.nan,
            )
            df['Ag'] = np.where(
                (df['TA'] > 0) & (df['TA_lag'] > 0),
                np.log(df['TA'] / df['TA_lag']),
                np.nan,
            )
            df['Yg'] = np.where(
                (df['NP_lag'] > 0) & (df['NP'] > 0),
                np.log(df['NP'] / df['NP_lag']),
                np.nan,
            )

        # ---- CSE and CSprof ----
        df['CSE'] = df['Div'] + (df['BE'] - df['BE_lag'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            csprof_raw = np.where(
                df['BE_lag'] > 0,
                np.log(1.0 + df['CSE'] / df['BE_lag']),
                np.nan,
            )
        df['CSprof'] = np.clip(csprof_raw, -CSPROFIT_CLIP, CSPROFIT_CLIP)

        # ---- Valuation ratios (divided by mc_ye, same yuan units) ----
        mc_ye = df['mc_ye'].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df['bm']  = np.where(mc_ye > 0, df['BE'].values  / mc_ye, np.nan)
            df['POy'] = np.where(mc_ye > 0, df['Div'].values / mc_ye, np.nan)
            df['Yy']  = np.where(mc_ye > 0, df['NP'].values  / mc_ye, np.nan)

        # ---- Profitability ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df['Gprof'] = np.where(
                df['TA'].values > 0,
                df['Rev'].values * df['GPM'].values / df['TA'].values,
                np.nan,
            )

        # ---- Leverage ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df['Mlev'] = np.where(
                mc_ye > 0,
                (df['TL'].values + mc_ye) / mc_ye,
                np.nan,
            )
            df['Blev'] = np.where(
                df['BE'].values > 0,
                df['TL'].values / df['BE'].values,
                np.nan,
            )

        # ---- Cash ratio ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df['Cash'] = np.where(
                df['TA'].values > 0,
                df['Cash_raw'].values / df['TA'].values,
                np.nan,
            )

        return df

    # ------------------------------------------------------------------
    # VAR model fitting
    # ------------------------------------------------------------------

    def _fit_var_gamma(self, df_vars: pd.DataFrame) -> np.ndarray:
        """
        Fit VAR(1) Gamma matrix from panel data.

        Creates (S_{t-1}, S_t) pairs for all consecutive years per stock,
        then runs OLS column-by-column.

        Returns:
            Gamma: (12, 12) float64 array, or None if insufficient data.
        """
        # Create lag pairs: increment year by 1 in the lag table,
        # then merge on (symbol, year) to get (S_{t-1} matched to S_t)
        lag_cols = [c + '_lag' for c in VAR_COLS]
        df_lag = df_vars[['symbol', 'year'] + VAR_COLS].copy()
        df_lag = df_lag.rename(columns={c: c + '_lag' for c in VAR_COLS})
        df_lag['year'] = df_lag['year'] + 1

        pairs = df_vars[['symbol', 'year'] + VAR_COLS].merge(
            df_lag, on=['symbol', 'year'], how='inner'
        )

        # Keep only rows with no NaN in any current or lagged variable
        all_cols = VAR_COLS + lag_cols
        valid_mask = pairs[all_cols].notna().all(axis=1)
        pairs = pairs[valid_mask]

        if len(pairs) < MIN_TRAIN_PAIRS:
            return None

        X = pairs[lag_cols].values.astype(np.float64)   # (n, 12)
        Y = pairs[VAR_COLS].values.astype(np.float64)   # (n, 12)

        # Fit Gamma row by row: Y[:, j] = X @ Gamma[j, :] + eps
        Gamma = np.zeros((12, 12), dtype=np.float64)
        for j in range(12):
            gamma_j, _, _, _ = np.linalg.lstsq(X, Y[:, j], rcond=None)
            Gamma[j, :] = gamma_j

        # Stabilize: scale Gamma if max eigenvalue modulus > threshold
        eigvals = np.linalg.eigvals(Gamma)
        max_mod = float(np.max(np.abs(eigvals)))
        if max_mod > GAMMA_EIGVAL_THRESH:
            Gamma = Gamma * (GAMMA_EIGVAL_THRESH / max_mod)

        return Gamma

    # ------------------------------------------------------------------
    # Main calculate
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Calculate FEP daily panel.

        Returns:
            FactorData: N stocks x T days, values = FE / mc (float64)
        """
        trading_dates = fundamental_data._get_trading_dates()
        if trading_dates.tz is not None:
            trading_dates = trading_dates.tz_localize(None)

        # ---- Load raw data ----
        print('  [FEP] Loading financial_statements data...')
        df_fs = self._load_fs_data(fundamental_data)
        if df_fs.empty:
            empty = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty, symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name, factor_type=self.factor_type, params=self.params,
            )

        print(f'  [FEP] fs rows: {len(df_fs)}, stocks: {df_fs["stock_code"].nunique()}')

        # ---- Build annual data ----
        df_annual = self._build_annual_data(df_fs)
        print(f'  [FEP] Annual obs: {len(df_annual)}')

        # ---- Load year-end market cap ----
        print('  [FEP] Loading year-end market cap...')
        df_mc_ye = self._load_year_end_mc(fundamental_data)

        # ---- Compute 12 VAR variables ----
        df_vars = self._compute_var_variables(df_annual, df_mc_ye)
        print(f'  [FEP] VAR obs after mainboard filter: {len(df_vars)}')

        # ---- Per-year VAR fitting and FE computation ----
        all_years = sorted(df_vars['year'].unique())
        fe_records = []

        beg_idx = VAR_COLS.index('BEg')
        csp_idx = VAR_COLS.index('CSprof')

        for year_t in all_years:
            # Training data: all years <= year_t
            df_train = df_vars[df_vars['year'] <= year_t].copy()

            Gamma = self._fit_var_gamma(df_train)
            if Gamma is None:
                continue

            # Current year state vectors
            df_curr = df_vars[df_vars['year'] == year_t].reset_index(drop=True)
            if df_curr.empty:
                continue

            # Build state matrix
            S_mat = df_curr[VAR_COLS].values.astype(np.float64)  # (n, 12)

            # Impute with cross-sectional mean
            n_missing = np.isnan(S_mat).sum(axis=1)
            col_means = np.nanmean(S_mat, axis=0)
            for j in range(12):
                nan_rows = np.isnan(S_mat[:, j])
                if nan_rows.any():
                    fill_val = col_means[j]
                    if np.isnan(fill_val):
                        fill_val = 0.0
                    S_mat[nan_rows, j] = fill_val

            # Validity mask
            valid_mask = (
                (n_missing <= MAX_MISSING_VARS) &
                ~np.isnan(S_mat).any(axis=1) &
                (df_curr['BE'].values > 0) &
                ~np.isnan(df_curr['BE'].values) &
                ~pd.isnull(df_curr['report_date'].values)
            )

            if not valid_mask.any():
                continue

            S_valid = S_mat[valid_mask]                            # (n_v, 12)
            BE_valid = df_curr['BE'].values[valid_mask]
            syms_valid = df_curr['symbol'].values[valid_mask]
            rdates_valid = df_curr['report_date'].values[valid_mask]

            # Propagate state vectors 10 steps ahead
            S_pred = [S_valid]
            for _ in range(N_FORECAST):
                S_pred.append(S_pred[-1] @ Gamma.T)  # (n_v, 12)

            # Compute FE_sum (vectorized)
            fe_sum = np.zeros(len(BE_valid), dtype=np.float64)
            cum_beg = np.zeros(len(BE_valid), dtype=np.float64)

            for h in range(1, N_FORECAST + 1):
                beg_h = np.clip(S_pred[h][:, beg_idx], -BEG_CLIP, BEG_CLIP)
                csp_h = np.clip(S_pred[h][:, csp_idx], -CSPROFIT_CLIP, CSPROFIT_CLIP)
                cum_beg += beg_h
                cum_beg_c = np.clip(cum_beg, -CUMBEG_CLIP, CUMBEG_CLIP)
                excess = np.exp(csp_h - beg_h) - 1.0
                discount = np.exp(cum_beg_c - h * DR)
                fe_sum += excess * discount

            FE_vec = BE_valid * (1.0 + fe_sum)
            # Negative FE is economically meaningless (discount sum < -1); set to NaN
            FE_vec = np.where(
                np.isinf(FE_vec) | np.isnan(FE_vec) | (FE_vec < 0), np.nan, FE_vec
            )

            # Convert report_date values to pandas Timestamps for comparison
            for i in range(len(BE_valid)):
                fe_val = FE_vec[i]
                if np.isnan(fe_val):
                    continue
                rdate = pd.Timestamp(rdates_valid[i])
                if rdate > fundamental_data.end_date:
                    continue
                fe_records.append({
                    'symbol':      syms_valid[i],
                    'report_date': rdate,
                    'FE':          fe_val,
                })

        print(f'  [FEP] Total FE records: {len(fe_records)}')

        if not fe_records:
            empty = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty, symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name, factor_type=self.factor_type, params=self.params,
            )

        # ---- Forward-fill FE to daily trading dates ----
        df_fe = pd.DataFrame(fe_records)
        pivot_fe = df_fe.pivot_table(
            index='symbol', columns='report_date', values='FE', aggfunc='last'
        )

        all_dates = pivot_fe.columns.union(trading_dates).sort_values()
        pivot_fe = pivot_fe.reindex(columns=all_dates)
        panel_fe = pivot_fe.ffill(axis=1).reindex(columns=trading_dates)

        # ---- Load daily mc panel; apply mainboard filter ----
        mc_vals, mc_syms, mc_dates = fundamental_data.get_market_cap_panel()

        mb_mc = np.array([_is_mainboard(s) for s in mc_syms])
        mc_vals = mc_vals[mb_mc]
        mc_syms = np.array(mc_syms)[mb_mc]

        mc_dates_idx = pd.DatetimeIndex(mc_dates)
        if mc_dates_idx.tz is not None:
            mc_dates_idx = mc_dates_idx.tz_localize(None)
        df_mc = pd.DataFrame(mc_vals, index=mc_syms, columns=mc_dates_idx)

        # ---- Align to union of symbols ----
        all_syms = sorted(set(panel_fe.index.tolist()) | set(mc_syms.tolist()))
        panel_fe = panel_fe.reindex(index=all_syms, columns=trading_dates)
        df_mc    = df_mc.reindex(index=all_syms, columns=trading_dates)

        arr_fe = panel_fe.values.astype(np.float64)
        arr_mc = df_mc.values.astype(np.float64)

        # ---- FEP = FE / mc ----
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fep = np.where(
                (arr_mc > MIN_MC) & ~np.isnan(arr_mc) & ~np.isnan(arr_fe),
                arr_fe / arr_mc,
                np.nan,
            )
        fep = np.where(np.isinf(fep), np.nan, fep).astype(np.float64)

        nan_ratio = np.isnan(fep).mean() if fep.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(f'FEP NaN ratio is high: {nan_ratio:.1%}')

        symbols_arr = np.array(all_syms)
        dates_arr   = np.array(trading_dates, dtype='datetime64[ns]')

        return FactorData(
            values=fep,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python fep.py)
# Uses full market with 2-year range to test VAR computation.
# ------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    print('=' * 70)
    print('FEP factor smoke test')
    print('=' * 70)

    TEST_START = '2022-01-01'
    TEST_END   = '2024-12-31'

    print(f'\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)')
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print('\n[Step 2] Compute FEP factor')
    calculator = FEP()
    result = calculator.calculate(fd)

    print(f'\nFactor shape : {result.values.shape}')
    print(f'Symbols (first 5): {result.symbols[:5].tolist()}')
    print(f'Date range   : {pd.Timestamp(result.dates[0]).date()} ~ '
          f'{pd.Timestamp(result.dates[-1]).date()}')

    # ---- Smoke-test assertions ----
    assert result.values.ndim == 2, 'values must be 2-D'
    assert result.values.dtype == np.float64, \
        f'expected float64, got {result.values.dtype}'

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f'NaN ratio too high: {nan_ratio:.1%}'
    print(f'NaN ratio    : {nan_ratio:.1%}')

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), 'Factor contains inf values'
    print('[PASS] No inf values')

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all(
        (result.values == result2.values) | both_nan
    ), 'Idempotency failed'
    print('[PASS] Idempotency check passed')

    # Cross-section stats
    print('\nLast cross-section stats:')
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f'  N valid    : {len(valid_cs)}')
        print(f'  mean       : {valid_cs.mean():.6f}')
        print(f'  std        : {valid_cs.std():.6f}')
        print(f'  min        : {valid_cs.min():.4f}')
        print(f'  max        : {valid_cs.max():.4f}')
        print(f'  median     : {np.median(valid_cs):.6f}')

    # Sample stock values from the full-market result
    from factors.fundamental.fundamental_data import lixinger_code_to_symbol
    TEST_CODES = ['600519', '000858', '601318', '000333', '600036']
    print(f'\nSample values (5 test stocks, last 5 dates from full-market result):')
    for code in TEST_CODES:
        sym = lixinger_code_to_symbol(code)
        sym_list = result.symbols.tolist()
        if sym in sym_list:
            idx = sym_list.index(sym)
            last5 = result.values[idx, -5:]
            print(f'  {sym}: {np.round(last5, 6)}')
        else:
            print(f'  {sym}: not found in result')

    print(f'\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}')

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calculator, fd_leak)
        report.print_report()
        if report.has_leakage:
            leakage_found = True
            print(f"[FAIL] Leakage detected at split_ratio={sr}")
        else:
            print(f"[OK] No leakage at split_ratio={sr}")

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
