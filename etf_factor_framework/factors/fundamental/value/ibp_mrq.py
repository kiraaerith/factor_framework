"""
IBP_MRQ factor (Intangible-adjusted Book-to-Price, Most Recent Quarter)

Extends RDBP_MRQ by also capitalizing organizational capital from admin expenses.

Formula:
  KC_MRQ         = sum_{i=0}^{N} RD_quarter_i * 0.9^(i/4)
  OC_MRQ         = sum_{i=0}^{N} (admin_quarter_i * 0.30) * 0.9^(i/4)
  adj_equity_mrq = q_bs_tetoshopc_t + KC_MRQ + OC_MRQ          (yuan)
  IBP_MRQ        = adj_equity_mrq / mc                           (yuan / yuan)

  where:
    q_bs_tetoshopc_t : equity attributable to parent shareholders (yuan, MRQ)
    q_ps_rade_c      : single-quarter R&D expense (yuan, already single-quarter in lixinger)
    q_ps_ae_c        : single-quarter admin expense (yuan, already single-quarter in lixinger)
    mc               : total market cap (yi-yuan/100M, daily) from lixinger.fundamental
    N                : look-back window = 40 quarters (10 years)
    annual decay     : 10%  =>  quarterly decay factor = 0.9^(i/4)
    OC ratio         : 30% of admin expense is capitalized (Peters & Taylor 2017)

Data fields:
  - q_bs_tetoshopc_t : lixinger.financial_statements (quarterly, forward-filled via report_date)
  - q_ps_rade_c      : lixinger.financial_statements (quarterly single-quarter R&D)
  - q_ps_ae_c        : lixinger.financial_statements (quarterly single-quarter admin expense)
  - mc               : lixinger.fundamental (daily, unit: yuan)

Factor direction: positive (higher adj BP = cheaper value = better)
Factor category: value - intangible-adjusted BP

Notes:
  - q_ps_rade_c NaN => treated as 0 (firm has no R&D, KC = 0)
  - q_ps_ae_c  NaN => treated as 0
  - Negative values for either => clipped to 0 (data anomaly)
  - q_bs_tetoshopc_t NaN => IBP_MRQ = NaN
  - mc = 0 or NaN => IBP_MRQ = NaN
  - Both adj_equity and mc are in yuan; ratio is dimensionless
  - Extended history: 12 years before start_date to accumulate KC/OC properly
  - Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only
"""

import os
import re
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData, lixinger_code_to_symbol
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME   = "IBP_MRQ"
FACTOR_DIRECTION = 1        # positive: higher adj book-to-price is better
N_QUARTERS    = 40          # look-back window (10 years)
ANNUAL_DECAY  = 0.90        # annual depreciation rate for both KC and OC
OC_RATIO      = 0.30        # fraction of admin expense capitalized as OC
MIN_MC        = 1e4         # mc threshold (yuan) below which => NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _impute_report_date(row) -> pd.Timestamp:
    """Impute missing report_date from period-end date."""
    d = row["date"]
    if d.month == 3:
        return d + pd.DateOffset(months=1)
    elif d.month == 6:
        return d + pd.DateOffset(months=2)
    elif d.month == 9:
        return d + pd.DateOffset(months=1)
    else:
        return d + pd.DateOffset(months=4)


def _compute_capital_info_consistent(
    invest_vals: np.ndarray,
    quarter_idxs: np.ndarray,
    n_quarters: int,
    annual_decay: float,
) -> np.ndarray:
    """
    Compute accumulated (depreciated) capital for each quarter,
    using information-consistent ordering (sorted by report_date).

    At position i, only contributions from rows j <= i are included,
    preventing future-data leakage. Decay is based on actual quarter
    distance, correctly handling gaps and out-of-order filings.

    Args:
        invest_vals: 1-D array of quarterly investment values, sorted by
                     report_date (oldest first), already clipped to >= 0.
        quarter_idxs: 1-D int array of quarter indices (year*4 + q_num 0-3).
        n_quarters: maximum look-back window.
        annual_decay: annual depreciation rate (e.g. 0.90).

    Returns:
        capital_vals: 1-D array same length as invest_vals.
    """
    n_total = len(invest_vals)
    capital_vals = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        current_q = quarter_idxs[i]
        q_back = current_q - quarter_idxs[:i + 1]
        valid = (q_back >= 0) & (q_back < n_quarters)
        if valid.any():
            decay = annual_decay ** (q_back[valid] / 4.0)
            capital_vals[i] = float(np.dot(invest_vals[:i + 1][valid], decay))
    return capital_vals


class IBP_MRQ(FundamentalFactorCalculator):
    """
    Intangible-adjusted Book-to-Price (MRQ) factor.

    Adds accumulated (depreciated) R&D investment (knowledge capital, KC)
    and 30% of admin expenses (organizational capital, OC) to book equity,
    correcting for systematic under-statement of intangible assets.
    Then divides by daily market cap.

    Reference: Peters & Taylor (2017), Journal of Financial Economics.

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION}

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Calculate IBP_MRQ daily panel.

        Returns:
            FactorData: N stocks x T days, values = adj_equity_yi / mc (float64)
        """
        # ------------------------------------------------------------------
        # Step 1: Ensure raw data loaded; get trading dates
        # ------------------------------------------------------------------
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()

        if trading_dates.tz is not None:
            trading_dates = trading_dates.tz_localize(None)

        # ------------------------------------------------------------------
        # Step 2: Query extended R&D + admin + equity history from lixinger
        #         Go back 12 years to accumulate up to 48 quarters of KC/OC
        # ------------------------------------------------------------------
        rd_load_start = (
            fundamental_data.start_date - pd.DateOffset(years=12)
        ).date()
        load_end = fundamental_data.end_date.date()

        if fundamental_data._stock_codes:
            codes_str = "','".join(str(c) for c in fundamental_data._stock_codes)
            codes_filter = f"AND stock_code IN ('{codes_str}')"
        else:
            codes_filter = ""

        conn = sqlite3.connect(fundamental_data._lixinger_db)
        df_raw = pd.read_sql_query(
            f"""
            SELECT stock_code, date, report_date,
                   q_ps_rade_c, q_ps_ae_c, q_bs_tetoshopc_t
            FROM financial_statements
            WHERE date BETWEEN '{rd_load_start}' AND '{load_end}'
            {codes_filter}
            """,
            conn,
        )
        conn.close()

        if df_raw.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ------------------------------------------------------------------
        # Step 3: Date cleanup + report_date imputation
        # ------------------------------------------------------------------
        df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.tz_localize(None)
        df_raw["report_date"] = pd.to_datetime(
            df_raw["report_date"], errors="coerce"
        ).dt.tz_localize(None)

        null_mask = df_raw["report_date"].isna()
        if null_mask.any():
            df_raw.loc[null_mask, "report_date"] = df_raw.loc[null_mask].apply(
                _impute_report_date, axis=1
            )

        # Remove anomalous records where period_end > report_date
        df_raw = df_raw[df_raw["date"] <= df_raw["report_date"]].copy()

        # Deduplicate: keep last record per (stock_code, report_date)
        df_raw = df_raw.sort_values(["stock_code", "report_date", "date"])
        df_raw = df_raw.groupby(
            ["stock_code", "report_date"], as_index=False
        ).last()

        # Drop future report_dates
        df_raw = df_raw[df_raw["report_date"] <= fundamental_data.end_date]

        # Add symbol + mainboard filter
        df_raw["symbol"] = df_raw["stock_code"].apply(lixinger_code_to_symbol)
        mb_mask = np.array([_is_mainboard(s) for s in df_raw["symbol"]])
        df_raw = df_raw[mb_mask].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 4: Compute Knowledge Capital (KC) and Organizational Capital (OC)
        #         Sort by report_date for information-consistent accumulation
        # ------------------------------------------------------------------
        df_raw = df_raw.sort_values(
            ["symbol", "report_date", "date"]
        ).reset_index(drop=True)

        # Clean R&D: NaN -> 0, negative -> 0
        df_raw["rd_clean"] = df_raw["q_ps_rade_c"].fillna(0.0).clip(lower=0.0)

        # Clean admin expense: NaN -> 0, negative -> 0; then apply OC_RATIO
        df_raw["oc_invest"] = (
            df_raw["q_ps_ae_c"].fillna(0.0).clip(lower=0.0) * OC_RATIO
        )

        # Quarter index: year*4 + quarter_num(0-3), for decay computation
        dates_idx = pd.DatetimeIndex(df_raw["date"])
        df_raw["q_idx"] = (
            dates_idx.year * 4 + (dates_idx.month - 1) // 3
        ).astype(np.int32)

        kc_all = np.zeros(len(df_raw), dtype=np.float64)
        oc_all = np.zeros(len(df_raw), dtype=np.float64)
        idx_offset = 0
        for _sym, grp in df_raw.groupby("symbol", sort=False):
            q_idxs = grp["q_idx"].values
            n = len(q_idxs)

            kc_vals = _compute_capital_info_consistent(
                grp["rd_clean"].values, q_idxs, N_QUARTERS, ANNUAL_DECAY
            )
            oc_vals = _compute_capital_info_consistent(
                grp["oc_invest"].values, q_idxs, N_QUARTERS, ANNUAL_DECAY
            )
            kc_all[idx_offset: idx_offset + n] = kc_vals
            oc_all[idx_offset: idx_offset + n] = oc_vals
            idx_offset += n

        df_raw["kc"] = kc_all
        df_raw["oc"] = oc_all

        # ------------------------------------------------------------------
        # Step 5: adj_equity = tetoshopc + KC + OC (yuan)
        #         NaN tetoshopc => NaN adj_equity
        # ------------------------------------------------------------------
        tetoshopc = df_raw["q_bs_tetoshopc_t"].values.astype(np.float64)
        adj_equity = np.where(
            np.isnan(tetoshopc),
            np.nan,
            tetoshopc + df_raw["kc"].values + df_raw["oc"].values,
        )
        df_raw["adj_equity"] = adj_equity

        # ------------------------------------------------------------------
        # Step 6: Forward-fill adj_equity to daily trading dates
        #         Use report_date as the signal effective date
        # ------------------------------------------------------------------
        df_fe = df_raw[["symbol", "report_date", "adj_equity"]].dropna(
            subset=["adj_equity"]
        ).copy()

        pivot_ae = df_fe.pivot_table(
            index="symbol",
            columns="report_date",
            values="adj_equity",
            aggfunc="last",
        )

        all_dates_ae = pivot_ae.columns.union(trading_dates).sort_values()
        pivot_ae = pivot_ae.reindex(columns=all_dates_ae)
        panel_ae = pivot_ae.ffill(axis=1).reindex(columns=trading_dates)

        # ------------------------------------------------------------------
        # Step 7: Load daily market cap panel; apply mainboard filter
        # ------------------------------------------------------------------
        mc_vals, mc_syms, mc_dates = fundamental_data.get_market_cap_panel()

        mb_mc_mask = np.array([_is_mainboard(s) for s in mc_syms])
        mc_vals = mc_vals[mb_mc_mask]
        mc_syms = np.array(mc_syms)[mb_mc_mask]

        mc_dates_idx = pd.DatetimeIndex(mc_dates)
        if mc_dates_idx.tz is not None:
            mc_dates_idx = mc_dates_idx.tz_localize(None)

        df_mc = pd.DataFrame(mc_vals, index=mc_syms, columns=mc_dates_idx)

        # ------------------------------------------------------------------
        # Step 8: Align both panels to unified (symbols x trading_dates)
        # ------------------------------------------------------------------
        all_syms = sorted(
            set(panel_ae.index.tolist()) | set(mc_syms.tolist())
        )

        panel_ae = panel_ae.reindex(index=all_syms, columns=trading_dates)
        df_mc    = df_mc.reindex(index=all_syms, columns=trading_dates)

        # ------------------------------------------------------------------
        # Step 9: IBP_MRQ = adj_equity / mc  (both in yuan)
        # ------------------------------------------------------------------
        arr_ae = panel_ae.values.astype(np.float64)
        arr_mc = df_mc.values.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ibp = np.where(
                (arr_mc > MIN_MC) & ~np.isnan(arr_mc) & ~np.isnan(arr_ae),
                arr_ae / arr_mc,
                np.nan,
            )

        ibp = np.where(np.isinf(ibp), np.nan, ibp).astype(np.float64)

        nan_ratio = np.isnan(ibp).mean() if ibp.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"IBP_MRQ NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        symbols_arr = np.array(all_syms)
        dates_arr   = np.array(trading_dates, dtype='datetime64[ns]')

        return FactorData(
            values=ibp,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python ibp_mrq.py)
# Uses full market with short date range to ensure enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("IBP_MRQ factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute IBP_MRQ factor")
    calculator = IBP_MRQ()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"NaN ratio    : {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid    : {len(valid_cs)}")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")
        print(f"  median     : {np.median(valid_cs):.6f}")

    # Sanity check: known stocks
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]
    fd_test = FundamentalData(
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=TEST_CODES,
    )
    result_test = calculator.calculate(fd_test)
    print(f"\nSample values (5 test stocks, last 5 dates):")
    for code in TEST_CODES:
        sym = lixinger_code_to_symbol(code)
        sym_list = result_test.symbols.tolist()
        if sym in sym_list:
            idx = sym_list.index(sym)
            last5 = result_test.values[idx, -5:]
            print(f"  {sym}: {np.round(last5, 4)}")
        else:
            print(f"  {sym}: not found in result")

    print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")

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
