"""
run_factor_research.py
----------------------
Single-process research variant of run_factor_grid_v3.py.

Strips parallel/SharedMemory machinery, adds:
  - ST stock filtering (via tushare stock_st table)
  - Export of daily portfolio returns, positions, and factor values to CSV
  - Designed for one-off research runs with a single parameter set

Usage (run from etf_factor_framework directory):
    python scripts/run_factor_research.py --config config/factors/SIZE_research.yaml
    python scripts/run_factor_research.py --config config/factors/SIZE_research.yaml --output-dir /path/to/output

Config: same as run_factor_grid_v3.py, plus:
    backtest.exclude_st: true          # exclude ST stocks from factor ranking
    research.output_dir: "path"        # where to save CSV outputs (overridden by --output-dir)
"""

import os
import re
import sys
import time
import sqlite3
import warnings
import argparse
import importlib

warnings.filterwarnings("ignore")

# --- path setup: scripts/ -> etf_factor_framework/ ---
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT  = os.path.dirname(FRAMEWORK_DIR)
sys.path.insert(0, FRAMEWORK_DIR)

import numpy as np
import pandas as pd
import yaml

from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
from factors.fundamental.neutralization import apply_neutralization
from data.stock_data_loader import StockDataLoader
from mappers.position_mappers import RankBasedMapper
from evaluation.evaluator import FactorEvaluator
from core.factor_data import FactorData
from core.ohlcv_data import OHLCVData


def is_mainboard_symbol(symbol: str) -> bool:
    match = re.search(r'(\d{6})', symbol)
    if not match:
        return False
    code = match.group(1)
    return code.startswith('60') or code.startswith('00')


BENCHMARK_INDEX_MAP = {
    'csi300':  '000300.SH',
    'csi500':  '000905.SH',
    'csi2000': '932000.CSI',
}


def load_benchmark_returns(benchmarks, start_date, end_date, db_path):
    if not benchmarks:
        return {}
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        print(f"[WARN] Tushare DB not found at {db_path}, skipping benchmark loading.")
        return {}
    result = {}
    conn = sqlite3.connect(db_path)
    try:
        sd = start_date.replace('-', '')
        ed = end_date.replace('-', '')
        for bm_name in benchmarks:
            ts_code = BENCHMARK_INDEX_MAP.get(bm_name)
            if not ts_code:
                continue
            df = pd.read_sql_query(
                "SELECT trade_date, pct_chg FROM index_daily "
                "WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
                conn, params=(ts_code, sd, ed)
            )
            if df.empty:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date')
            result[bm_name] = df['pct_chg'] / 100.0
    finally:
        conn.close()
    return result


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_factor_class(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def mask_by_trade_context(raw_factor, trade_ctx, attr_name, label, log_prefix=""):
    """Mask factor values where trade_ctx.<attr_name> is True. Reusable for delisted/new_stock/ST."""
    mask_arr = getattr(trade_ctx, attr_name, None)
    if mask_arr is None or not mask_arr.any():
        return

    tc_sym_map  = {s: i for i, s in enumerate(trade_ctx.symbols)}   # Loop: ~5000次
    tc_date_map = {d: i for i, d in enumerate(trade_ctx.dates)}     # Loop: ~800次

    fac_si, tc_si = [], []
    for i, s in enumerate(raw_factor.symbols):  # Loop: ~6000次
        if s in tc_sym_map:
            fac_si.append(i)
            tc_si.append(tc_sym_map[s])

    fac_di, tc_di = [], []
    for j, d in enumerate(raw_factor.dates):  # Loop: ~800次
        if d in tc_date_map:
            fac_di.append(j)
            tc_di.append(tc_date_map[d])

    if not fac_si or not fac_di:
        return

    sub = mask_arr[np.ix_(tc_si, tc_di)]
    n_masked = int(sub.sum())
    if n_masked > 0:
        fac_si_arr  = np.array(fac_si)
        fac_di_arr  = np.array(fac_di)
        raw_factor._values[np.ix_(fac_si_arr, fac_di_arr)] = np.where(
            sub, np.nan, raw_factor._values[np.ix_(fac_si_arr, fac_di_arr)]
        )
        nan_ratio = np.isnan(raw_factor._values).sum() / raw_factor._values.size
        print(f"  {log_prefix}{label}: masked {n_masked} cells -> NaN ratio: {nan_ratio:.1%}")


def build_st_mask_from_tushare(symbols, dates, tushare_db_path, start_date, end_date):
    """Build (N, T) bool ST mask directly from tushare stock_st table.

    Returns ndarray of shape (len(symbols), len(dates)) with True = ST.
    """
    conn = sqlite3.connect(tushare_db_path)
    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')
    st_df = pd.read_sql_query(
        f"SELECT ts_code, trade_date FROM stock_st "
        f"WHERE trade_date >= '{sd}' AND trade_date <= '{ed}'",
        conn,
    )
    conn.close()

    if st_df.empty:
        return np.zeros((len(symbols), len(dates)), dtype=bool)

    # Convert ts_code -> framework symbol
    def ts_to_sym(tc):
        code, ex = tc.split('.')
        return f"SHSE.{code}" if ex == 'SH' else f"SZSE.{code}"

    st_df['symbol'] = st_df['ts_code'].apply(ts_to_sym)
    st_df['date'] = pd.to_datetime(st_df['trade_date'], format='%Y%m%d')

    sym_map  = {s: i for i, s in enumerate(symbols)}   # Loop: ~6000次
    date_map = {pd.Timestamp(d): j for j, d in enumerate(dates)}  # Loop: ~800次

    is_st = np.zeros((len(symbols), len(dates)), dtype=bool)
    si_list, di_list = [], []
    for _, row in st_df.iterrows():  # Loop: ~110000次 (dict O(1) each)
        si = sym_map.get(row['symbol'])
        di = date_map.get(row['date'])
        if si is not None and di is not None:
            si_list.append(si)
            di_list.append(di)
    if si_list:
        is_st[si_list, di_list] = True

    return is_st


def build_delist_transition_mask(symbols, dates, tushare_db_path):
    """Build (N, T) bool mask for stocks in delisting transition period (退市整理期).

    Uses namechange table: change_reason='终止上市' gives the exact start_date
    of the delisting transition period. Masks all dates from start_date onwards
    (the existing is_delisted mask handles dates >= delist_date, but this covers
    the gap between 退市整理期 start and delist_date where stock_st has no records).

    Returns ndarray of shape (len(symbols), len(dates)) with True = in transition.
    """
    conn = sqlite3.connect(tushare_db_path)
    nc_df = pd.read_sql_query(
        "SELECT ts_code, start_date FROM namechange WHERE change_reason='终止上市'",
        conn,
    )
    conn.close()

    if nc_df.empty:
        return np.zeros((len(symbols), len(dates)), dtype=bool)

    def ts_to_sym(tc):
        code, ex = tc.split('.')
        return f"SHSE.{code}" if ex == 'SH' else f"SZSE.{code}"

    nc_df['symbol'] = nc_df['ts_code'].apply(ts_to_sym)
    nc_df['start_ts'] = pd.to_datetime(nc_df['start_date'], format='%Y%m%d')

    sym_map = {s: i for i, s in enumerate(symbols)}    # Loop: ~6000次
    dates_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])

    mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    for _, row in nc_df.iterrows():  # Loop: ~142次
        si = sym_map.get(row['symbol'])
        if si is None:
            continue
        # Mask all dates >= transition start_date
        mask[si, :] = dates_ts >= row['start_ts']

    return mask


# ─────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────

def dates_to_str(dates_arr):
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates_arr]


def export_research_data(evaluator, factor_data, position_data, output_dir):
    """Save portfolio returns, positions, and factor values to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Daily portfolio returns
    if hasattr(evaluator, 'portfolio_returns') and evaluator.portfolio_returns is not None:
        ret_df = pd.DataFrame({
            'date': evaluator.portfolio_returns.index,
            'gross_return': evaluator.portfolio_returns.values,
        })
        if hasattr(evaluator, 'net_portfolio_returns') and evaluator.net_portfolio_returns is not None:
            ret_df['net_return'] = evaluator.net_portfolio_returns.values
        ret_df.to_csv(os.path.join(output_dir, "daily_returns.csv"), index=False)
        print(f"  Saved daily_returns.csv ({len(ret_df)} rows)")

    # 2. Positions matrix (sparse: only non-zero weights)
    w = position_data.weights
    syms = position_data.symbols
    dts  = dates_to_str(position_data.dates)
    # Save as sparse long format to avoid huge matrix CSV
    rows = []
    nonzero = np.nonzero(w)
    for idx in range(len(nonzero[0])):  # Loop: ~TOP_K * T次
        si, di = nonzero[0][idx], nonzero[1][idx]
        if not np.isnan(w[si, di]) and w[si, di] > 0:
            rows.append((dts[di], syms[si], w[si, di]))
    pos_df = pd.DataFrame(rows, columns=['date', 'symbol', 'weight'])
    pos_df.to_csv(os.path.join(output_dir, "positions.csv"), index=False)
    print(f"  Saved positions.csv ({len(pos_df)} rows)")

    # 3. Factor values (only for stocks that appear in positions)
    held_syms = set(pos_df['symbol'].unique()) if len(pos_df) > 0 else set()
    sym_idx = [i for i, s in enumerate(factor_data.symbols) if s in held_syms]  # Loop: ~6000次
    if sym_idx:
        fac_sub = factor_data.values[sym_idx, :]
        fac_syms = factor_data.symbols[sym_idx] if isinstance(factor_data.symbols, np.ndarray) else [factor_data.symbols[i] for i in sym_idx]
        fac_df = pd.DataFrame(fac_sub, index=fac_syms, columns=dates_to_str(factor_data.dates))
        fac_df.index.name = 'symbol'
        fac_df.to_csv(os.path.join(output_dir, "factor_values.csv"))
        print(f"  Saved factor_values.csv ({fac_df.shape})")

    # 4. Daily holding list (simple: date -> list of symbols)
    daily_holdings = pos_df.groupby('date')['symbol'].apply(list)
    # Save count per day
    count_df = pos_df.groupby('date').agg(
        n_stocks=('symbol', 'count'),
        total_weight=('weight', 'sum'),
    ).reset_index()
    count_df.to_csv(os.path.join(output_dir, "daily_holding_count.csv"), index=False)
    print(f"  Saved daily_holding_count.csv ({len(count_df)} rows)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single-process factor research evaluation")
    parser.add_argument("--config",     required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None,  help="Directory to save research CSV outputs")
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to china_stock_data directory. Defaults to <project_root>/../china_stock_data"
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg         = load_config(config_path)

    # --- resolve paths ---
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else \
               os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'china_stock_data'))
    tushare_db = os.path.join(data_dir, 'tushare.db')

    # --- parse config ---
    factor_class_path      = cfg["factor"]["class"]
    factor_init_params     = cfg["factor"].get("params", {})
    factor_direction       = cfg["factor"].get("direction", 1)
    start_date             = cfg.get("backtest", {}).get("start_date", "2016-01-01")
    end_date               = cfg.get("backtest", {}).get("end_date",   "2026-01-01")
    skip_leakage           = cfg.get("backtest", {}).get("skip_leakage_check", False)
    filter_mainboard_only  = cfg.get("backtest", {}).get("filter_mainboard_only", True)
    filter_new_stock_days  = cfg.get("backtest", {}).get("filter_new_stock_days", 365)
    exclude_st             = cfg.get("backtest", {}).get("exclude_st", False)
    execution_price        = cfg.get("backtest", {}).get("execution_price", "open")
    benchmarks             = cfg.get("benchmarks", ["csi300", "csi500", "csi2000"])

    # Research: single combo (no grid)
    research_cfg    = cfg.get("research", {})
    top_k           = research_cfg.get("top_k", 100)
    rebalance_freq  = research_cfg.get("rebalance_freq", 1)
    neutral_method  = research_cfg.get("neutralization", "raw")
    output_dir      = args.output_dir or research_cfg.get("output_dir", os.path.join(FRAMEWORK_DIR, "research_output"))

    factor_class = load_factor_class(factor_class_path)

    print("=" * 70)
    print(f"Factor Research Evaluation: {factor_class.__name__}")
    print(f"Config    : {config_path}")
    print(f"Period    : {start_date} ~ {end_date}")
    print(f"Params    : top_k={top_k}  freq={rebalance_freq}  neutral={neutral_method}  direction={factor_direction}")
    print(f"Filters   : mainboard={filter_mainboard_only}  new_stock={filter_new_stock_days}d  exclude_st={exclude_st}  exec_price={execution_price}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    t_start = time.time()

    # ----------------------------------------------------------
    # [1/6] Load market data
    # ----------------------------------------------------------
    print("\n[1/6] Loading market data...")
    loader = StockDataLoader(tushare_db_path=tushare_db)
    ohlcv  = loader.load_ohlcv(start_date, end_date, use_adjusted=True)
    raw_open_arr, raw_open_symbols, raw_open_dates = loader.load_raw_open(start_date, end_date)
    loader.close()

    if filter_mainboard_only:
        mb_mask = np.array([is_mainboard_symbol(s) for s in ohlcv.symbols])  # Loop: ~5000次
        mb_idx  = np.where(mb_mask)[0]
        n_before = len(ohlcv.symbols)
        ohlcv = OHLCVData(
            open=ohlcv.open[mb_idx],   high=ohlcv.high[mb_idx],
            low=ohlcv.low[mb_idx],     close=ohlcv.close[mb_idx],
            volume=ohlcv.volume[mb_idx], symbols=ohlcv.symbols[mb_idx],
            dates=ohlcv.dates,
        )
        mb_mask_raw      = np.array([is_mainboard_symbol(s) for s in raw_open_symbols])  # Loop: ~5000次
        raw_open_arr     = raw_open_arr[mb_mask_raw]
        raw_open_symbols = raw_open_symbols[mb_mask_raw]
        print(f"  Mainboard filter: {n_before} -> {len(ohlcv.symbols)} symbols")

    print(f"  OHLCV shape: {ohlcv.close.shape}")

    trade_ctx = loader.load_trade_context(
        start_date, end_date,
        raw_open_arr=raw_open_arr, symbols=raw_open_symbols,
        dates=raw_open_dates, new_stock_filter_days=filter_new_stock_days,
    )

    # Load benchmarks
    benchmark_returns = {}
    if benchmarks:
        print(f"  Loading benchmarks: {benchmarks}")
        benchmark_returns = load_benchmark_returns(benchmarks, start_date, end_date, db_path=tushare_db)
        for bm_name, bm_ret in benchmark_returns.items():
            print(f"    {bm_name}: {len(bm_ret)} days")

    # ----------------------------------------------------------
    # [2/6] Compute raw factor
    # ----------------------------------------------------------
    print("\n[2/6] Computing raw factor...")
    fd         = FundamentalData(start_date=start_date, end_date=end_date)
    calculator = factor_class(**factor_init_params)
    raw_factor = calculator.calculate(fd)
    print(f"  raw_factor shape: {raw_factor.shape}, NaN: {np.isnan(raw_factor.values).mean():.1%}")

    # ----------------------------------------------------------
    # [3/6] Apply filters to factor values
    # ----------------------------------------------------------
    print("\n[3/6] Applying filters...")

    # 3a. Mask untradeable (not in OHLCV)
    ohlcv_sym_set = set(ohlcv.symbols.tolist())
    untradeable = np.array([s not in ohlcv_sym_set for s in raw_factor.symbols])  # Loop: ~6000次
    n_untrade = int(untradeable.sum())
    if n_untrade > 0:
        raw_factor._values[untradeable, :] = np.nan
        print(f"  Untradeable: masked {n_untrade} symbols")

    # 3b. Mask delisted
    mask_by_trade_context(raw_factor, trade_ctx, 'is_delisted', 'Delisted')

    # 3c. Mask mainboard
    if filter_mainboard_only:
        non_mb = np.array([not is_mainboard_symbol(s) for s in raw_factor.symbols])  # Loop: ~6000次
        n_non_mb = int(non_mb.sum())
        if n_non_mb > 0:
            raw_factor._values[non_mb, :] = np.nan
            print(f"  Non-mainboard: masked {n_non_mb} symbols")

    # 3d. Mask new stocks
    mask_by_trade_context(raw_factor, trade_ctx, 'is_new_stock', f'New stock ({filter_new_stock_days}d)')

    # 3e. Mask ST stocks (NEW: research feature)
    if exclude_st:
        print("  Building ST mask from tushare...")
        is_st = build_st_mask_from_tushare(
            raw_factor.symbols, raw_factor.dates, tushare_db, start_date, end_date
        )
        n_st = int(is_st.sum())
        if n_st > 0:
            raw_factor._values[is_st] = np.nan
            nan_ratio = np.isnan(raw_factor._values).sum() / raw_factor._values.size
            print(f"  ST filter: masked {n_st} cells -> NaN ratio: {nan_ratio:.1%}")
        else:
            print("  ST filter: no ST cells found")

    # 3f. Mask stocks in delisting transition period (退市整理期)
    #     Uses namechange.change_reason='终止上市' for precise start_date
    print("  Building delisting transition mask from namechange...")
    is_delist_transition = build_delist_transition_mask(
        raw_factor.symbols, raw_factor.dates, tushare_db
    )
    n_dt = int(is_delist_transition.sum())
    if n_dt > 0:
        raw_factor._values[is_delist_transition] = np.nan
        nan_ratio = np.isnan(raw_factor._values).sum() / raw_factor._values.size
        print(f"  Delist transition: masked {n_dt} cells -> NaN ratio: {nan_ratio:.1%}")
    else:
        print("  Delist transition: no cells masked")

    print(f"  Final NaN ratio: {np.isnan(raw_factor.values).mean():.1%}")

    # ----------------------------------------------------------
    # [4/6] Leakage detection
    # ----------------------------------------------------------
    print("\n[4/6] Leakage detection...")
    if skip_leakage:
        print("  Skipped (skip_leakage_check=true).")
    else:
        leakage_found = False
        checked = 0
        for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
            detector = FundamentalLeakageDetector(split_ratio=sr)
            try:
                report = detector.detect(calculator, fd)
            except ValueError as e:
                if "panel is empty" in str(e):
                    continue
                raise
            report.print_report()
            checked += 1
            if report.has_leakage:
                leakage_found = True
        if leakage_found:
            raise RuntimeError("Leakage detected -- aborting.")
        print(f"  Passed ({checked} splits checked).")

    # ----------------------------------------------------------
    # [5/6] Neutralization + Position mapping + Evaluation
    # ----------------------------------------------------------
    print("\n[5/6] Neutralization & Evaluation...")
    industry_map          = fd.get_industry_map()
    mc_arr, mc_symbols, _ = fd.get_market_cap_panel()

    neutralized_arr = apply_neutralization(
        factor_arr=raw_factor.values, symbols=raw_factor.symbols,
        method=neutral_method, industry_map=industry_map,
        market_cap_arr=mc_arr, market_cap_symbols=mc_symbols,
    )
    factor_obj = FactorData(
        values=neutralized_arr, symbols=raw_factor.symbols,
        dates=raw_factor.dates, name=f"{raw_factor.name}_{neutral_method}",
        factor_type=raw_factor.factor_type, params=calculator.params,
    )

    mapper        = RankBasedMapper(top_k=top_k, direction=factor_direction)
    position_data = mapper.map_to_position(factor_obj)

    evaluator = FactorEvaluator(
        factor_data=factor_obj,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=rebalance_freq,
        execution_price=execution_price,
        trade_context=trade_ctx,
        delay=1,
        rebalance_freq=rebalance_freq,
        buy_commission_rate=0.0003,
        sell_commission_rate=0.0003,
        stamp_tax_rate=0.001,
        benchmark_returns=benchmark_returns if benchmark_returns else None,
    )
    results = evaluator.run_full_evaluation()

    # Print summary
    ic   = results.get("ic_metrics", {})
    radj = results.get("risk_adjusted_metrics", {})
    risk = results.get("risk_metrics", {})
    ret  = results.get("returns_metrics", {})
    bm   = results.get("benchmark_metrics", {})

    print(f"\n  IC:       mean={ic.get('ic_mean', float('nan')):.4f}  ir={ic.get('ic_ir', float('nan')):.3f}")
    print(f"  Rank IC:  mean={ic.get('rank_ic_mean', float('nan')):.4f}  ir={ic.get('rank_ic_ir', float('nan')):.3f}")
    print(f"  Sharpe:   {radj.get('sharpe_ratio', float('nan')):.3f}")
    print(f"  Annual Return: {ret.get('annualized_return', float('nan')):.2%}")
    print(f"  Cumul Return:  {ret.get('cumulative_return', float('nan')):.2%}")
    print(f"  Max DD:   {risk.get('max_drawdown', float('nan')):.2%}")
    for bm_name, bm_vals in bm.items():
        exc = bm_vals.get('excess_annual_return', float('nan'))
        print(f"  Excess vs {bm_name}: {exc:.2%}")

    # ----------------------------------------------------------
    # [6/6] Export research data
    # ----------------------------------------------------------
    print(f"\n[6/6] Exporting research data to {output_dir} ...")
    export_research_data(evaluator, factor_obj, position_data, output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done! Total time: {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
