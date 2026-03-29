"""
run_factor_grid_v3.py
---------------------
Parallel version of run_factor_grid_v2.py.

Uses ProcessPoolExecutor with initializer to share large data (ohlcv, trade_ctx,
neutralized factor panels) across workers without per-task serialization overhead.
DB writes happen in the main process only (no concurrent SQLite writes).

Usage (run from etf_factor_framework directory):
    python scripts/run_factor_grid_v3.py --config config/factors/pb_grid.yaml
    python scripts/run_factor_grid_v3.py --config config/factors/pb_grid.yaml --workers 4

Config keys: same as run_factor_grid_v2.py, plus optional:
    backtest.skip_leakage_check: true   # skip leakage check for daily valuation factors
"""

import os
import sys
import time
import logging
import warnings
import argparse
import importlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing.shared_memory import SharedMemory

warnings.filterwarnings("ignore")

# --- path setup: scripts/ -> etf_factor_framework/ ---
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT  = os.path.dirname(FRAMEWORK_DIR)
sys.path.insert(0, FRAMEWORK_DIR)

import numpy as np
import yaml

from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
from factors.fundamental.neutralization import apply_neutralization
from data.stock_data_loader import StockDataLoader
from mappers.position_mappers import RankBasedMapper
from evaluation.evaluator import FactorEvaluator
from storage.result_storage import ResultStorage, StorageConfig
from core.factor_data import FactorData


# ─────────────────────────────────────────────────────────────
# Worker-side globals — populated once by _worker_init()
# ─────────────────────────────────────────────────────────────
_ohlcv               = None
_trade_ctx           = None
_neutralized_factors = None   # dict: neutral_method -> FactorData
_benchmark_returns   = None   # dict: benchmark_name -> pd.Series (daily returns)


def _worker_init(shm_names: list, shm_sizes: list) -> None:
    """
    Called once per worker process at startup.
    Reads pickled data from SharedMemory blocks (avoids Windows pipe size limit).
    Each worker opens the shared memory by name, unpickles, then closes its view.
    """
    global _ohlcv, _trade_ctx, _neutralized_factors, _benchmark_returns
    warnings.filterwarnings("ignore")

    data_objects = []
    for name, size in zip(shm_names, shm_sizes):
        shm = SharedMemory(name=name, create=False)
        pkl_bytes = bytes(shm.buf[:size])
        shm.close()
        data_objects.append(pickle.loads(pkl_bytes))

    _ohlcv               = data_objects[0]
    _trade_ctx           = data_objects[1]
    _neutralized_factors = data_objects[2]
    _benchmark_returns   = data_objects[3] if len(data_objects) > 3 else {}
    bm_info = f"benchmarks={list(_benchmark_returns.keys())}" if _benchmark_returns else "no benchmarks"
    print(
        f"[Worker {os.getpid()}] ready: "
        f"ohlcv={_ohlcv.shape}  "
        f"methods={list(_neutralized_factors.keys())}  "
        f"{bm_info}",
        flush=True,
    )


def _eval_one_combo(args: dict) -> dict:
    """
    Evaluate a single (neutral_method, top_k, freq) combination.
    Runs in a worker process; reads shared data from process globals.
    Returns a plain dict so the main process can write to DB and emit log.
    """
    neutral_method    = args["neutral_method"]
    top_k             = args["top_k"]
    freq              = args["freq"]
    direction         = args["direction"]
    factor_params     = args["factor_params"]
    start_date        = args["start_date"]
    end_date          = args["end_date"]
    dataset_name_base = args["dataset_name_base"]
    combo_idx         = args["combo_idx"]
    total             = args["total"]

    factor_obj = _neutralized_factors[neutral_method]
    pid = os.getpid()
    tag = f"{factor_obj.name} | top_k={top_k} | freq={freq}"

    print(f"[Worker {pid}] [{combo_idx:2d}/{total}] START  {tag}", flush=True)
    t0 = time.time()

    try:
        mapper        = RankBasedMapper(top_k=top_k, direction=direction)
        position_data = mapper.map_to_position(factor_obj)
        evaluator     = FactorEvaluator(
            factor_data=factor_obj,
            ohlcv_data=_ohlcv,
            position_data=position_data,
            forward_period=freq,
            execution_price="open",
            trade_context=_trade_ctx,
            delay=1,
            rebalance_freq=freq,
            buy_commission_rate=0.0003,
            sell_commission_rate=0.0003,
            stamp_tax_rate=0.001,
            benchmark_returns=_benchmark_returns if _benchmark_returns else None,
        )
        results = evaluator.run_full_evaluation()
        elapsed = time.time() - t0

        ic   = results.get("ic_metrics", {})
        radj = results.get("risk_adjusted_metrics", {})
        risk = results.get("risk_metrics", {})
        print(
            f"[Worker {pid}] [{combo_idx:2d}/{total}] DONE   {tag}  "
            f"ic_ir={ic.get('ic_ir', float('nan')):.3f}  "
            f"rank_icir={ic.get('rank_ic_ir', float('nan')):.3f}  "
            f"sharpe={radj.get('sharpe_ratio', float('nan')):.3f}  "
            f"mdd={risk.get('max_drawdown', float('nan')):.3f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        return {
            "ok":             True,
            "combo_idx":      combo_idx,
            "tag":            tag,
            "results":        results,
            "factor_name":    factor_obj.name,
            "dataset_name":   f"{dataset_name_base}_{neutral_method}_top{top_k}_freq{freq}",
            "factor_params":  factor_params,
            "dataset_params": {
                "start_date":            start_date,
                "end_date":              end_date,
                "top_k":                 top_k,
                "rebalance_freq":        freq,
                "forward_period":        freq,
                "neutralization_method": neutral_method,
            },
            "neutral_method": neutral_method,
            "top_k":          top_k,
            "freq":           freq,
            "elapsed":        elapsed,
        }

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(
            f"[Worker {pid}] [{combo_idx:2d}/{total}] ERROR  {tag}: {exc}",
            flush=True,
        )
        return {
            "ok":        False,
            "combo_idx": combo_idx,
            "tag":       tag,
            "error":     str(exc),
            "traceback": traceback.format_exc()[:600],
            "elapsed":   elapsed,
        }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

import sqlite3
import pandas as pd


# Benchmark index code mapping
BENCHMARK_INDEX_MAP = {
    'csi300':  '000300.SH',
    'csi500':  '000905.SH',
    'csi2000': '932000.CSI',
}

TUSHARE_DB_PATH = os.path.join(PROJECT_ROOT, '..', 'china_stock_data', 'tushare.db')


def load_benchmark_returns(benchmarks: list, start_date: str, end_date: str) -> dict:
    """Load daily returns for benchmark indices from tushare.db.

    Args:
        benchmarks: list of benchmark names, e.g. ['csi300', 'csi500', 'csi2000']
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        dict: {benchmark_name: pd.Series} with DatetimeIndex and daily returns (decimal)
    """
    if not benchmarks:
        return {}

    db_path = os.path.abspath(TUSHARE_DB_PATH)
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
                print(f"[WARN] Unknown benchmark '{bm_name}', skipping.")
                continue
            df = pd.read_sql_query(
                "SELECT trade_date, pct_chg FROM index_daily "
                "WHERE ts_code=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
                conn, params=(ts_code, sd, ed)
            )
            if df.empty:
                print(f"[WARN] No data for benchmark '{bm_name}' ({ts_code}).")
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date')
            # pct_chg is in percentage, convert to decimal
            result[bm_name] = df['pct_chg'] / 100.0
    finally:
        conn.close()

    return result


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_factor_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def setup_logging(factor_class_name: str) -> logging.Logger:
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(
        log_dir, f"{factor_class_name.lower()}_grid_parallel_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log file: {log_file}")
    return log


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel fundamental factor grid evaluation"
    )
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default 4, capped at cpu_count-1)"
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg         = load_config(config_path)

    # --- parse config ---
    factor_class_path      = cfg["factor"]["class"]
    factor_direction       = cfg["factor"]["direction"]
    start_date             = cfg.get("backtest", {}).get("start_date", "2016-01-01")
    end_date               = cfg.get("backtest", {}).get("end_date",   "2026-01-01")
    skip_leakage           = cfg.get("backtest", {}).get("skip_leakage_check", False)
    neutralization_methods = cfg["grid"]["neutralization_methods"]
    top_k_list             = cfg["grid"]["top_k_list"]
    rebalance_freqs        = cfg["grid"]["rebalance_freqs"]
    eval_db_path           = cfg["storage"]["eval_db_path"]
    benchmarks             = cfg.get("benchmarks", ["csi300", "csi500", "csi2000"])

    n_workers    = min(args.workers, max(1, os.cpu_count() - 1))
    factor_class = load_factor_class(factor_class_path)
    log          = setup_logging(factor_class.__name__)

    total = len(neutralization_methods) * len(top_k_list) * len(rebalance_freqs)
    log.info("=" * 70)
    log.info(f"Factor grid evaluation (PARALLEL): {factor_class.__name__}")
    log.info(f"Config  : {config_path}")
    log.info(f"Period  : {start_date} ~ {end_date}")
    log.info(f"Grid    : {len(neutralization_methods)} neutralizations x "
             f"{len(top_k_list)} top_k x {len(rebalance_freqs)} freqs = {total} combos")
    log.info(f"Workers : {n_workers}  (cpu_count={os.cpu_count()})")
    log.info("=" * 70)

    t_start = time.time()

    # ----------------------------------------------------------
    # [1/5] Load market data
    # ----------------------------------------------------------
    log.info("[1/5] Loading market data...")
    loader = StockDataLoader()
    ohlcv = loader.load_ohlcv(start_date, end_date, use_adjusted=True)
    raw_open_arr, raw_open_symbols, raw_open_dates = loader.load_raw_open(start_date, end_date)
    trade_ctx = loader.load_trade_context(
        start_date,
        end_date,
        raw_open_arr=raw_open_arr,
        symbols=raw_open_symbols,
        dates=raw_open_dates,
        new_stock_filter_days=365,
    )
    loader.close()
    log.info(f"  ohlcv close shape: {ohlcv.close.shape}")

    # Load benchmark index returns
    benchmark_returns = {}
    if benchmarks:
        log.info(f"  Loading benchmark returns: {benchmarks}")
        benchmark_returns = load_benchmark_returns(benchmarks, start_date, end_date)
        for bm_name, bm_ret in benchmark_returns.items():
            log.info(f"    {bm_name}: {len(bm_ret)} days, range {bm_ret.index[0].date()}~{bm_ret.index[-1].date()}")
    else:
        log.info("  No benchmarks configured, skipping.")

    # ----------------------------------------------------------
    # [2/5] Load fundamental data & compute raw factor
    # ----------------------------------------------------------
    log.info("[2/5] Loading fundamental data & computing raw factor...")
    fd            = FundamentalData(start_date=start_date, end_date=end_date)
    calculator    = factor_class()
    raw_factor    = calculator.calculate(fd)
    factor_name   = raw_factor.name
    factor_params = calculator.params
    log.info(f"  raw_factor shape: {raw_factor.shape}")
    nan_ratio = np.isnan(raw_factor.values).sum() / raw_factor.values.size
    log.info(f"  NaN ratio: {nan_ratio:.1%}")

    # Mask out symbols not in OHLCV (untradeable / delisted stocks)
    # so that RankBasedMapper only ranks among tradeable stocks.
    ohlcv_sym_set = set(ohlcv.symbols.tolist())
    untradeable_mask = np.array([s not in ohlcv_sym_set for s in raw_factor.symbols])
    n_masked = int(untradeable_mask.sum())
    if n_masked > 0:
        raw_factor._values[untradeable_mask, :] = np.nan
        nan_ratio_after = np.isnan(raw_factor._values).sum() / raw_factor._values.size
        log.info(f"  Masked {n_masked} untradeable symbols -> NaN ratio: {nan_ratio_after:.1%}")

    # ----------------------------------------------------------
    # [3/5] Leakage detection
    # ----------------------------------------------------------
    log.info("[3/5] Leakage detection...")
    if skip_leakage:
        log.info("  Skipped (skip_leakage_check=true in config).")
    else:
        leakage_found = False
        checked_count = 0
        for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
            detector = FundamentalLeakageDetector(split_ratio=sr)
            try:
                leakage_report = detector.detect(calculator, fd)
            except ValueError as e:
                if "panel is empty" in str(e):
                    log.info(f"  split_ratio={sr}: panel empty for truncated period, skipping.")
                    continue
                raise
            leakage_report.print_report()
            checked_count += 1
            if leakage_report.has_leakage:
                leakage_found = True
        if leakage_found:
            raise RuntimeError("Leakage detected -- aborting.")
        if checked_count == 0:
            raise RuntimeError("All split_ratios produced empty panels -- cannot verify leakage.")
        log.info(f"  Passed ({checked_count} split_ratios checked, all clean).")

    # ----------------------------------------------------------
    # [4/5] Pre-compute neutralized factors
    # ----------------------------------------------------------
    log.info("[4/5] Pre-computing neutralized factors...")
    industry_map            = fd.get_industry_map()
    mc_arr, mc_symbols, _   = fd.get_market_cap_panel()
    log.info(f"  industry_map: {len(industry_map)} stocks")
    log.info(f"  market_cap shape: {mc_arr.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(eval_db_path)), exist_ok=True)
    storage = ResultStorage(
        config=StorageConfig(storage_mode="database", db_path=eval_db_path)
    )

    neutralized_factors = {}
    for neutral_method in neutralization_methods:
        neutralized_arr = apply_neutralization(
            factor_arr=raw_factor.values,
            symbols=raw_factor.symbols,
            method=neutral_method,
            industry_map=industry_map,
            market_cap_arr=mc_arr,
            market_cap_symbols=mc_symbols,
        )
        factor_obj = FactorData(
            values=neutralized_arr,
            symbols=raw_factor.symbols,
            dates=raw_factor.dates,
            name=f"{factor_name}_{neutral_method}",
            factor_type=raw_factor.factor_type,
            params=factor_params,
        )
        neutralized_factors[neutral_method] = factor_obj
    log.info(f"  Neutralized factors ready: {list(neutralized_factors.keys())}")

    # ----------------------------------------------------------
    # [5/5] Parallel grid evaluation
    # ----------------------------------------------------------
    log.info("[5/5] Serializing shared data into SharedMemory...")
    t_pkl         = time.time()
    ohlcv_pkl     = pickle.dumps(ohlcv)
    trade_ctx_pkl = pickle.dumps(trade_ctx)
    nf_pkl        = pickle.dumps(neutralized_factors)
    bm_pkl        = pickle.dumps(benchmark_returns)
    log.info(
        f"  Serialized in {time.time() - t_pkl:.1f}s  |  "
        f"ohlcv={len(ohlcv_pkl) // 1024 // 1024}MB  "
        f"trade_ctx={len(trade_ctx_pkl) // 1024 // 1024}MB  "
        f"factors={len(nf_pkl) // 1024 // 1024}MB  "
        f"benchmarks={len(bm_pkl) // 1024}KB"
    )

    # Write pickled bytes into SharedMemory blocks (avoids Windows pipe size limit)
    shm_blocks = []
    shm_names  = []
    shm_sizes  = []
    for label, pkl_data in [("ohlcv", ohlcv_pkl), ("trade_ctx", trade_ctx_pkl), ("factors", nf_pkl), ("benchmarks", bm_pkl)]:
        shm = SharedMemory(create=True, size=len(pkl_data))
        shm.buf[:len(pkl_data)] = pkl_data
        shm_blocks.append(shm)
        shm_names.append(shm.name)
        shm_sizes.append(len(pkl_data))
        log.info(f"  SharedMemory '{shm.name}' ({label}): {len(pkl_data) // 1024 // 1024}MB")
    # Free pickle bytes from main process memory
    del ohlcv_pkl, trade_ctx_pkl, nf_pkl, bm_pkl

    # Build task list — each item contains only small scalars + metadata
    dataset_name_base = "A_shares"
    combo_list = []
    idx = 0
    for neutral_method in neutralization_methods:
        for top_k in top_k_list:
            for freq in rebalance_freqs:
                idx += 1
                combo_list.append({
                    "neutral_method":    neutral_method,
                    "top_k":             top_k,
                    "freq":              freq,
                    "direction":         factor_direction,
                    "factor_params":     factor_params,
                    "start_date":        start_date,
                    "end_date":          end_date,
                    "dataset_name_base": dataset_name_base,
                    "combo_idx":         idx,
                    "total":             total,
                })

    log.info(f"[5/5] Launching {n_workers} workers for {total} combos...")

    success = 0
    failed  = 0

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(shm_names, shm_sizes),
        ) as executor:
            futures = {executor.submit(_eval_one_combo, combo): combo for combo in combo_list}

            for future in as_completed(futures):
                res = future.result()

                if res["ok"]:
                    record_id = storage.save_evaluation_result(
                        result=res["results"],
                        factor_name=res["factor_name"],
                        dataset_name=res["dataset_name"],
                        params=res["factor_params"],
                        dataset_params=res["dataset_params"],
                        neutralization_method=res["neutral_method"],
                        top_k=res["top_k"],
                        rebalance_freq=res["freq"],
                        forward_period=res["freq"],
                    )
                    ic   = res["results"].get("ic_metrics", {})
                    radj = res["results"].get("risk_adjusted_metrics", {})
                    risk = res["results"].get("risk_metrics", {})
                    bm   = res["results"].get("benchmark_metrics", {})
                    bm_str = ""
                    for bm_name, bm_vals in bm.items():
                        exc = bm_vals.get('excess_annual_return', float('nan'))
                        bm_str += f"  exc_{bm_name}={exc:.3f}"
                    log.info(
                        f"  [SAVED {res['combo_idx']:2d}/{total}] {res['tag']}"
                        f"  ic_ir={ic.get('ic_ir', float('nan')):.3f}"
                        f"  rank_icir={ic.get('rank_ic_ir', float('nan')):.3f}"
                        f"  sharpe={radj.get('sharpe_ratio', float('nan')):.3f}"
                        f"  mdd={risk.get('max_drawdown', float('nan')):.3f}"
                        f"{bm_str}"
                        f"  ({res['elapsed']:.1f}s)  (id={record_id})"
                    )
                    success += 1
                else:
                    log.error(
                        f"  [FAIL  {res['combo_idx']:2d}/{total}] {res['tag']}: "
                        f"{res.get('error')}"
                    )
                    log.error(res.get("traceback", "")[:400])
                    failed += 1
    finally:
        # Clean up SharedMemory blocks
        for shm in shm_blocks:
            shm.close()
            shm.unlink()

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info(
        f"Done!  {success}/{total} succeeded  {failed} failed  |  "
        f"wall time: {elapsed:.1f}s  |  workers: {n_workers}"
    )
    log.info(f"Results stored in: {eval_db_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
