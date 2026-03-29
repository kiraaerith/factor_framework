"""
run_roe_factor_parallel.py

ROE 因子全 A 股并行网格评估。
参数网格：3 neutralization × 4 top_k × 4 rebalance_freq = 48 组。

Workers: 2（固定，避免内存压力 / Manager 死锁）
DB writes: multiprocessing.Lock 保护（Strategy C）
API: 全 numpy（etf_factor_framework dataframe2numpy 版）
"""

import os
import sys
import time
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

# --- path setup ---
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)))
sys.path.insert(0, os.path.join(FRAMEWORK_DIR, "etf_factor_framework"))

import numpy as np

from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
from factors.fundamental.neutralization import apply_neutralization
from factors.fundamental.profitability.roe import ROE
from data.stock_data_loader import StockDataLoader
from mappers.position_mappers import RankBasedMapper
from evaluation.evaluator import FactorEvaluator
from storage.result_storage import ResultStorage, StorageConfig
from core.factor_data import FactorData

# ================================================================
# CONFIG
# ================================================================

FACTOR_CLASS       = ROE
FACTOR_DIRECTION   = 1
DATASET_NAME       = "全A股"
START_DATE         = "2013-01-01"
END_DATE           = "2025-12-31"
EVAL_DB_PATH       = r"E:\code_project_v2\factor_eval_result\factor_eval.db"

NEUTRALIZATION_METHODS = ['raw', 'industry', 'size']
TOP_K_LIST             = [10, 25, 50, 100]
REBALANCE_FREQS        = [5, 10, 20, 60]

N_WORKERS = 2

# ================================================================
# Global lock (set per worker process via initializer)
# ================================================================

_db_write_lock = None


def _worker_init(lock):
    global _db_write_lock
    _db_write_lock = lock
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"


# ================================================================
# Worker function (runs in subprocess)
# ================================================================

def run_single_config_worker(args):
    """Single task: one (neutral_method, top_k, freq) combination."""
    global _db_write_lock

    (factor_obj, ohlcv, trade_ctx,
     top_k, rebalance_freq, neutral_method,
     factor_name, factor_params) = args

    tag = f"{factor_name} | {neutral_method} | top_k={top_k} | freq={rebalance_freq}"

    try:
        mapper = RankBasedMapper(top_k=top_k, direction=FACTOR_DIRECTION)
        position_data = mapper.map_to_position(factor_obj)

        evaluator = FactorEvaluator(
            factor_data=factor_obj,
            ohlcv_data=ohlcv,
            position_data=position_data,
            forward_period=rebalance_freq,
            execution_price="open",
            trade_context=trade_ctx,
            delay=1,
            rebalance_freq=rebalance_freq,
            buy_commission_rate=0.0003,
            sell_commission_rate=0.0003,
            stamp_tax_rate=0.001,
        )
        results = evaluator.run_full_evaluation()

        with _db_write_lock:
            storage = ResultStorage(
                config=StorageConfig(storage_mode="database", db_path=EVAL_DB_PATH)
            )
            record_id = storage.save_evaluation_result(
                result=results,
                factor_name=factor_name,
                dataset_name=f"{DATASET_NAME}_{neutral_method}_top{top_k}_freq{rebalance_freq}",
                params=factor_params,
                dataset_params={
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "top_k": top_k,
                    "rebalance_freq": rebalance_freq,
                    "forward_period": rebalance_freq,
                    "neutralization_method": neutral_method,
                },
                neutralization_method=neutral_method,
                top_k=top_k,
                rebalance_freq=rebalance_freq,
                forward_period=rebalance_freq,
            )

        ic    = results.get("ic_metrics", {})
        radj  = results.get("risk_adjusted_metrics", {})
        risk  = results.get("risk_metrics", {})
        ic_ir     = ic.get("ic_ir", float("nan"))
        rank_icir = ic.get("rank_ic_ir", float("nan"))
        sharpe    = radj.get("sharpe_ratio", float("nan"))
        mdd       = risk.get("max_drawdown", float("nan"))

        return {
            "tag": tag,
            "record_id": record_id,
            "ic_ir": ic_ir,
            "rank_icir": rank_icir,
            "sharpe": sharpe,
            "mdd": mdd,
            "error": None,
        }

    except Exception as e:
        import traceback
        return {"tag": tag, "error": f"{e}\n{traceback.format_exc()}"}


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print(f"[Parallel] Factor grid evaluation: {FACTOR_CLASS.__name__}")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    total = len(NEUTRALIZATION_METHODS) * len(TOP_K_LIST) * len(REBALANCE_FREQS)
    print(f"Tasks: {len(NEUTRALIZATION_METHODS)} x {len(TOP_K_LIST)} x {len(REBALANCE_FREQS)} = {total}")
    print(f"Workers: {N_WORKERS}  (cpu_count={os.cpu_count()}, reserved={os.cpu_count() - N_WORKERS})")
    print("=" * 70)

    t_total_start = time.time()

    # --- [1/4] Load market data ---
    print("\n[1/4] Loading market data...")
    loader = StockDataLoader()
    ohlcv = loader.load_ohlcv(START_DATE, END_DATE, use_adjusted=True)

    # load_raw_open returns (ndarray, symbols, dates) in new numpy API
    raw_open_arr, raw_open_symbols, raw_open_dates = loader.load_raw_open(START_DATE, END_DATE)

    trade_ctx = loader.load_trade_context(
        START_DATE, END_DATE,
        raw_open_arr=raw_open_arr,
        symbols=raw_open_symbols,
        dates=raw_open_dates,
        new_stock_filter_days=365,
    )
    loader.close()
    print(f"  ohlcv shape: {ohlcv.close.shape}")

    # --- [2/4] Load fundamental data & compute raw factor ---
    print("\n[2/4] Loading fundamental data & computing raw factor...")
    fd = FundamentalData(start_date=START_DATE, end_date=END_DATE)
    calculator = FACTOR_CLASS()
    raw_factor = calculator.calculate(fd)
    factor_name = raw_factor.name
    factor_params = calculator.params
    print(f"  shape: {raw_factor.shape}")
    nan_ratio = np.isnan(raw_factor.values).sum() / raw_factor.values.size
    print(f"  NaN: {nan_ratio:.1%}")

    # --- [3/4] Leakage check ---
    print("\n[3/4] Leakage check...")
    detector = FundamentalLeakageDetector(split_ratio=0.7)
    leakage_report = detector.detect(calculator, fd)
    leakage_report.print_report()
    if leakage_report.has_leakage:
        raise RuntimeError("Leakage detected! Aborting.")
    print("  Passed.")

    # --- [4/4] Pre-compute neutralized factors & build task list ---
    print("\n[4/4] Computing neutralized factors & building task list...")
    industry_map = fd.get_industry_map()

    # get_market_cap_panel returns (ndarray, symbols, dates) in new numpy API
    mc_arr, mc_symbols, _mc_dates = fd.get_market_cap_panel()

    os.makedirs(os.path.dirname(EVAL_DB_PATH), exist_ok=True)

    tasks = []
    for neutral_method in NEUTRALIZATION_METHODS:
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
        for top_k in TOP_K_LIST:
            for freq in REBALANCE_FREQS:
                tasks.append((
                    factor_obj, ohlcv, trade_ctx,
                    top_k, freq, neutral_method,
                    factor_obj.name, factor_params,
                ))

    print(f"  {len(tasks)} tasks ready.")

    # ================================================================
    # Parallel execution
    # ================================================================
    print(f"\n[Running] {len(tasks)} tasks on {N_WORKERS} workers...\n")
    t_run_start = time.time()

    lock = multiprocessing.Manager().Lock()
    results_list = []

    with multiprocessing.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(lock,),
    ) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_config_worker, tasks), 1):
            if res["error"]:
                print(f"  [{i:2d}/{total}] ERROR: {res['tag']}")
                print(f"          {res['error'][:200]}")
            else:
                print(f"  [{i:2d}/{total}] {res['tag']}")
                print(f"          ic_ir={res['ic_ir']:.3f}  rank_icir={res['rank_icir']:.3f}"
                      f"  sharpe={res['sharpe']:.3f}  mdd={res['mdd']:.3f}  (id={res['record_id']})")
            results_list.append(res)

    t_run   = time.time() - t_run_start
    t_total = time.time() - t_total_start

    success = [r for r in results_list if not r["error"]]
    failed  = [r for r in results_list if r["error"]]

    print(f"\n{'='*70}")
    print(f"Done! {len(success)}/{total} succeeded, {len(failed)} failed")
    print(f"  Parallel run time: {t_run:.1f}s")
    print(f"  Total time (incl. data load): {t_total:.1f}s")
    if failed:
        print("\nFailed tasks:")
        for r in failed:
            print(f"  {r['tag']}: {r['error'][:300]}")
    print(f"\nResults stored in: {EVAL_DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
