"""
run_factor_grid_v2.py
---------------------
Generic fundamental factor grid evaluation driven by YAML config.

Grid: neutralization_methods x top_k_list x rebalance_freqs

Usage (run from etf_factor_framework directory):
    python scripts/run_factor_grid_v2.py --config config/factors/pb_grid.yaml

Config keys:
    factor.class          - full dotted path to factor class, e.g. factors.fundamental.valuation.pb.PB
    factor.direction      - 1 (higher is better) or -1 (lower is better)
    backtest.start_date   - e.g. "2016-01-01"
    backtest.end_date     - e.g. "2025-12-31"
    grid.neutralization_methods - list, e.g. [raw, industry, size]
    grid.top_k_list             - list, e.g. [10, 25, 50, 100]
    grid.rebalance_freqs        - list, e.g. [5, 10, 20, 60]
    storage.eval_db_path  - path to evaluation result SQLite DB
"""

import os
import sys
import time
import logging
import warnings
import argparse
import importlib
from datetime import datetime

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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_factor_class(class_path: str):
    """Dynamically import factor class from dotted string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def setup_logging(factor_class_name: str) -> logging.Logger:
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{factor_class_name.lower()}_grid_{timestamp}.log")
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


def main():
    parser = argparse.ArgumentParser(description="Fundamental factor grid evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)

    # --- parse config ---
    factor_class_path = cfg["factor"]["class"]
    factor_direction  = cfg["factor"]["direction"]
    start_date = cfg["backtest"]["start_date"]
    end_date   = cfg["backtest"]["end_date"]
    neutralization_methods = cfg["grid"]["neutralization_methods"]
    top_k_list             = cfg["grid"]["top_k_list"]
    rebalance_freqs        = cfg["grid"]["rebalance_freqs"]
    eval_db_path           = cfg["storage"]["eval_db_path"]

    factor_class = load_factor_class(factor_class_path)

    log = setup_logging(factor_class.__name__)

    total = len(neutralization_methods) * len(top_k_list) * len(rebalance_freqs)
    log.info("=" * 70)
    log.info(f"Factor grid evaluation: {factor_class.__name__}")
    log.info(f"Config: {config_path}")
    log.info(f"Period: {start_date} ~ {end_date}")
    log.info(f"Grid: {len(neutralization_methods)} x {len(top_k_list)} x {len(rebalance_freqs)} = {total} combos")
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

    # ----------------------------------------------------------
    # [2/5] Load fundamental data & compute raw factor
    # ----------------------------------------------------------
    log.info("[2/5] Loading fundamental data & computing raw factor...")
    fd = FundamentalData(start_date=start_date, end_date=end_date)
    calculator = factor_class()
    raw_factor = calculator.calculate(fd)
    factor_name   = raw_factor.name
    factor_params = calculator.params
    log.info(f"  raw_factor shape: {raw_factor.shape}")
    nan_ratio = np.isnan(raw_factor.values).sum() / raw_factor.values.size
    log.info(f"  NaN ratio: {nan_ratio:.1%}")

    # ----------------------------------------------------------
    # [3/5] Leakage detection (multi split_ratio)
    # ----------------------------------------------------------
    log.info("[3/5] Leakage detection (split_ratios: 0.4, 0.5, 0.6, 0.7, 0.8)...")
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        detector = FundamentalLeakageDetector(split_ratio=sr)
        leakage_report = detector.detect(calculator, fd)
        leakage_report.print_report()
        if leakage_report.has_leakage:
            leakage_found = True
    if leakage_found:
        raise RuntimeError("Leakage detected -- aborting.")
    log.info("  Passed (all split_ratios clean).")

    # ----------------------------------------------------------
    # [4/5] Pre-compute neutralized factors
    # ----------------------------------------------------------
    log.info("[4/5] Pre-computing neutralized factors...")
    industry_map = fd.get_industry_map()
    log.info(f"  industry_map: {len(industry_map)} stocks")
    mc_arr, mc_symbols, _mc_dates = fd.get_market_cap_panel()
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
    # [5/5] Parameter grid loop
    # ----------------------------------------------------------
    log.info(f"[5/5] Running {total} grid combinations...")
    count   = 0
    success = 0
    dataset_name_base = "A_shares"

    for neutral_method, factor_obj in neutralized_factors.items():
        log.info(f"--- Neutralization: {neutral_method} ---")
        for top_k in top_k_list:
            for freq in rebalance_freqs:
                count += 1
                tag = f"{factor_name} | {neutral_method} | top_k={top_k} | freq={freq}"
                log.info(f"  [{count:2d}/{total}] {tag}")
                try:
                    mapper = RankBasedMapper(top_k=top_k, direction=factor_direction)
                    position_data = mapper.map_to_position(factor_obj)

                    evaluator = FactorEvaluator(
                        factor_data=factor_obj,
                        ohlcv_data=ohlcv,
                        position_data=position_data,
                        forward_period=freq,
                        execution_price="open",
                        trade_context=trade_ctx,
                        delay=1,
                        rebalance_freq=freq,
                        buy_commission_rate=0.0003,
                        sell_commission_rate=0.0003,
                        stamp_tax_rate=0.001,
                    )
                    results = evaluator.run_full_evaluation()

                    record_id = storage.save_evaluation_result(
                        result=results,
                        factor_name=factor_obj.name,
                        dataset_name=f"{dataset_name_base}_{neutral_method}_top{top_k}_freq{freq}",
                        params=factor_params,
                        dataset_params={
                            "start_date": start_date,
                            "end_date": end_date,
                            "top_k": top_k,
                            "rebalance_freq": freq,
                            "forward_period": freq,
                            "neutralization_method": neutral_method,
                        },
                        neutralization_method=neutral_method,
                        top_k=top_k,
                        rebalance_freq=freq,
                        forward_period=freq,
                    )

                    ic   = results.get("ic_metrics", {})
                    radj = results.get("risk_adjusted_metrics", {})
                    risk = results.get("risk_metrics", {})
                    log.info(
                        f"         ic_ir={ic.get('ic_ir', float('nan')):.3f}"
                        f"  rank_icir={ic.get('rank_ic_ir', float('nan')):.3f}"
                        f"  sharpe={radj.get('sharpe_ratio', float('nan')):.3f}"
                        f"  mdd={risk.get('max_drawdown', float('nan')):.3f}"
                        f"  (id={record_id})"
                    )
                    success += 1

                except Exception as e:
                    import traceback
                    log.error(f"         ERROR: {e}")
                    log.error(traceback.format_exc()[:400])

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info(f"Done! {success}/{total} succeeded  |  total time: {elapsed:.1f}s")
    log.info(f"Results stored in: {eval_db_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
