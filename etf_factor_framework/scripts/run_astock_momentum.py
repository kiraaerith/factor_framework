"""
A股动量因子测评脚本

用法（在项目根目录执行）：
    conda activate etf_cross_ml
    cd etf_factor_framework
    python scripts/run_astock_momentum.py

对应配置文件：
    config/astock_factors/momentum_astock_rf5.json

评估内容：
    - 动量因子 lookback = 20d / 60d / 120d（offset=1，跳过最近1日）
    - 数据源：DuckDB A股，2021-01-01 ~ 2024-12-31，后复权
    - 执行模式：T+1 开盘买入（execution_price=open）
    - 调仓频率：每5个交易日
    - 交易成本：佣金万3 + 卖出印花税万5 + 滑点千1
    - 停牌/涨跌停处理：价值冻结（freeze），涨停不买，跌停不卖
    - 新股过滤：上市满365日历天后才参与

结果保存至：
    ../factor_eval_result/astock/momentum（相对于项目根目录）
"""

import sys
import os

# 将 etf_factor_framework 目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.run_from_config import run_evaluation_from_config

CONFIG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "config", "astock_factors", "momentum_astock_rf5.json"
))

if __name__ == '__main__':
    run_evaluation_from_config(CONFIG_PATH)


