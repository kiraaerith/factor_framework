# RSI因子配置

## 配置文件说明

| 文件 | period | 适用场景 | 特点 |
|------|--------|----------|------|
| rsi_period_5.json | 5 | 短线交易 | 信号频繁，噪声较大 |
| rsi_period_14.json | 14 | 标准周期 | 平衡信号质量与频率 |
| rsi_period_20.json | 20 | 中线交易 | 信号稳健，适合低频调仓 |

## 参数说明

- **period**: RSI计算周期
- **direction: -1**: RSI越低越好（超卖反弹策略）
- **top_k: 5**: 每天选5个ETF
- **weight_method: equal**: 等权重分配

## 使用建议

1. **短线交易**（日度调仓）：使用 period=5 或 period=10
2. **标准配置**（周度调仓）：使用 period=14
3. **长线配置**（月度调仓）：使用 period=20 或 period=30

## 回测执行

```bash
cd etf_factor_framework
python examples/run_from_config.py --config config/factors/rsi/rsi_period_14.json
```

## 结果存储位置

```
E:\code_project\factor_eval_result\etf\RSI_14\period=14\
```
