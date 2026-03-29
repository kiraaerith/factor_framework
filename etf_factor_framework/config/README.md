# ETF因子框架配置系统

## 快速开始

### 1. 查看所有配置

```bash
cd etf_factor_framework
python scripts/config_manager_cli.py list
```

### 2. 按因子类型筛选

```bash
python scripts/config_manager_cli.py list --factor-type RSI
```

### 3. 搜索配置

```bash
python scripts/config_manager_cli.py search rsi
```

### 4. 对比配置

```bash
python scripts/config_manager_cli.py compare \
    config/factors/rsi/rsi_period_5.json \
    config/factors/rsi/rsi_period_14.json
```

### 5. 运行配置

```bash
python examples/run_from_config.py --config config/factors/rsi/rsi_period_14.json
```

---

## 目录结构

```
config/
├── _base/                          # 基础模板
│   ├── base_data.json              # 数据配置模板
│   ├── base_evaluation.json        # 评估配置模板
│   └── base_storage.json           # 存储配置模板
│
├── _templates/                     # 配置模板
│   ├── single_factor_template.json
│   └── multi_factor_template.json
│
├── factors/                        # 按因子类型组织
│   ├── rsi/                        # RSI因子配置
│   │   ├── rsi_period_5.json
│   │   ├── rsi_period_14.json
│   │   ├── rsi_period_20.json
│   │   └── README.md               # 该目录说明文档
│   ├── momentum/                   # 动量因子
│   ├── macd/                       # MACD因子
│   ├── ma/                         # 均线因子
│   └── multi/                      # 多因子组合
│
├── strategies/                     # 按策略类型组织
│   ├── mean_reversion/             # 均值回归
│   ├── trend_following/            # 趋势跟踪
│   └── long_short/                 # 多空策略
│
├── mappers/                        # 映射器配置
│   ├── top_k_comparison/           # Top K对比
│   └── weight_methods/             # 权重方法对比
│
├── evaluations/                    # 评估参数配置
│   ├── short_term/                 # 短线参数
│   └── medium_term/                # 中线参数
│
├── experiments/                    # 实验配置
├── production/                     # 生产环境配置
└── index.json                      # 配置索引
```

---

## 配置层级

### 1. 基础模板 (`_base/`)

通用的基础配置，供其他配置引用：

- `base_data.json` - 数据源配置
- `base_evaluation.json` - 评估参数
- `base_storage.json` - 存储选项

### 2. 配置模板 (`_templates/`)

完整的配置模板：

- `single_factor_template.json` - 单因子模板
- `multi_factor_template.json` - 多因子模板

### 3. 具体配置 (`factors/`, `strategies/` 等)

实际使用的配置文件，按类型组织。

---

## 命名规范

```
{因子}_{参数}_{变体}.json

示例：
- rsi_period_14.json           # 基础命名
- rsi_period_14_top5.json      # 带映射器参数
- rsi_oversold_strategy.json   # 策略命名
```

---

## 配置管理工具

### 命令行工具

| 命令 | 用途 |
|------|------|
| `list` | 列出所有配置 |
| `list --factor-type RSI` | 按因子类型筛选 |
| `search rsi` | 搜索配置 |
| `compare path1 path2` | 对比配置 |
| `stats` | 显示统计信息 |
| `show path` | 显示配置详情 |
| `rebuild-index` | 重建索引 |

### Python API

```python
from config import ConfigManager

# 创建管理器
manager = ConfigManager("config")

# 列出配置
configs = manager.list_configs(factor_type="RSI")

# 搜索配置
results = manager.find_config(name_contains="period_14")

# 对比配置
df = manager.compare_configs([
    "factors/rsi/rsi_period_5.json",
    "factors/rsi/rsi_period_14.json"
])

# 创建新配置
new_path = manager.create_from_template(
    template_name="single_factor_template",
    new_name="rsi_period_30",
    output_dir="factors/rsi",
    factors_0_params_period=30
)

# 添加标签
manager.add_tag("factors/rsi/rsi_period_30.json", "rsi")
manager.add_tag("factors/rsi/rsi_period_30.json", "long_term")
```

---

## 配置文件示例

### 最简单的配置

```json
{
  "data": {"csv_path": "data.csv"},
  "factors": [{"name": "RSI", "type": "RSI", "params": {"period": 14}}],
  "mapper": {"type": "rank_based", "params": {"top_k": 5, "direction": -1}}
}
```

### 完整配置

```json
{
  "name": "RSI超卖反弹策略",
  "version": "1.0",
  "description": "RSI周期14，选超卖Top5",
  
  "data": {
    "csv_path": "../../../etf_rotation_daily.csv",
    "symbol_col": "symbol",
    "date_col": "eob"
  },
  
  "factors": [{
    "name": "RSI_14",
    "type": "RSI",
    "params": {"period": 14}
  }],
  
  "mapper": {
    "type": "rank_based",
    "params": {
      "top_k": 5,
      "direction": -1,
      "weight_method": "equal"
    }
  },
  
  "evaluation": {
    "forward_period": 5,
    "periods_per_year": 252,
    "risk_free_rate": 0.03,
    "commission_rate": 0.0002,
    "delay": 1
  },
  
  "storage": {
    "base_path": "E:/factor_results",
    "save_metrics": true,
    "save_plots": true
  }
}
```

---

## 配置存储结构

运行后会生成以下存储结构：

```
E:\code_project\factor_eval_result\etf\
├── RSI_14\                        # 因子名称
│   └── period=14\                 # 参数组合
│       ├── metrics.json           # 指标数据
│       ├── config.json            # 配置备份
│       ├── report.txt             # 文本报告
│       ├── cumulative_ic.png      # 累积IC曲线
│       ├── returns_drawdown.png   # 收益回撤曲线
│       └── ...
└── comparison_*.csv               # 多因子对比表
```

---

## 扩展阅读

- [详细配置说明](docs/config_system.md) - 完整的配置参数说明
- [快速参考卡](docs/config_cheatsheet.md) - 常用配置速查
- [配置管理方案](docs/config_management_guide.md) - 大规模配置管理设计

---

## 最佳实践

1. **使用基础模板**：新建配置时从 `_templates/` 复制模板
2. **分类存放**：按因子类型或策略类型放入对应目录
3. **命名清晰**：从文件名即可看出配置内容
4. **添加文档**：在目录中添加 README.md 说明配置用途
5. **定期归档**：不再使用的配置移入 `experiments/archive/`
