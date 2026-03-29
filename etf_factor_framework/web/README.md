# Factor Evaluation Dashboard

基本面因子测评的 Web 可视化工具。从 SQLite 数据库和因子状态 JSON 实时读取数据，提供因子索引和测评结果两个视图。

## 启动

```bash
cd etf_factor_framework/web
python app.py
# 浏览器访问 http://localhost:5001
```

需要 `etf_cross_ml` conda 环境（依赖 Flask）。

## 功能

页面包含两个 Tab：

### Tab 1: Factor Index（因子索引）

展示所有注册因子（含未测评的），数据来自 `factor_eval_status.json`。

- 列：Factor ID、名称、类别、方向、状态、Rank ICIR、Sharpe、公式、更新时间
- 状态标签颜色区分：completed（绿）、pending（灰）、data_missing / step_failed（红）
- 搜索框、类别筛选、状态筛选
- 每行提供 Spec / Code 按钮，弹窗查看因子规格文档和源代码

### Tab 2: Evaluation Results（测评结果）

展示已完成测评的因子汇总，每因子取 Rank IC IR 最高的组合作为代表行。

- 列：Factor、类别、评级、Rank ICIR、ICIR、Sharpe、年化收益、最大回撤、最优组合
- 评级规则：|Rank ICIR| >= 1.0 = A，0.5~1.0 = B，< 0.5 = C
- 点击列头排序，搜索框和类别/评级筛选

### 因子详情 + 收益曲线

点击测评结果表中的因子名称展开：

1. **Grid Eval 详情表**：该因子全部 12 个组合（2 中性化 x 2 top_k x 3 频率）的指标对比
2. **累计收益曲线**：ECharts 图表，支持缩放和 tooltip
3. **点击表格行切换曲线**：点击详情表任意一行，下方曲线自动切换到对应组合（双向同步下拉框）

### 文件查看

汇总表和索引表每行都有 Spec / Code 按钮，弹出 Modal 查看：
- **Spec**：因子规格文档（Markdown 源码）
- **Code**：因子 Python 源代码

支持 ESC 或点击背景关闭。

## 数据源

| 数据 | 路径 |
|------|------|
| 测评结果 DB | `{project_root}/factor_eval_result/factor_eval.db` |
| 因子状态 JSON | `{project_root}/agent_project/Fundamental_Factors/factor_eval_status.json` |
| 因子 Spec 文件 | `{project_root}/agent_project/Fundamental_Factors/factor_specs/` |
| 因子代码文件 | `{project_root}/etf_factor_framework/factors/fundamental/` |

数据库由 `run_factor_grid_v3.py` 网格评估写入，本工具只读访问。

## API

| 端点 | 说明 |
|------|------|
| `GET /` | 主页面 |
| `GET /api/factor_index` | 所有因子索引（状态、元数据） |
| `GET /api/summary` | 已测评因子汇总（每因子最优 combo） |
| `GET /api/details/<factor_id>` | 单因子全部 combo 指标 |
| `GET /api/curve/<factor_id>` | 单因子累计收益曲线数据 |
| `GET /api/file/<factor_id>/<type>` | 因子文件内容（type: spec / code） |
