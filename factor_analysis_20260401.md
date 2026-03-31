# 因子开发方向分析报告
**生成时间：** 2026-04-01
**目的：** 基于现有因子库和数据库可用字段，分析还有哪些可做的因子方向

---

## 一、现状总结

### 1.1 已实现因子概览

| 大类 | 子类 | 已实现数量 | 代表因子 |
|------|------|:----------:|---------|
| 技术因子 | OHLCV指标 | 6 | CloseOverMA, RSI, MACD, BollingerBands, Momentum |
| CTC量价因子 | 成交量拆分 | 20+ | HighVolReturnSum, PVCorr, VolAmplitudeImbalance |
| 动量因子 | 多周期动量 | 1类 | MomentumFactor (offset × lookback 参数化) |
| 成长因子 | 基本面增长 | 44 | ROE/ROA/ROIC/GPM/NI/REV/OCF_GROWTH_COMP, NI_FG, NI_ST_FT 等 |
| 盈利因子 | 盈利能力水平 | 4 | ROE, ROA, ROIC, GPM |
| 价值因子 | 估值倒数/相对估值 | 27 | EP_TTM, BP_MRQ, SP_TTM, DP_TTM, CFP_TTM, EbitdaToEv, EbitToEv, RIMVP 等 |
| 质量因子 | 盈余质量/杠杆 | 6 | PiotroskiFScore, AccrualRatio, EarningsQuality, GoodwillRisk, InterestCoverage, NetDebtToEquity |
| 效率因子 | 周转率 | 4 | AR_TURNOVER, INVENTORY_TURNOVER, AR_GROWTH_VS_REV, AssetEfficiencyComp |
| 现金流因子 | 应计 | 1 | cashflow_ACC |
| 技术扩展 | 回撤 | 1 | MaxDrawdownFactor |
| ML因子 | 机器学习 | 1 | MLCrossSectionalFactor (LightGBM) |

**合计：约 115+ 个因子类**

### 1.2 上次分析（20260329）提出的因子实现进度

| 类别 | 提出 | 已实现 | 完成率 | 未完成因子 |
|------|:----:|:------:|:------:|-----------|
| A 质量 | 5 | 2 | 40% | NON_RECURRING_RATIO, OCF_OP_RATIO, CASH_EARNINGS_COMP |
| B 杠杆 | 6 | 3 | 50% | DEBT_RATIO, CURRENT_RATIO, FINANCIAL_LEVERAGE_GROWTH |
| C 效率 | 5 | 4 | 80% | INVENTORY_GROWTH_VS_REV |
| D 估值扩展 | 8 | 2 | 25% | PEG, BP_WO_GW, PE/PB_HIST_PERCENTILE, EBITDA_MARGIN, NET_PROFIT_MARGIN |
| E 成长质量 | 5 | 1 | 20% | GROWTH_QUALITY_COMP, PROFITABLE_GROWTH, QUALITY_MOMENTUM, FCF_YIELD_GROWTH |
| F 流动性 | 3 | 0 | 0% | 全部未实现 |
| G 技术扩展 | 6 | 1 | 17% | ATR, PRICE_REVERSAL_SHORT, VOLUME_TREND, HIGH_LOW_RATIO, CONSECUTIVE_GAINS |
| **合计** | **38** | **13** | **34%** | 25个待实现 |

---

## 二、数据库字段利用率分析

### 2.1 lixinger.fundamental（日频估值，48字段）

| 状态 | 字段数 | 示例 |
|------|:------:|------|
| ✅ 已用于因子 | 6 | pe_ttm, pb, ps_ttm, dyr, pcf_ttm, mc |
| ⚠️ 已有字段但未专门出因子 | 5 | ev_ebit_r, ev_ebitda_r (已用于EbitToEv/EbitdaToEv) |
| ❌ 完全未使用（高质量） | 12 | pb_wo_gw, ey, d_pe_ttm, to_r, cmc, ecmc, shn, spa, pe/pb/ps/dyr 历史分位共16个 |
| ❌ 完全未使用（中等质量） | 8 | 融资融券系列 (47.8% non-null) |
| ⛔ 不可用 | 1 | peg (0% non-null，无数据) |

**字段利用率：约 23%**

### 2.2 lixinger.financial_statements（季度财务，306字段）

| 状态 | 字段数 | 示例 |
|------|:------:|------|
| ✅ 已用于因子 | ~35 | ROE/ROA/ROIC水平、增长、营收/利润/现金流 |
| ❌ 利润表未用 | ~15 | q_ps_fe_c (财务费用), q_ps_ebitda_t, q_ps_np_s_r_t, q_ps_op_s_r_t, q_ps_beps_c, q_ps_npadnrpatoshaopc_c, q_ps_cp_c |
| ❌ 资产负债表未用 | ~15 | q_bs_cabb_t (货币资金), q_bs_ar_t, q_bs_i_t, q_bs_gw_t, q_bs_ap_t, q_bs_ca_t, q_bs_cl_t, q_bs_nwc_t, q_bs_lwi_t, YoY增长 |
| ❌ 现金流未用 | ~5 | q_cfs_ncffia_c (投资CF), q_cfs_ncfffa_c (筹资CF), q_cfs_niicace_c |
| ❌ 派生指标未用 | ~10 | q_m_c_r_t, q_m_q_r_t, q_m_tl_ta_r_t, q_m_ar_tor_t (部分已间接用), q_m_i_tor_t, q_m_ap_tor_t, q_m_ncffoa_op_r_t, q_m_fcf_c |

**字段利用率：约 12%**

### 2.3 tushare.report_rc（分析师预测，21字段）

| 状态 | 字段数 | 示例 |
|------|:------:|------|
| ✅ 已用于因子 | 3 | np (净利润预测), quarter, ts_code → NI_FG, NI_ST_FT, NI_LT_FT 等 |
| ❌ 高质量未用 | 3 | eps (99%), rating (100%), op_rt (76%) |
| ❌ 中等质量未用 | 3 | roe (69%), pe (61%), ev_ebitda (49%) |
| ⛔ 低质量 | 3 | op_pr (8.7%), max_price (0.9%), rd (20.6%) |

**字段利用率：约 14%**

### 2.4 juejin 行情数据

| 表 | 状态 |
|----|------|
| ohlcv / ohlcv_adjusted | ✅ 已充分利用 |
| trade_status | ⚠️ 仅用于过滤（is_suspended），turn_rate/is_st 可做因子 |
| daily_basic | ⚠️ 仅用于过滤，股本结构数据未用于因子 |

---

## 三、新因子方向分析

### 方向1：估值历史分位因子（数据现成，即插即用）

**动机：** lixinger.fundamental 提供了PE/PB/PS/DYR的1/3/5/10年历史分位数，完全现成，无需计算，是"相对自身历史便宜"的直接衡量。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `PE_HIST_PCTL` | 1 - pe_ttm_y5_cvpos | 99% | +1 | ★☆☆ |
| `PB_HIST_PCTL` | 1 - pb_y5_cvpos | 99% | +1 | ★☆☆ |
| `PS_HIST_PCTL` | 1 - ps_ttm_y5_cvpos | 99% | +1 | ★☆☆ |
| `DYR_HIST_PCTL` | dyr_y5_cvpos | 99% | +1 | ★☆☆ |
| `VALUATION_HIST_COMP` | PE+PB+PS分位综合 | 99% | +1 | ★★☆ |
| `MULTI_HORIZON_PE_PCTL` | pe_ttm_y{1,3,5,10}_cvpos 趋势/差异 | 97-99% | ±1 | ★★☆ |

**优先级：高。** 直接可用，无需财务数据对齐，A股均值回归效应显著。

---

### 方向2：剔除商誉P/B（BP_WO_GW）

**动机：** A股商誉减值频发，pb_wo_gw 字段直接可用（99.4% non-null），比普通PB更"干净"。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `BP_WO_GW` | 1 / pb_wo_gw | 99.4% | +1 | ★☆☆ |
| `GW_VALUATION_GAP` | pb_wo_gw - pb 差异 | 99.4% | -1 | ★☆☆ |

**优先级：高。** 数据现成，一行代码即可实现。

---

### 方向3：换手率/流动性因子（全新类别）

**动机：** 流动性因子在A股有显著的反转效应（高换手→短期反转），且数据质量极好。目前完全空白。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `TURNOVER_REVERSAL` | to_r (fundamental) 短期均值 | 99.4% | -1 | ★☆☆ |
| `TURNOVER_MOMENTUM` | to_r 长短均值差 | 99.4% | +1 | ★★☆ |
| `ABNORMAL_TURNOVER` | to_r / MA(to_r, 60) - 1 | 99.4% | -1 | ★★☆ |
| `TURNOVER_VOLATILITY` | to_r 滚动标准差 | 99.4% | -1 | ★★☆ |
| `SIZE_ADJ_TURNOVER` | to_r / cmc | 99.4% | -1 | ★★☆ |
| `ILLIQUIDITY` | Amihud illiquidity (|ret| / volume) | OHLCV | +1 | ★★☆ |

**优先级：高。** A股散户占比高，换手率是强反转信号，文献支持充分。

---

### 方向4：分析师预期因子扩展

**动机：** 目前仅用了tushare的净利润预测（np字段），rating（100%）、eps（99%）、op_rt（76%）等高质量字段完全未用。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `CONSENSUS_RATING` | rating 评级编码截面排名 | 100% | +1 | ★★☆ |
| `RATING_CHANGE` | rating 评级调升/调降 | 100% | +1 | ★★☆ |
| `EPS_REVISION` | eps 一致预期修正动量 | 99% | +1 | ★★☆ |
| `EPS_FG` | (FY1_eps - actual_eps) / actual_eps | 99% | +1 | ★★★ |
| `REV_FG` | (FY1_rev - actual_rev) / actual_rev | 76% | +1 | ★★★ |
| `ANALYST_COVERAGE` | 覆盖分析师家数 / org_name去重 | 100% | +1 | ★★☆ |
| `ANALYST_DIVERGENCE` | np预测的标准差/均值（分歧度） | 97% | -1 | ★★★ |
| `ROE_FWD` | roe 前向ROE中位数 | 69% | +1 | ★★☆ |
| `FWD_EV_EBITDA` | ev_ebitda 前向EV/EBITDA倒数 | 49% | +1 | ★★★ |

**优先级：高。** 分析师预期修正是全球alpha工厂的核心信号之一，当前利用率极低。

---

### 方向5：利润率水平与趋势因子

**动机：** 现有盈利因子只有 ROE/ROA/ROIC/GPM 的水平值和增长综合，但净利率、营业利润率、EBITDA利润率等水平和趋势因子缺失。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `NET_PROFIT_MARGIN` | q_ps_np_s_r_t | 62% | +1 | ★☆☆ |
| `OP_MARGIN` | q_ps_op_s_r_t | 62% | +1 | ★☆☆ |
| `NPM_TREND` | q_ps_np_s_r_t 多季度趋势 | 62% | +1 | ★★☆ |
| `MARGIN_STABILITY` | 利润率波动率（取倒数） | 62% | +1 | ★★☆ |
| `EBITDA_MARGIN` | q_ps_ebitda_t / q_ps_toi_c | 27%/60% | +1 | ★★☆ |
| `MARGIN_EXPANSION` | 毛利率-净利率差的变化 | 61%/62% | -1 | ★★★ |

**优先级：中高。** 利润率趋势是DuPont分解的核心，补充后盈利因子体系更完整。

---

### 方向6：杠杆/偿债安全因子（补完上次提案）

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `DEBT_RATIO` | q_m_tl_ta_r_t | 60% | -1 | ★☆☆ |
| `CURRENT_RATIO` | q_m_c_r_t | 59% | +1 | ★☆☆ |
| `QUICK_RATIO` | q_m_q_r_t | 59% | +1 | ★☆☆ |
| `IB_DEBT_RATIO` | q_bs_lwi_t / q_bs_ta_t（有息负债率） | 92%/94% | -1 | ★★☆ |
| `NET_CASH_RATIO` | (q_bs_cabb_t - q_bs_lwi_t) / mc | 99%/92% | +1 | ★★☆ |
| `LEVERAGE_DELTA` | q_bs_tl_t_y2y - q_bs_ta_t_y2y | 55% | -1 | ★★☆ |
| `CASH_BUFFER` | q_bs_cabb_t / q_bs_cl_t | 99%/100% | +1 | ★★☆ |

**优先级：中高。** 杠杆因子在市场下行期收益显著，且数据质量好。

---

### 方向7：营运资本与供应链效率

**动机：** 应付账款周转率完全未用，NWC/营收比率未用。交叉对比应收-存货-应付可以识别供应链话语权。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `AP_TURNOVER` | q_m_ap_tor_t | 60% | 视行业 | ★☆☆ |
| `CASH_CONVERSION_CYCLE` | 1/AR_TOR + 1/INV_TOR - 1/AP_TOR | 59-60% | -1 | ★★☆ |
| `NWC_TO_REV` | q_bs_nwc_t / q_ps_toi_c | 98%/60% | -1 | ★★☆ |
| `NWC_TREND` | NWC/营收的变化趋势 | 同上 | -1 | ★★★ |
| `INVENTORY_GROWTH_VS_REV` | 存货增速 - 营收增速 | q_bs_i_t, q_ps_toi_c | -1 | ★★☆ |
| `AP_GROWTH_VS_COST` | 应付增速 - 成本增速 | q_bs_ap_t, q_ps_toc_c | +1 | ★★☆ |
| `SUPPLY_CHAIN_POWER` | AP周转/AR周转（占用上游资金能力） | 60% | +1 | ★★☆ |

**优先级：中。** 周期行业效率信号强，但需行业中性化处理。

---

### 方向8：现金流结构因子

**动机：** 经营/投资/筹资三活动现金流的结构信号完全未用。现金流结构可以识别企业生命周期阶段。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `OCF_OP_RATIO` | q_m_ncffoa_op_r_t | 62% | +1 | ★☆☆ |
| `CAPEX_INTENSITY` | q_cfs_cpfpfiaolta_c / q_ps_toi_c | 61%/60% | 视情 | ★★☆ |
| `FINANCING_DEPENDENCE` | q_cfs_ncfffa_c / q_cfs_ncffoa_c | 61% | -1 | ★★☆ |
| `CASH_LIFECYCLE` | OCF、ICF、FCF三符号组合编码 | 61% | 分类 | ★★★ |
| `FCF_STABILITY` | q_m_fcf_c 波动率（取倒数） | 60% | +1 | ★★☆ |
| `FCF_YIELD` | q_m_fcf_ttm / mc | 61%/99% | +1 | ★★☆ |
| `REINVESTMENT_RATE` | capex / OCF | 61% | 视情 | ★★☆ |
| `NON_RECURRING_RATIO` | q_ps_npadnrpatoshaopc_c / q_ps_npatoshopc_c | 59% | -1 | ★★☆ |

**优先级：中。** 自由现金流和现金流结构在A股价值投资体系中日益重要。

---

### 方向9：股东结构/筹码因子

**动机：** `shn`（股东人数）和 `ecmc_psh`（人均自由流通市值）在 fundamental 表中非空率 97-98%，是天然的筹码集中度指标。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `HOLDER_CONCENTRATION` | ecmc / shn（人均持股市值） | 97-99% | +1 | ★☆☆ |
| `HOLDER_COUNT_CHANGE` | shn 环比变化 | 98% | -1 | ★★☆ |
| `FLOAT_RATIO` | cmc / mc（流通率） | 99% | 视情 | ★☆☆ |
| `FREE_FLOAT_RATIO` | ecmc / mc（自由流通率） | 99% | 视情 | ★☆☆ |

**优先级：中。** A股特有的筹码因子，在中小盘效果尤其好。

---

### 方向10：融资融券因子

**动机：** lixinger.fundamental 有融资融券余额和净买入/卖出数据，非空率约47.8%（仅覆盖两融标的，约2000+只）。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `MARGIN_BALANCE_RATIO` | fb / mc（融资余额占比） | 48% | 视情 | ★★☆ |
| `NET_MARGIN_PURCHASE` | fnpa / mc（融资净买入占比） | 48% | +1 | ★★☆ |
| `SHORT_INTEREST_RATIO` | sb / mc（融券余额占比） | 48% | -1 | ★★☆ |
| `MARGIN_MOMENTUM` | fnpa 滚动均值趋势 | 48% | +1 | ★★★ |
| `MARGIN_LONG_SHORT` | fb / (fb + sb)（多空比） | 48% | +1 | ★★★ |

**优先级：中低。** 数据覆盖率有限（约一半），但在覆盖范围内信号质量较高。

---

### 方向11：技术因子扩展

| 因子 | 数据源 | 方向 | 难度 |
|------|--------|:----:|:----:|
| `ATR` | OHLCV | -1 (波动率反转) | ★☆☆ |
| `PRICE_REVERSAL_1W` | close 5日收益 | -1 | ★☆☆ |
| `PRICE_REVERSAL_1M` | close 20日收益 | -1 | ★☆☆ |
| `HIGH_LOW_RATIO` | high/low 滚动比 | -1 | ★☆☆ |
| `VOLUME_TREND` | OBV类 | +1 | ★★☆ |
| `CONSECUTIVE_GAINS` | 连涨天数 | -1 (反转) | ★★☆ |
| `INTRADAY_MOMENTUM` | (close-open)/(high-low) | +1 | ★★☆ |
| `OVERNIGHT_RETURN` | open/prev_close - 1 | +1 | ★★☆ |
| `SMART_MONEY_FLOW` | 大单资金流向代理 | +1 | ★★★ |
| `REALIZED_SKEWNESS` | 收益率偏度 | -1 | ★★☆ |
| `REALIZED_KURTOSIS` | 收益率峰度 | -1 | ★★☆ |
| `DOWNSIDE_BETA` | 下行Beta | -1 | ★★★ |

**优先级：中。** A股短期反转强，技术类因子单独效果有限但和基本面复合效果好。

---

### 方向12：成长质量复合因子

**动机：** 单维度增长因子容易被"注水"误导，结合质量约束可提高预测力。

| 因子 | 组合维度 | 方向 | 难度 |
|------|---------|:----:|:----:|
| `GROWTH_QUALITY_COMP` | ROE_GROWTH_COMP × EarningsQuality 加权 | +1 | ★★★ |
| `PROFITABLE_GROWTH` | ROE水平 + REV_GROWTH_COMP 双筛选 | +1 | ★★★ |
| `QUALITY_MOMENTUM` | (ROE + ROIC) × Momentum | +1 | ★★★ |
| `FCF_YIELD_GROWTH` | FCF_MC趋势 + FCF水平 | +1 | ★★★ |
| `SHAREHOLDER_FRIENDLY` | 分红率 + 回购 + 低杠杆综合 | +1 | ★★★ |

**优先级：中低。** 需要跨模块组合，工程复杂度高，但alpha衰减更慢。

---

### 方向13：EPS相关因子

**动机：** q_ps_beps_c（基本EPS）及其YoY增长在数据库中存在但完全未用。

| 因子 | 数据字段 | 非空率 | 方向 | 难度 |
|------|---------|:------:|:----:|:----:|
| `EPS_YOY` | q_ps_beps_c_y2y | 54% | +1 | ★☆☆ |
| `EPS_LEVEL` | q_ps_beps_c | 59% | +1 | ★☆☆ |
| `EPS_SURPRISE` | actual EPS vs consensus EPS 差异 | 59%/99% | +1 | ★★★ |
| `EPS_ACCELERATION` | EPS YoY 的变化 | 54% | +1 | ★★☆ |

**优先级：中。** EPS是最基础的估值锚点。

---

### 方向14：波动率因子

**动机：** OHLCV数据完备，但对波动率的利用仅有MaxDrawdown一个因子。

| 因子 | 数据源 | 方向 | 难度 |
|------|--------|:----:|:----:|
| `REALIZED_VOL` | close 实现波动率 | -1 | ★☆☆ |
| `IDIOSYNCRATIC_VOL` | 残差波动率(回归市场后) | -1 | ★★★ |
| `VOL_OF_VOL` | 波动率的波动率 | -1 | ★★☆ |
| `GARMAN_KLASS_VOL` | OHLC高效波动率估计 | -1 | ★★☆ |
| `PARKINSON_VOL` | High-Low波动率 | -1 | ★★☆ |
| `DOWNSIDE_VOL` | 仅下行波动率 | -1 | ★★☆ |
| `VOL_REGIME` | 高/低波动状态标记 | 分类 | ★★★ |

**优先级：中。** 低波动异象是全球最稳定的alpha来源之一。

---

## 四、优先级总排序

### 第一梯队：数据现成、实现简单、文献支撑强

| 序号 | 方向 | 因子数 | 理由 |
|:----:|------|:------:|------|
| 1 | 估值历史分位 | 6 | 字段直接可用，A股均值回归效应强 |
| 2 | BP_WO_GW | 2 | 一行代码，高质量数据 |
| 3 | 换手率/流动性 | 6 | 数据99%+完整，A股反转信号强 |
| 4 | 分析师预期扩展 | 9 | rating/eps高质量字段未用，全球alpha核心 |

### 第二梯队：字段可用、需适度工程

| 序号 | 方向 | 因子数 | 理由 |
|:----:|------|:------:|------|
| 5 | 利润率水平与趋势 | 6 | DuPont体系核心，补盈利因子缺口 |
| 6 | 杠杆/偿债安全 | 7 | 市场下行期alpha显著 |
| 7 | 营运资本/供应链效率 | 7 | 周期行业信号强 |
| 8 | 现金流结构 | 8 | FCF和现金流结构日益受关注 |

### 第三梯队：有潜力但需更多开发

| 序号 | 方向 | 因子数 | 理由 |
|:----:|------|:------:|------|
| 9 | 股东结构/筹码 | 4 | A股特色因子 |
| 10 | 技术因子扩展 | 12 | 反转+高阶矩因子 |
| 11 | EPS因子 | 4 | 基础但必要 |
| 12 | 波动率因子 | 7 | 低波异象全球稳定 |

### 第四梯队：复杂度高，长期方向

| 序号 | 方向 | 因子数 | 理由 |
|:----:|------|:------:|------|
| 13 | 成长质量复合 | 5 | 跨模块组合，alpha衰减慢 |
| 14 | 融资融券 | 5 | 覆盖率有限（48%） |

---

## 五、数据字段利用率热力图（更新版）

```
模块                已有因子数   数据库字段利用率   可新增方向数
------              ---------   ----------------   ----------
基本面-成长          44           ~40%                3 (EPS、利润率、前向增长)
基本面-盈利          4            ~60%                6 (利润率水平/趋势/稳定性)
基本面-价值          27           ~55%                8 (历史分位、BP_WO_GW、PEG)
基本面-质量          6            ~30%                5+ (现金流结构、非经常占比)
基本面-效率          4            ~25%                7 (CCC、NWC、供应链)
基本面-杠杆          3(含质量)     ~15%                7 (DEBT/CURRENT/QUICK/IB_DEBT)
技术/量价            27+          OHLCV全覆盖         12 (波动率、反转、高阶矩)
流动性/筹码          0            ~0%                 10+ (换手率、股东结构)
分析师预期           4(NI系列)     ~14%                9 (rating、EPS、覆盖度、分歧度)
融资融券             0            0%                  5 (余额、净买入、多空比)
波动率               1            ~5%                 7 (多种波动率估计)
复合因子             0            N/A                 5 (跨维度组合)
```

**合计可新增因子方向：约 84 个**

---

## 六、实施建议

1. **第一批（1-2周）：** 估值历史分位 + BP_WO_GW + 换手率反转。数据直接可用，代码量最小，覆盖3个全新维度。
2. **第二批（2-3周）：** 分析师预期扩展。tushare数据高质量字段利用率极低，EPS修正和评级变动是学术研究中alpha最强的信号之一。
3. **第三批（3-4周）：** 利润率因子 + 杠杆因子 + 现金流结构。补齐基本面因子体系的核心缺口。
4. **长期：** 波动率因子家族 + 成长质量复合因子。需要更多设计和回测验证。

---

*报告基于 factor_framework 代码库扫描、lixinger/tushare/juejin 数据库字段分析、以及 factor_analysis_20260329.md 的实现进度检查。*
