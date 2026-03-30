# 因子体系分析报告
**生成时间：** 2026-03-29
**目的：** 汇总现有因子，对比数据库可用字段，推理可补充的因子组合

---

## 一、现有因子总览

### 1. 技术因子（OHLCV数据）
| 因子类 | 参数 | 描述 |
|--------|------|------|
| `CloseOverMA` | period, field | 价格/移动均线 |
| `RSI` | period, field | 相对强弱指数 |
| `Momentum` | period, field, log_return | 价格动量（收益率） |
| `MACD` | fast/slow/signal period | 异同移动平均线 |
| `BollingerBands` | period, std_multiplier | 布林带位置 |
| `FutureReturn` | period, field | 前瞻收益（评估基准） |

### 2. CTC量价因子（日频OHLCV）
| 分类 | 因子 |
|------|------|
| 高/低成交量收益拆分 | HighVolReturnSum, LowVolReturnSum, HighVolReturnStd, LowVolReturnStd |
| 高/低成交量振幅拆分 | HighVolAmplitude, LowVolAmplitude |
| 放量/缩量收益拆分 | HighVolChangeReturnSum, LowVolChangeReturnSum, HighVolChangeReturnStd, LowVolChangeReturnStd |
| 放量/缩量振幅拆分 | HighVolChangeAmplitude, LowVolChangeAmplitude |
| 价格量能拆分 | HighPriceRelativeVolume, LowPriceRelativeVolume, HighPriceVolumeChange, LowPriceVolumeChange |
| 不平衡因子 | VolAmplitudeImbalance, VolReturnStdImbalance |
| 价量相关 | PVCorr, DPVCorr, PdVCorr, DPdVCorr, PriceVolumeCorrelation |

### 3. 动量因子
| 因子类 | 参数 | 描述 |
|--------|------|------|
| `MomentumFactor` | offset, lookback | 跳过offset期后lookback期收益率 |

### 4. 基本面 - 成长因子（44个）
#### 4.1 增长加速度类
- `ACCEL` - 二阶增长率（增速的增速）
- `COMP_ACCEL` - 复合加速度
- `COMP_GROWTH` - 复合增长
- `COMP_ROBUST_GROWTH` - 复合稳健增长
- `SURPRISE_GROWTH` - 增长惊喜（vs历史均值）

#### 4.2 R&D投入类
- `RD_GROWTH` - R&D增长
- `RD_CAP_RATE` - R&D资本化率（q_bs_rade_t / (q_bs_rade_t + q_ps_rade_c)）
- `RD_COMPOSITE` - R&D综合
- `RD_TO_COST`, `RD_TO_MV`, `RD_TO_NP`, `RD_TO_REV` - R&D占比系列

#### 4.3 广告/专利投入类
- `AD_TO_COST`, `AD_TO_MV`, `AD_TO_NP`, `AD_TO_REV` - 广告费占比
- `PATENT_TO_MV`, `PATENT_TO_NP` - 专利占比

#### 4.4 净利润增长类
- `NI_FG` - 净利润前向增长
- `NI_GROWTH_COMP` - 净利润增长综合
- `NI_LS_FD` - 净利润长短全年预测
- `NI_LT_FT`, `NI_ST_FT` - 净利润长/短期预测趋势

#### 4.5 营收增长类
- `REV_FG` - 营收前向增长
- `REV_GROWTH_COMP` - 营收增长综合
- `REV_LT_FT`, `REV_ST_FT` - 营收长/短期预测趋势
- `SALG_YOY` - 销售额YoY增长

#### 4.6 盈利指标增长综合类
- `ROE_GROWTH_COMP` - ROE增长综合（8子因子）
- `ROA_GROWTH_COMP` - ROA增长综合
- `ROIC_GROWTH_COMP` - ROIC增长综合
- `GPM_GROWTH_COMP` - 毛利率增长综合
- `NPM_GROWTH_COMP` - 净利率增长综合
- `OCF_GROWTH_COMP` - 经营性现金流增长综合
- `OP_GROWTH_COMP` - 营业利润增长综合
- `ATO_GROWTH_COMP` - 资产周转率增长综合
- `NA_GROWTH_COMP` - 净资产增长综合
- `TA_GROWTH_COMP` - 总资产增长综合
- `MULTI_DIM_GROWTH` - 多维增长综合
- `PROG_YOY` - 利润YoY增长
- `PCTS_GROWTH` - 百分比增长
- `RANK_GROWTH` - 排名增长
- `ROBUST_GROWTH` - 稳健增长综合

### 5. 基本面 - 盈利能力因子（4个）
| 因子 | 数据字段 | 描述 |
|------|---------|------|
| `GPM` | q_ps_gp_m_t | 毛利率 |
| `ROE` | q_m_roe_t | 净资产收益率（单季） |
| `ROA` | q_m_roa_t | 总资产收益率（单季） |
| `ROIC` | q_m_roic_t | 投入资本回报率（单季） |

### 6. 基本面 - 价值因子（27个）
| 类别 | 因子 |
|------|------|
| E/P类 | EP_TTM, EP_TTM_RK, EP_TTM_TREND, EP_TTM_ZS, EPFWD, OEP_TTM |
| B/P类 | BP_MRQ, BP_MRQ_RK, BP_MRQ_TREND, BP_MRQ_ZS, IBP_MRQ, RDBP_MRQ |
| S/P类 | SP_TTM |
| 股息类 | DP_TTM, DPFWD |
| 现金流类 | CFP_TTM, CFNOA_TTM, CFEV_TTM |
| EV类 | SEV_TTM, SED_TTM, CFEV_TTM, ENOA_TTM |
| 现金类 | CCP_MRQ |
| 自由现金流 | FCF_MC |
| 前向估值 | FEP, EPFWD |
| 相对估值 | RIMVP, RP_BP, RP_EP |
| 净经营资产 | BNOA_MRQ |

### 7. ML因子
| 因子类 | 描述 |
|--------|------|
| `MLCrossSectionalFactor` | 滚动LightGBM截面因子，输入多个基本面字段 |

---

## 二、数据库可用字段 vs 已使用字段对比

### lixinger.financial_statements — 已使用字段
```
q_m_roe_t, q_m_roa_t, q_m_roic_t, q_m_ta_to_t, q_m_fcf_ttm
q_ps_toi_c, q_ps_oi_c, q_ps_toc_c, q_ps_oc_c, q_ps_se_c, q_ps_ae_c, q_ps_rade_c
q_ps_op_c, q_ps_ebit_c, q_ps_np_c, q_ps_npatoshopc_c, q_ps_gp_m_t
q_ps_toi_c_y2y, q_ps_npatoshopc_c_y2y, q_ps_op_c_y2y, q_ps_np_c_y2y
q_bs_ta_t, q_bs_tl_t, q_bs_toe_t, q_bs_etmsh_t, q_bs_ia_t, q_bs_rade_t
q_bs_stl_t, q_bs_ltl_t, q_bs_fa_t, q_bs_tfa_t
q_cfs_ncffoa_c, q_cfs_ncffoa_ttm, q_cfs_ncffoa_c_y2y, q_cfs_cpfdapdoi_c
```

### lixinger.fundamental — 已使用字段
```
pe_ttm, pb, ps_ttm, dyr, pcf_ttm, mc
```

### 数据库中存在但**尚未使用**的关键字段

#### 利润表未用字段
| 字段 | 含义 | 潜在用途 |
|------|------|---------|
| `q_ps_fe_c` | 财务费用（利息净支出） | 利息覆盖率、财务杠杆质量 |
| `q_ps_npadnrpatoshaopc_c` | 非经常性损益 | 扣非净利润占比、盈余质量 |
| `q_ps_beps_c` | 基本EPS | EPS增长、估值 |
| `q_ps_beps_c_y2y` | EPS同比增长率 | EPS增长因子 |
| `q_ps_npatmsh_c` | 少数股东损益 | 少数股东权益占比（控制权质量） |
| `q_ps_ebitda_t` | EBITDA（TTM） | EV/EBITDA估值、现金创造能力 |
| `q_ps_np_s_r_t` | 净利率（TTM） | 净利率水平及趋势 |
| `q_ps_op_s_r_t` | 营业利润率（TTM） | 营业利润率水平及趋势 |
| `q_ps_wroe_t` | 加权ROE（TTM） | TTM口径ROE（vs单季） |
| `q_ps_cp_c` | 综合收益 | 与净利润差额 = OCI |

#### 资产负债表未用字段
| 字段 | 含义 | 潜在用途 |
|------|------|---------|
| `q_bs_cabb_t` | 货币资金（非空率99.2%） | 现金充裕度、净现金/市值 |
| `q_bs_mc_t` | ~~货币资金~~ → **市值**（已在 financial_statements 中冗余存储，非空率53.9%，建议用 fundamental.mc 代替） | 不建议直接用于因子，易误用 |
| `q_bs_ar_t` | 应收账款 | AR增长 vs 收入增长（信号：坏账风险） |
| `q_bs_nr_t` | 应收票据 | 合并应收分析 |
| `q_bs_i_t` | 存货 | 存货增速 vs 收入增速（信号：需求减弱） |
| `q_bs_gw_t` | 商誉 | 商誉/净资产（商誉风险） |
| `q_bs_ap_t` | 应付账款 | 应付增速 vs 成本增速（供应链话语权） |
| `q_bs_ca_t` | 流动资产 | 流动性分析 |
| `q_bs_cl_t` | 流动负债合计 | 流动比率 |
| `q_bs_nwc_t` | 净营运资本 | NWC/营收（运营效率） |
| `q_bs_ltl_t` | **长期借款**（非长期负债合计，非空率63.2%） | 长期有息债务代理，用于 Piotroski P5 |
| `q_bs_lwi_t` | 有息负债合计（非空率92.0%） | 综合杠杆风险，覆盖率更高 |
| `q_bs_ta_t_y2y` | 总资产YoY增长 | 资产扩张速度 |
| `q_bs_toe_t_y2y` | 净资产YoY增长 | 净资产增速 |
| `q_bs_tl_t_y2y` | 负债YoY增长 | 负债扩张速度 |

#### 派生财务指标未用字段
| 字段 | 含义 | 潜在用途 |
|------|------|---------|
| `q_m_c_r_t` | 流动比率 | 短期偿债能力 |
| `q_m_q_r_t` | 速动比率 | 更严格短期偿债 |
| `q_m_tl_ta_r_t` | 资产负债率 | 财务杠杆风险 |
| `q_m_ar_tor_t` | 应收账款周转率 | 收款效率 |
| `q_m_i_tor_t` | 存货周转率 | 库存管理效率 |
| `q_m_ap_tor_t` | 应付账款周转率 | 应付管理（付款周期） |
| `q_m_ncffoa_np_r_t` | 经营现金流/净利润 | 盈余质量（现金含量） |
| `q_m_ncffoa_op_r_t` | 经营现金流/营业利润 | 利润现金转化率 |
| `q_m_fcf_c` | 自由现金流（单季） | FCF质量分析 |

#### 现金流量表未用字段
| 字段 | 含义 | 潜在用途 |
|------|------|---------|
| `q_cfs_ncffia_c` | 投资活动现金流（单季） | 资本支出强度、扩张型 vs 维持型 |
| `q_cfs_ncfffa_c` | 筹资活动现金流（单季） | 融资依赖度 |
| `q_cfs_np_c` | 现金流量表中的净利润 | 直接法与间接法差异 |

#### lixinger.fundamental 未用字段
| 字段 | 含义 | 潜在用途 |
|------|------|---------|
| `pb_wo_gw` | 剔除商誉的P/B | 更清洁的账面价值估值 |
| `peg` | PEG比率 | 成长性调整的估值 |
| `ev_ebit_r` | EV/EBIT | 企业价值/经营利润 |
| `ev_ebitda_r` | EV/EBITDA | 企业价值/EBITDA（国际常用） |
| `d_pe_ttm` | 动态PE | 基于预测EPS的动态估值 |
| `to_r` | 换手率 | 流动性溢价/流动性因子 |
| `cmc` | 流通市值 | 流通盘大小（vs总市值） |
| `pe_ttm_y5_cvpos` | PE的5年历史分位 | 相对历史估值便宜程度 |
| `pb_y5_cvpos` | PB的5年历史分位 | 相对历史估值 |
| `ps_ttm_y5_cvpos` | PS的5年历史分位 | 相对历史估值 |
| `dyr_y5_cvpos` | 股息率的5年历史分位 | 相对历史股息水平 |

---

## 三、可补充因子组合推理

### 类别A：质量因子（Quality Factors）
**动机：** 现有盈利因子只有水平（ROE/ROA/ROIC）和增长，缺少盈余质量和财务稳健性维度。

| 建议因子 | 计算逻辑 | 使用字段 | 方向 |
|---------|---------|---------|------|
| `EARNINGS_QUALITY` | OCF / Net Profit = `q_m_ncffoa_np_r_t` | q_m_ncffoa_np_r_t | +1 |
| `ACCRUAL_RATIO` | (扣非净利润 - 经营现金流) / 总资产 = (q_ps_npadnrpatoshaopc_c - q_cfs_ncffoa_c) / q_bs_ta_t<br>注: 使用扣非净利润更准确，排除了非经常性损益影响 | q_ps_npadnrpatoshaopc_c,<br>q_cfs_ncffoa_c,<br>q_bs_ta_t | -1 |
| `NON_RECURRING_RATIO` | 非经常性损益 / 净利润 = q_ps_npadnrpatoshaopc_c / q_ps_npatoshopc_c | 两者 | -1 |
| `OCF_OP_RATIO` | 经营现金流/营业利润 = q_m_ncffoa_op_r_t | q_m_ncffoa_op_r_t | +1 |
| `CASH_EARNINGS_COMP` | OCF质量综合（多窗口q_m_ncffoa_np_r_t增长趋势） | q_m_ncffoa_np_r_t | +1 |

### 类别B：财务杠杆/安全边际因子（Leverage & Safety）
**动机：** 现有因子对财务风险覆盖不足；高杠杆股在市场波动时容易暴雷。

| 建议因子 | 计算逻辑 | 使用字段 | 方向 |
|---------|---------|---------|------|
| `DEBT_RATIO` | 资产负债率 = q_m_tl_ta_r_t | q_m_tl_ta_r_t | -1 |
| `NET_DEBT_TO_EQUITY` | (短期借款+长期借款-货币资金) / 净资产 | q_bs_stl_t, q_bs_ltl_t, q_bs_mc_t, q_bs_toe_t | -1 |
| `INTEREST_COVERAGE` | EBIT / 财务费用 = q_ps_ebit_c / q_ps_fe_c | 两者 | +1 |
| `CURRENT_RATIO` | 流动比率 = q_m_c_r_t | q_m_c_r_t | +1 |
| `FINANCIAL_LEVERAGE_GROWTH` | 资产负债率变化趋势（负债扩张速度） | q_bs_tl_t_y2y | -1 |
| `GOODWILL_RISK` | 商誉/净资产 = q_bs_gw_t / q_bs_toe_t | 两者 | -1 |

### 类别C：效率/周转因子（Efficiency Factors）
**动机：** 现有ATO_GROWTH_COMP仅覆盖资产周转率增长，缺少细分周转指标和水平型因子。

| 建议因子 | 计算逻辑 | 使用字段 | 方向 |
|---------|---------|---------|------|
| `AR_TURNOVER` | 应收账款周转率水平 = q_m_ar_tor_t | q_m_ar_tor_t | +1 |
| `INVENTORY_TURNOVER` | 存货周转率水平 = q_m_i_tor_t | q_m_i_tor_t | +1 |
| `AR_GROWTH_VS_REV` | 应收账款增速 - 营收增速（异常时→风险信号） | q_bs_ar_t, q_ps_toi_c | -1 |
| `INVENTORY_GROWTH_VS_REV` | 存货增速 - 营收增速 | q_bs_i_t, q_ps_toi_c | -1 |
| `ASSET_EFFICIENCY_COMP` | AR周转+存货周转+资产周转综合 | q_m_ar_tor_t, q_m_i_tor_t, q_m_ta_to_t | +1 |
| `AR_TURNOVER_GROWTH` | 应收账款周转率增长综合（多窗口） | q_m_ar_tor_t | +1 |
| `INVENTORY_TURNOVER_GROWTH` | 存货周转率增长综合（多窗口） | q_m_i_tor_t | +1 |

### 类别D：估值扩展因子（Extended Valuation）
**动机：** 现有价值因子基于PE/PB/PS，缺乏EV类和动态估值视角。

| 建议因子 | 计算逻辑 | 使用字段 | 方向 |
|---------|---------|---------|------|
| `EV_EBITDA` | EV/EBITDA倒数 = 1 / ev_ebitda_r | ev_ebitda_r | +1 |
| `EV_EBIT` | EV/EBIT倒数 = 1 / ev_ebit_r | ev_ebit_r | +1 |
| `PEG_RATIO` | 1/PEG（成长调整后估值） | peg | +1（PEG低=便宜）|
| `BP_WO_GW` | 剔除商誉的B/P（更干净账面） | pb_wo_gw | +1 |
| `PE_HIST_PERCENTILE` | PE历史分位数（反转） = 1 - pe_ttm_y5_cvpos | pe_ttm_y5_cvpos | +1 |
| `PB_HIST_PERCENTILE` | PB历史分位数（反转） | pb_y5_cvpos | +1 |
| `EBITDA_MARGIN` | EBITDA利润率 = q_ps_ebitda_t / q_ps_toi_c | 两者 | +1 |
| `NET_PROFIT_MARGIN` | 净利率水平 = q_ps_np_s_r_t | q_ps_np_s_r_t | +1 |

### 类别E：成长质量复合因子（Growth-Quality Composite）
**动机：** 单维度增长容易被"注水"增长误导，结合质量约束可提高预测力。

| 建议因子 | 计算逻辑 | 组合维度 |
|---------|---------|---------|
| `GROWTH_QUALITY_COMP` | ROE增长 × OCF质量评分加权 | ROE_GROWTH_COMP + EARNINGS_QUALITY |
| `PROFITABLE_GROWTH` | 高ROE + 高营收增长双筛选 | ROE + REV_GROWTH_COMP |
| `QUALITY_MOMENTUM` | 质量因子 × 价格动量（仅高质量股中找动量） | ROE/ROIC + Momentum |
| `FCF_YIELD_GROWTH` | 自由现金流收益率增长 | q_m_fcf_ttm趋势 + FCF_MC |
| `PIOTROSKI_FSCORE` | Piotroski F-Score（9个二元信号综合） | 见下 |

#### Piotroski F-Score 实现方案（已实现 ✅）
基于9个binary信号，每个满足=1分，满分9分：
```
盈利维度（4项）：
  P1: ROA > 0                        → q_m_roa_t > 0
  P2: OCF > 0                        → q_cfs_ncffoa_c > 0
  P3: ROA增加（YoY）                  → q_m_roa_t vs lag ~250交易日
  P4: OCF > Net Income（质量信号）    → q_m_ncffoa_np_r_t > 1

杠杆/流动性维度（3项）：
  P5: 长期借款比率下降                → q_bs_ltl_t / q_bs_ta_t vs lag
       ⚠️ q_bs_ltl_t = 长期借款（非长期负债合计），为长期有息债务代理
  P6: 流动比率提升                    → q_m_c_r_t vs lag
  P7: 未增发股本（代理）              → (q_ps_np_c / q_ps_beps_c) YoY 未增加
       ⚠️ 原文 q_bs_mc_t 有误（实为市值，非货币资金）
          lixinger 无直接股本字段，改用净利润/基本EPS作为隐含股数代理
          当NP或EPS≤0时该信号为NaN（不纳入计分）

效率维度（2项）：
  P8: 毛利率提升                      → q_ps_gp_m_t vs lag
  P9: 资产周转率提升                  → q_m_ta_to_t vs lag
```
**方向：** +1（F-Score越高越好）
**实现文件：** `factors/fundamental/quality/piotroski_fscore.py`
**配置文件：** `config/factors/piotroski_fscore_grid.yaml`

### 类别F：流动性/筹码因子（Liquidity Factors）
**动机：** 换手率、流通盘等流动性因子在A股具有显著的反转预测效应。

| 建议因子 | 计算逻辑 | 使用字段 | 方向 |
|---------|---------|---------|------|
| `TURNOVER_RATE_REVERSAL` | 短期换手率（反转信号）| to_r（fundamental表） | -1 |
| `TURNOVER_MOMENTUM` | 换手率动量（趋势信号） | to_r多期 | +1 |
| `SIZE_ADJUSTED_TURNOVER` | 换手率 / 流通市值（标准化） | to_r, cmc | 视情 |

### 类别G：技术指标扩展
**动机：** 现有技术因子较基础，可补充更多价格行为特征。

| 建议因子 | 计算逻辑 | 数据 |
|---------|---------|------|
| `ATR_FACTOR` | 平均真实波幅（波动率反转） | OHLCV |
| `PRICE_REVERSAL_SHORT` | 过去1周收益（短期反转） | close |
| `MAX_DRAWDOWN_FACTOR` | 滚动最大回撤 | OHLCV |
| `VOLUME_TREND` | 成交量趋势（OBV类） | vol, close |
| `HIGH_LOW_RATIO` | 近N日最高价/最低价比值 | high, low |
| `CONSECUTIVE_GAINS` | 连续上涨天数 | close |

---

## 四、优先级建议

### 高优先级（实现价值高、字段现成可用）
1. **Piotroski F-Score** — 经典学术因子，多维度综合，A股实证效果好
2. **EARNINGS_QUALITY（OCF/NI）** — q_m_ncffoa_np_r_t直接可用
3. **ACCRUAL_RATIO** — 盈余质量，经典alpha信号
4. **EV_EBITDA / EV_EBIT** — fundamental表直接有ev_ebitda_r, ev_ebit_r字段
5. **GOODWILL_RISK** — A股商誉暴雷频发，负向因子
6. **AR/Inventory Turnover水平因子** — q_m_ar_tor_t, q_m_i_tor_t直接可用

### 中优先级（需要组合多字段）
7. **INTEREST_COVERAGE** — EBIT/财务费用
8. **NET_DEBT_TO_EQUITY** — 净债务/净资产
9. **ASSET_EFFICIENCY_COMP** — 多周转率综合
10. **PEG_RATIO** — peg字段直接可用
11. **PE/PB历史分位** — pe_ttm_y5_cvpos等直接可用

### 低优先级（需要更多工程开发）
12. **GROWTH_QUALITY_COMP** — 需跨模块组合
13. **TURNOVER_RATE_REVERSAL** — to_r字段是否完整待验证
14. **技术指标扩展** — ATR, MaxDrawdown等

---

## 五、数据字段覆盖热力图

```
模块                已有因子数   数据库字段利用率（估计）
------              ---------   ----------------------
基本面-成长          44           ~40%（大量q_m_*字段未用）
基本面-盈利          4            ~60%（缺质量维度）
基本面-价值          27           ~55%（缺EV类、PEG、历史分位）
基本面-质量          0            0%（完全空白）
基本面-效率          0            0%（周转率字段未用）
基本面-杠杆          0            0%（负债类字段未用）
技术/量价            6+20+1       OHLCV全覆盖
ML                  1            部分基本面字段
```

---

*报告生成于 2026-03-29，基于 lixinger.db schema 分析和现有代码扫描*
