"""
SMALL_CAP_COMPOSITE 因子（小市值复合因子）

核心思路：
    A股市场长期存在显著的小市值溢价，本因子以 -log(mc) 为主体信号（市值越小因子值越大），
    同时加入基本面质量过滤，排除盈利极差或高杠杆的壳公司，提升因子质量。

Factor formula:
    SMALL_CAP_COMPOSITE = -log(mc)
    (market-cap floor/ceiling + ROE + debt ratio filters applied via NaN masking)

Data fields:
    - mc              : 总市值（亿元），来源 lixinger.fundamental（日频，无季报延迟）
    - q_m_roe_t       : 净资产收益率TTM，来源 lixinger.financial_statements（季报前推）
    - q_m_tl_ta_r_t   : 资产负债率，来源 lixinger.financial_statements（季报前推）

Factor direction: +1（-log(mc) 值越大 = 市值越小 = 该因子方向下越好）

Filters (applied by setting factor values to NaN, not by removing stocks):
    mc_floor       : 市值下限（亿元），排除微盘壳股，默认 10.0
    mc_cap         : 市值上限（亿元），限制在小盘范围内，默认 200.0（None=不限）
    roe_floor      : ROE(TTM) 下限，排除严重亏损，默认 -0.20（即 -20%）
    debt_ceiling   : 资产负债率上限，排除高杠杆风险，默认 0.90（即 90%）

Notes:
    - 框架已通过 filter_mainboard_only=true 排除 B股/科创板/创业板/北交所，
      本因子无需重复过滤。
    - mc 是日频数据，与估值指标同频，无前瞻泄露。
    - ROE/负债率来自报告日期前推的季报，已通过 report_date 前推，无未来数据泄露。
    - 本因子不做行业/市值中性化（grid.neutralization_methods: [raw]），
      因为中性化会消除市值暴露本身。
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "SMALL_CAP_COMPOSITE"
FACTOR_DIRECTION = 1   # +1: 因子值越大（市值越小）越好

# 财务字段
_FIELD_ROE = "q_m_roe_t"           # 净资产收益率(TTM)
_FIELD_DEBT = "q_m_tl_ta_r_t"      # 资产负债率


def _align_panel(src_vals: np.ndarray, src_syms, target_syms) -> np.ndarray:
    """
    将 src 数据面板按 target_syms 顺序对齐，返回 (N_target, T) 数组。
    未找到的股票行填充 NaN。
    """
    N_target = len(target_syms)
    T = src_vals.shape[1]
    src_idx = {s: i for i, s in enumerate(src_syms.tolist())}
    aligned = np.full((N_target, T), np.nan, dtype=np.float64)
    for i, sym in enumerate(target_syms):
        j = src_idx.get(sym)
        if j is not None:
            aligned[i, :] = src_vals[j, :]
    return aligned


class SmallCapComposite(FundamentalFactorCalculator):
    """
    小市值复合因子

    以 -log(总市值) 为主体信号，叠加基本面质量过滤：
    排除市值过低（壳股）、ROE 极差、高杠杆的公司，
    在小盘股宇宙内按市值升序选股。

    Parameters
    ----------
    mc_floor : float
        市值下限（亿元），低于此值的股票当日置 NaN。默认 10.0。
    mc_cap : float or None
        市值上限（亿元），超过此值的股票当日置 NaN。None 表示不限。默认 200.0。
    roe_floor : float
        ROE(TTM) 下限（小数，如 -0.20 = -20%），低于此值的股票当日置 NaN。默认 -0.20。
    debt_ceiling : float
        资产负债率上限（小数，如 0.90 = 90%），超过此值的股票当日置 NaN。默认 0.90。
    """

    def __init__(
        self,
        mc_floor: float = 10.0,
        mc_cap: float = 200.0,
        roe_floor: float = -0.20,
        debt_ceiling: float = 0.90,
    ):
        self.mc_floor = mc_floor
        self.mc_cap = mc_cap
        self.roe_floor = roe_floor
        self.debt_ceiling = debt_ceiling

    # ------------------------------------------------------------------
    # FundamentalFactorCalculator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        cap_str = f"cap{int(self.mc_cap)}" if self.mc_cap is not None else "capInf"
        return (
            f"{FACTOR_NAME}"
            f"_floor{int(self.mc_floor)}"
            f"_{cap_str}"
            f"_roe{int(self.roe_floor * 100)}"
            f"_debt{int(self.debt_ceiling * 100)}"
        )

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "mc_floor":      self.mc_floor,
            "mc_cap":        self.mc_cap,
            "roe_floor":     self.roe_floor,
            "debt_ceiling":  self.debt_ceiling,
            "direction":     FACTOR_DIRECTION,
        }

    # ------------------------------------------------------------------
    # calculate
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 SMALL_CAP_COMPOSITE 因子日频面板。

        Steps:
        1. 加载日频总市值面板 (mc)
        2. 计算核心因子值: -log(mc)
        3. 施加市值下限/上限过滤
        4. 加载 ROE(TTM) 并施加下限过滤
        5. 加载资产负债率并施加上限过滤
        6. 返回 FactorData
        """
        # ------------------------------------------------------------------
        # Step 1: 加载总市值面板
        # ------------------------------------------------------------------
        mc_raw, mc_syms, mc_dates = fundamental_data.get_market_cap_panel()

        if mc_raw.size == 0:
            raise ValueError(
                "SmallCapComposite: get_market_cap_panel() 返回空数组，"
                "请检查 lixinger.fundamental 中的 mc 字段。"
            )

        mc_raw = mc_raw.astype(np.float64)
        N, T = mc_raw.shape
        symbols_list = mc_syms.tolist()

        # lixinger fundamental.mc 存储单位为【元（yuan）】，
        # 转换为【亿元】方便与用户参数（mc_floor/mc_cap 单位均为亿）对比
        MC_YUAN_PER_YI = 1e8  # 1亿 = 100,000,000元
        mc_vals = mc_raw / MC_YUAN_PER_YI  # 单位: 亿元

        # ------------------------------------------------------------------
        # Step 2: 计算核心信号 -log(mc_亿元)
        # ------------------------------------------------------------------
        # mc 单位为亿元，使用自然对数；mc <= 0 时置 NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            values = np.where(mc_vals > 0, -np.log(mc_vals), np.nan)

        # ------------------------------------------------------------------
        # Step 3: 市值区间过滤（均以亿元为单位）
        # ------------------------------------------------------------------
        # 下限：排除微盘壳股
        below_floor = mc_vals < self.mc_floor
        values[below_floor] = np.nan
        n_below = int(below_floor.sum())

        # 上限：只保留中小市值股票
        n_above = 0
        if self.mc_cap is not None:
            above_cap = mc_vals > self.mc_cap
            values[above_cap] = np.nan
            n_above = int(above_cap.sum())

        if self.mc_cap is not None:
            print(
                f"  [{self.name}] 市值区间过滤(亿): "
                f"<{self.mc_floor:.0f}亿 {n_below} 个股-日, "
                f">{self.mc_cap:.0f}亿 {n_above} 个股-日"
            )
        else:
            print(
                f"  [{self.name}] 市值下限过滤(亿): "
                f"<{self.mc_floor:.0f}亿 {n_below} 个股-日"
            )

        # ------------------------------------------------------------------
        # Step 4: ROE(TTM) 质量过滤
        # ------------------------------------------------------------------
        try:
            roe_vals, roe_syms, _ = fundamental_data.get_daily_panel(_FIELD_ROE)
            roe_aligned = _align_panel(roe_vals, roe_syms, symbols_list)
            # ROE 字段在 lixinger 中存储为小数（如 0.12 = 12%）
            below_roe = (~np.isnan(roe_aligned)) & (roe_aligned < self.roe_floor)
            values[below_roe] = np.nan
            n_roe = int(below_roe.sum())
            print(
                f"  [{self.name}] ROE过滤: "
                f"ROE<{self.roe_floor:.0%} {n_roe} 个股-日"
            )
        except Exception as exc:
            warnings.warn(
                f"SmallCapComposite: ROE 过滤失败 ({exc})，跳过ROE过滤。"
            )

        # ------------------------------------------------------------------
        # Step 5: 资产负债率过滤
        # ------------------------------------------------------------------
        try:
            debt_vals, debt_syms, _ = fundamental_data.get_daily_panel(_FIELD_DEBT)
            debt_aligned = _align_panel(debt_vals, debt_syms, symbols_list)
            # 资产负债率存储为小数（如 0.70 = 70%）
            above_debt = (~np.isnan(debt_aligned)) & (debt_aligned > self.debt_ceiling)
            values[above_debt] = np.nan
            n_debt = int(above_debt.sum())
            print(
                f"  [{self.name}] 负债率过滤: "
                f"负债率>{self.debt_ceiling:.0%} {n_debt} 个股-日"
            )
        except Exception as exc:
            warnings.warn(
                f"SmallCapComposite: 负债率过滤失败 ({exc})，跳过负债率过滤。"
            )

        # ------------------------------------------------------------------
        # Step 6: 质量检查 & 返回
        # ------------------------------------------------------------------
        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.95:
            warnings.warn(
                f"SmallCapComposite NaN 比例极高 ({nan_ratio:.1%})，"
                "请检查 lixinger 数据库的 mc/ROE/负债率字段是否正常。"
            )

        return FactorData(
            values=values,
            symbols=mc_syms,
            dates=mc_dates,
            name=self.name,
            params=self.params,
        )
