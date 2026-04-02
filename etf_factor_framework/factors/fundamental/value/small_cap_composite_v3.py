"""
SMALL_CAP_COMPOSITE_V3 因子（小市值复合因子 v3）

v3 在 v2 基础上修复了三个结构性问题：

    F1. 【quality_score NaN 传播修复】
        v2 中当 ROE/GPM/CFNI 三项均缺失时 quality_score 为 NaN，
        叠加后导致已通过市值/换手/ROE/负债过滤的合格股票被误杀。
        v3 修复：叠加前将 quality_score 的 NaN 替换为 0，
        即"无质量数据"等同于"质量中性"，不给予惩罚也不给予奖励。

    F2. 【行业约束前预过滤非主板股票】
        v2 中 _apply_industry_cap() 在全部 ~6050 只股票上执行（含创业板/科创板），
        非主板股票占据行业名额后被框架删除，导致主板小盘股被挤出该行业的 top-N。
        v3 修复：在行业约束前将非主板股票（非 SHSE.6xxxxx/SZSE.0xxxxx）设为 NaN，
        使行业名额只在主板内竞争。

    F3. 【quality z-score 限定在过滤后宇宙】
        v2 中 quality_score 的 z-score 参考宇宙包含全市场所有股票（大盘/非主板），
        小盘股内部质量差异被稀释。
        v3 修复：在市值/换手/ROE/负债过滤完成后（Steps 2-4），
        用 values 的 NaN mask 过滤质量面板，只有通过基本面过滤的候选股参与 z-score 计算。

Factor formula (unchanged from v2):
    quality_score = cross_zscore_on_filtered_universe(
        nanmean( zscore(ROE_TTM), zscore(GrossMargin_TTM), zscore(CFNI_TTM) )
    )
    三项均缺失 → quality_score = 0（v3 修复 F1）

    SMALL_CAP_COMPOSITE_V3 = -log(mc_亿) + quality_alpha * quality_score

Data fields (same as v2):
    - mc              : 总市值（元，lixinger.fundamental，日频）
    - to_r            : 换手率（lixinger.fundamental，日频）
    - q_m_roe_t       : 净资产收益率TTM（季报前推）
    - q_m_tl_ta_r_t   : 资产负债率（季报前推）
    - q_ps_gp_m_t     : 毛利率TTM（季报前推）
    - q_m_ncffoa_np_r_t: 经营现金流/净利润TTM（季报前推）

Factor direction: +1（因子值越大 = 市值越小且质量更好 = 越好）

Parameters (same as v2)
----------
mc_floor : float
    市值下限（亿元）。默认 10.0。
mc_cap : float or None
    市值上限（亿元）。默认 200.0。None 表示不限。
roe_floor : float
    ROE(TTM) 下限（小数）。默认 -0.20。
debt_ceiling : float
    资产负债率上限（小数）。默认 0.90。
gpm_floor : float or None
    毛利率(TTM) 下限（小数）。默认 None（不过滤，仅参与 quality_score）。
cfni_floor : float or None
    经营现金流/净利润(TTM) 下限。默认 None（不过滤，仅参与 quality_score）。
turnover_floor : float
    换手率20日滚动均值下限（小数，如 0.005 = 0.5%/天）。默认 0.005。
turnover_window : int
    计算滚动均值的窗口（交易日数）。默认 20。
quality_alpha : float
    quality_score 的权重系数。默认 0.3。
max_per_industry : int or None
    每个行业最多保留的候选股数量。默认 3。
"""

import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "SMALL_CAP_V3"
FACTOR_DIRECTION = 1  # +1: 因子值越大越好

# 财务/估值字段
_FIELD_ROE   = "q_m_roe_t"            # 净资产收益率(TTM)
_FIELD_DEBT  = "q_m_tl_ta_r_t"        # 资产负债率
_FIELD_GPM   = "q_ps_gp_m_t"          # 毛利率(TTM)
_FIELD_CFNI  = "q_m_ncffoa_np_r_t"    # 经营现金流/净利润(TTM)
_FIELD_TOR   = "to_r"                  # 换手率（日频，来自 get_valuation_panel）


# ---------------------------------------------------------------------------
# 工具函数（与 v2 相同）
# ---------------------------------------------------------------------------

def _align_panel(src_vals: np.ndarray, src_syms, target_syms) -> np.ndarray:
    """将 src 面板按 target_syms 顺序对齐，未找到的行填充 NaN。"""
    N_target = len(target_syms)
    T = src_vals.shape[1]
    src_idx = {s: i for i, s in enumerate(src_syms.tolist())}
    aligned = np.full((N_target, T), np.nan, dtype=np.float64)
    for i, sym in enumerate(target_syms):
        j = src_idx.get(sym)
        if j is not None:
            aligned[i, :] = src_vals[j, :]
    return aligned


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    计算逐行滚动均值（沿 T 轴，即时间轴为 axis=1）。
    前 window-1 个时间步的结果为 NaN（无足够历史）。
    输入 arr: (N, T)，输出 (N, T)。
    """
    N, T = arr.shape
    result = np.full_like(arr, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        window_slice = arr[:, t - window + 1 : t + 1]  # (N, window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[:, t] = np.nanmean(window_slice, axis=1)
    return result


def _cross_zscore(arr: np.ndarray) -> np.ndarray:
    """
    逐日截面标准化（沿 N 轴）。
    arr: (N, T)，返回 (N, T) z-score，NaN 不参与均值/标准差计算。
    标准差为 0 时该列输出 NaN。
    """
    result = np.full_like(arr, np.nan, dtype=np.float64)
    N, T = arr.shape
    for t in range(T):
        col = arr[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        mu = np.nanmean(col)
        sigma = np.nanstd(col)
        if sigma < 1e-10:
            continue
        result[valid, t] = (col[valid] - mu) / sigma
    return result


def _apply_industry_cap(
    values: np.ndarray,
    symbols_list: list,
    industry_map: dict,
    max_per_industry: int,
) -> np.ndarray:
    """
    行业分散约束：对每个交易日，每个行业只保留因子值最高的 max_per_industry 只股票，
    其余置 NaN。

    注意：调用此函数前应已将非主板股票设为 NaN（F2 修复），
    确保行业名额只在主板股票间竞争。

    Parameters
    ----------
    values : np.ndarray (N, T)，会被就地修改并返回
    symbols_list : list of str，长度 N
    industry_map : dict {symbol: industry_str}
    max_per_industry : int

    Returns
    -------
    values : np.ndarray (N, T)，已施加行业约束
    """
    N, T = values.shape

    sym_industry = [industry_map.get(sym, "_unknown_") for sym in symbols_list]

    industry_to_indices: dict = defaultdict(list)
    for i, ind in enumerate(sym_industry):
        industry_to_indices[ind].append(i)

    for ind, idxs in industry_to_indices.items():
        if len(idxs) <= max_per_industry:
            continue

        idxs_arr = np.array(idxs)
        sub = values[idxs_arr, :]

        for t in range(T):
            col = sub[:, t]
            not_nan = ~np.isnan(col)
            n_valid = not_nan.sum()
            if n_valid <= max_per_industry:
                continue
            valid_positions = np.where(not_nan)[0]
            valid_values    = col[valid_positions]
            order = np.argsort(-valid_values)
            to_nan = valid_positions[order[max_per_industry:]]
            for pos in to_nan:
                values[idxs_arr[pos], t] = np.nan

    return values


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------

class SmallCapCompositeV3(FundamentalFactorCalculator):
    """
    小市值复合因子 v3

    v2 的全部改进 + 三项结构性修复：
      F1. quality_score 全 NaN → 替换为 0，不惩罚无数据股票
      F2. 行业约束前预过滤非主板股票，避免创业板/科创板占据名额
      F3. quality z-score 限定在通过基本面过滤的候选股宇宙内，
          提升小盘股内部质量区分度

    Parameters
    ----------
    mc_floor : float, default 10.0
    mc_cap : float or None, default 200.0
    roe_floor : float, default -0.20
    debt_ceiling : float, default 0.90
    gpm_floor : float or None, default None
    cfni_floor : float or None, default None
    turnover_floor : float, default 0.005
    turnover_window : int, default 20
    quality_alpha : float, default 0.3
    max_per_industry : int or None, default 3
    """

    def __init__(
        self,
        mc_floor: float = 10.0,
        mc_cap: float = 200.0,
        roe_floor: float = -0.20,
        debt_ceiling: float = 0.90,
        gpm_floor: float = None,
        cfni_floor: float = None,
        turnover_floor: float = 0.005,
        turnover_window: int = 20,
        quality_alpha: float = 0.3,
        max_per_industry: int = 3,
    ):
        self.mc_floor         = mc_floor
        self.mc_cap           = mc_cap
        self.roe_floor        = roe_floor
        self.debt_ceiling     = debt_ceiling
        self.gpm_floor        = gpm_floor
        self.cfni_floor       = cfni_floor
        self.turnover_floor   = turnover_floor
        self.turnover_window  = turnover_window
        self.quality_alpha    = quality_alpha
        self.max_per_industry = max_per_industry

    # ------------------------------------------------------------------
    # FundamentalFactorCalculator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        cap_str = f"cap{int(self.mc_cap)}" if self.mc_cap is not None else "capInf"
        ind_str = f"ind{self.max_per_industry}" if self.max_per_industry is not None else "indInf"
        gpm_str = f"gpm{int(self.gpm_floor * 100)}" if self.gpm_floor is not None else "gpmOff"
        cfni_str = f"cfni{int(self.cfni_floor * 10)}" if self.cfni_floor is not None else "cfniOff"
        return (
            f"{FACTOR_NAME}"
            f"_fl{int(self.mc_floor)}"
            f"_{cap_str}"
            f"_roe{int(self.roe_floor * 100)}"
            f"_dbt{int(self.debt_ceiling * 100)}"
            f"_{gpm_str}"
            f"_{cfni_str}"
            f"_tor{int(self.turnover_floor * 1000)}"
            f"_qa{int(self.quality_alpha * 10)}"
            f"_{ind_str}"
        )

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "mc_floor":         self.mc_floor,
            "mc_cap":           self.mc_cap,
            "roe_floor":        self.roe_floor,
            "debt_ceiling":     self.debt_ceiling,
            "gpm_floor":        self.gpm_floor,
            "cfni_floor":       self.cfni_floor,
            "turnover_floor":   self.turnover_floor,
            "turnover_window":  self.turnover_window,
            "quality_alpha":    self.quality_alpha,
            "max_per_industry": self.max_per_industry,
            "direction":        FACTOR_DIRECTION,
        }

    # ------------------------------------------------------------------
    # calculate
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 SMALL_CAP_COMPOSITE_V3 因子日频面板。

        Steps:
         1. 加载市值面板，转换单位，计算 -log(mc)
         2. 市值区间过滤
         3. 流动性过滤（换手率滚动均值）
         4. ROE 过滤 + 资产负债率过滤
         5. 毛利率过滤 + 现金流质量过滤
         6. [F3] 取当前 values NaN mask 作为质量 z-score 的参考宇宙
         7. 计算 quality_score（截面 z-score 合成，仅在候选股宇宙内）
            [F1] 质量全缺失时 quality_score = 0，不惩罚无数据股票
         8. [F2] 行业约束前预过滤非主板股票（非 SHSE.6x/SZSE.0x 置 NaN）
         9. 行业分散约束（每行业最多 max_per_industry 只）
        10. 返回 FactorData
        """
        # ------------------------------------------------------------------
        # Step 1: 市值面板 + 核心信号
        # ------------------------------------------------------------------
        mc_raw, mc_syms, mc_dates = fundamental_data.get_market_cap_panel()

        if mc_raw.size == 0:
            raise ValueError(
                f"{FACTOR_NAME}: get_market_cap_panel() 返回空数组，"
                "请检查 lixinger.fundamental 中的 mc 字段。"
            )

        mc_raw = mc_raw.astype(np.float64)
        N, T = mc_raw.shape
        symbols_list = mc_syms.tolist()

        # lixinger.fundamental.mc 单位为【元】，转换为【亿元】
        MC_YUAN_PER_YI = 1e8
        mc_vals = mc_raw / MC_YUAN_PER_YI  # 亿元

        with np.errstate(invalid="ignore", divide="ignore"):
            base_signal = np.where(mc_vals > 0, -np.log(mc_vals), np.nan)

        values = base_signal.copy()

        # ------------------------------------------------------------------
        # Step 2: 市值区间过滤
        # ------------------------------------------------------------------
        below_floor = mc_vals < self.mc_floor
        values[below_floor] = np.nan

        if self.mc_cap is not None:
            above_cap = mc_vals > self.mc_cap
            values[above_cap] = np.nan

        n_mc_nan = int(np.isnan(values).sum() - np.isnan(base_signal).sum())
        print(f"  [{self.name}] 市值区间过滤后新增NaN: {n_mc_nan} 个股-日")

        # ------------------------------------------------------------------
        # Step 3: 流动性过滤（换手率20日滚动均值）
        # ------------------------------------------------------------------
        try:
            tor_vals, tor_syms, _ = fundamental_data.get_valuation_panel(_FIELD_TOR)
            tor_aligned = _align_panel(tor_vals, tor_syms, symbols_list)
            tor_rolling = _rolling_mean(tor_aligned, self.turnover_window)
            low_tor = (~np.isnan(tor_rolling)) & (tor_rolling < self.turnover_floor)
            values[low_tor] = np.nan
            n_tor = int(low_tor.sum())
            print(
                f"  [{self.name}] 流动性过滤: "
                f"换手率{self.turnover_window}日均<{self.turnover_floor:.3f}"
                f"({self.turnover_floor*100:.1f}%) {n_tor} 个股-日"
            )
        except Exception as exc:
            warnings.warn(f"{FACTOR_NAME}: 流动性过滤失败 ({exc})，已跳过。")

        # ------------------------------------------------------------------
        # Step 4: ROE 过滤 + 资产负债率过滤
        # ------------------------------------------------------------------
        roe_aligned = None
        try:
            roe_vals, roe_syms, _ = fundamental_data.get_daily_panel(_FIELD_ROE)
            roe_aligned = _align_panel(roe_vals, roe_syms, symbols_list)
            below_roe = (~np.isnan(roe_aligned)) & (roe_aligned < self.roe_floor)
            values[below_roe] = np.nan
            print(
                f"  [{self.name}] ROE过滤: "
                f"ROE<{self.roe_floor:.0%} {int(below_roe.sum())} 个股-日"
            )
        except Exception as exc:
            warnings.warn(f"{FACTOR_NAME}: ROE 过滤失败 ({exc})，已跳过。")

        try:
            debt_vals, debt_syms, _ = fundamental_data.get_daily_panel(_FIELD_DEBT)
            debt_aligned = _align_panel(debt_vals, debt_syms, symbols_list)
            above_debt = (~np.isnan(debt_aligned)) & (debt_aligned > self.debt_ceiling)
            values[above_debt] = np.nan
            print(
                f"  [{self.name}] 负债率过滤: "
                f"负债率>{self.debt_ceiling:.0%} {int(above_debt.sum())} 个股-日"
            )
        except Exception as exc:
            warnings.warn(f"{FACTOR_NAME}: 负债率过滤失败 ({exc})，已跳过。")

        # ------------------------------------------------------------------
        # Step 5: 毛利率过滤 + 现金流质量过滤
        # ------------------------------------------------------------------
        gpm_aligned = None
        try:
            gpm_vals, gpm_syms, _ = fundamental_data.get_daily_panel(_FIELD_GPM)
            gpm_aligned = _align_panel(gpm_vals, gpm_syms, symbols_list)
            if self.gpm_floor is not None:
                below_gpm = (~np.isnan(gpm_aligned)) & (gpm_aligned < self.gpm_floor)
                values[below_gpm] = np.nan
                print(
                    f"  [{self.name}] 毛利率过滤: "
                    f"毛利率<{self.gpm_floor:.0%} {int(below_gpm.sum())} 个股-日"
                )
            else:
                print(f"  [{self.name}] 毛利率过滤: 关闭（仅用于quality_score）")
        except Exception as exc:
            warnings.warn(f"{FACTOR_NAME}: 毛利率加载失败 ({exc})，已跳过。")

        cfni_aligned = None
        try:
            cfni_vals, cfni_syms, _ = fundamental_data.get_daily_panel(_FIELD_CFNI)
            cfni_aligned = _align_panel(cfni_vals, cfni_syms, symbols_list)
            if self.cfni_floor is not None:
                below_cfni = (~np.isnan(cfni_aligned)) & (cfni_aligned < self.cfni_floor)
                values[below_cfni] = np.nan
                print(
                    f"  [{self.name}] 现金流质量过滤: "
                    f"CFNI<{self.cfni_floor:.1f} {int(below_cfni.sum())} 个股-日"
                )
            else:
                print(f"  [{self.name}] 现金流质量过滤: 关闭（仅用于quality_score）")
        except Exception as exc:
            warnings.warn(f"{FACTOR_NAME}: 现金流质量加载失败 ({exc})，已跳过。")

        # ------------------------------------------------------------------
        # Step 6+7: quality_score（F3: 限定在过滤后宇宙; F1: NaN → 0）
        # ------------------------------------------------------------------
        if self.quality_alpha != 0:
            # F3: 取当前 values 的有效 mask 作为质量 z-score 的参考宇宙
            # 只有通过市值/换手/ROE/负债过滤的股票参与截面标准化
            candidate_mask = ~np.isnan(values)  # (N, T) bool

            quality_components = []

            if roe_aligned is not None:
                # 仅候选股的 ROE 参与 z-score
                roe_filtered = np.where(candidate_mask, roe_aligned, np.nan)
                quality_components.append(_cross_zscore(roe_filtered))

            if gpm_aligned is not None:
                gpm_filtered = np.where(candidate_mask, gpm_aligned, np.nan)
                quality_components.append(_cross_zscore(gpm_filtered))

            if cfni_aligned is not None:
                cfni_clip = np.clip(cfni_aligned, -5.0, 10.0)
                cfni_filtered = np.where(candidate_mask, cfni_clip, np.nan)
                quality_components.append(_cross_zscore(cfni_filtered))

            if quality_components:
                comp_stack = np.stack(quality_components, axis=0)  # (K, N, T)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    quality_score = np.nanmean(comp_stack, axis=0)   # (N, T)

                # 再做一次截面标准化（同样限定在候选股宇宙）
                quality_score_filtered = np.where(candidate_mask, quality_score, np.nan)
                quality_score = _cross_zscore(quality_score_filtered)

                # F1: 将 quality_score 中的 NaN 替换为 0
                # 三项均缺失的股票 quality_score = 0（质量中性），不被误杀
                qs = quality_score.copy()
                qs[np.isnan(qs)] = 0.0

                # 叠加到因子值
                active = ~np.isnan(values)
                values[active] += self.quality_alpha * qs[active]

                n_zero_qs = int((quality_score[active] != quality_score[active]).sum())
                print(
                    f"  [{self.name}] quality_score 合成 "
                    f"({len(quality_components)} 成分, alpha={self.quality_alpha}), "
                    f"NaN→0 修复股-日: {n_zero_qs}"
                )
            else:
                print(f"  [{self.name}] quality_score 无可用成分，跳过复合化")

        # ------------------------------------------------------------------
        # Step 8: F2 — 行业约束前预过滤非主板股票
        # 主板: SHSE.6xxxxx (沪市主板) 或 SZSE.0xxxxx (深市主板)
        # ------------------------------------------------------------------
        if self.max_per_industry is not None:
            n_nonmb_before = int((~np.isnan(values)).sum())
            for i, sym in enumerate(symbols_list):
                # 提取6位代码部分
                code = sym.split('.')[-1] if '.' in sym else sym
                if not (code.startswith('6') or code.startswith('0')):
                    values[i, :] = np.nan
            n_nonmb_filtered = n_nonmb_before - int((~np.isnan(values)).sum())
            print(
                f"  [{self.name}] 非主板预过滤(F2): "
                f"清除 {n_nonmb_filtered} 个非主板股-日（创业板/科创板等）"
            )

        # ------------------------------------------------------------------
        # Step 9: 行业分散约束
        # ------------------------------------------------------------------
        if self.max_per_industry is not None:
            industry_map = fundamental_data.get_industry_map()
            before_nan = int(np.isnan(values).sum())
            values = _apply_industry_cap(
                values, symbols_list, industry_map, self.max_per_industry
            )
            after_nan = int(np.isnan(values).sum())
            print(
                f"  [{self.name}] 行业分散约束(max={self.max_per_industry}/行业): "
                f"新增NaN {after_nan - before_nan} 个股-日"
            )

        # ------------------------------------------------------------------
        # Step 10: 质量检查 & 返回
        # ------------------------------------------------------------------
        nan_ratio = np.isnan(values).mean()
        print(f"  [{self.name}] 最终NaN比例: {nan_ratio:.1%}")
        if nan_ratio > 0.97:
            warnings.warn(
                f"{FACTOR_NAME} NaN 比例极高 ({nan_ratio:.1%})，"
                "请检查数据库字段或放宽过滤参数。"
            )

        return FactorData(
            values=values,
            symbols=mc_syms,
            dates=mc_dates,
            name=self.name,
            params=self.params,
        )
