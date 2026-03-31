"""
ASSET_EFFICIENCY_COMP 因子（资产运营效率综合因子）

综合应收账款周转率、存货周转率、资产周转率三个效率维度，
对各分量截面 z-score 标准化后等权平均，形成综合效率评分。

计算逻辑：
  z1 = cross_section_zscore(q_m_ar_tor_t)    # 应收账款周转率
  z2 = cross_section_zscore(q_m_i_tor_t)     # 存货周转率
  z3 = cross_section_zscore(q_m_ta_to_t)     # 资产周转率
  ASSET_EFFICIENCY_COMP = nanmean(z1, z2, z3)  # 至少 min_components 个分量有效才计算

数据字段（lixinger financial_statements）：
  - q_m_ar_tor_t : 应收账款周转率 TTM（周转次数，越高越好）
  - q_m_i_tor_t  : 存货周转率 TTM（周转次数，越高越好）
  - q_m_ta_to_t  : 总资产周转率 TTM（越高越好）

因子方向：+1（综合效率越高越好）
因子类别：运营效率 - 综合

Notes:
  - z-score 截面标准化基于当日所有有效股票，防止量纲差异影响合成权重。
  - 若某股票某日 z-score 中有效分量数 < min_components，结果置 NaN。
  - 极端值（>3σ）在 z-score 前先做截面 3σ winsorize，防止异常值拉偏均值。
  - 市值过滤：日市值 < 50亿 或无数据的股票当日因子值置 NaN。
  - q_m_i_tor_t 对金融股和无存货行业可能大量为 NaN，此时该分量缺失，
    其余两个分量仍可贡献评分（min_components 默认为 1）。
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "ASSET_EFFICIENCY_COMP"
FACTOR_DIRECTION = 1  # 正向

MC_MIN_BILLION = 50.0  # 市值下限（亿元）

_FIELDS = {
    "ar_turnover":        "q_m_ar_tor_t",   # 应收账款周转率 TTM
    "inventory_turnover": "q_m_i_tor_t",    # 存货周转率 TTM
    "asset_turnover":     "q_m_ta_to_t",    # 总资产周转率 TTM
}

_DEFAULT_MIN_COMPONENTS = 1   # 至少1个分量有有效数据即计算综合分


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cross_section_winsorize(arr: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """
    截面 3σ winsorize（逐日）

    arr: shape (N, T)，NaN 会被忽略
    Return: same shape, 截面极端值被裁剪，NaN 保留
    """
    out = arr.copy()
    N, T = out.shape
    for t in range(T):
        col = out[:, t]
        valid = col[~np.isnan(col)]
        if valid.size < 5:
            continue
        mu = valid.mean()
        sigma = valid.std()
        if sigma == 0:
            continue
        lo = mu - n_sigma * sigma
        hi = mu + n_sigma * sigma
        col_clipped = np.where(np.isnan(col), col, np.clip(col, lo, hi))
        out[:, t] = col_clipped
    return out


def _cross_section_zscore(arr: np.ndarray) -> np.ndarray:
    """
    截面 z-score（逐日）

    arr: shape (N, T)，NaN 会被忽略
    Return: same shape，每日截面标准化后的 z-score，NaN 保留
    """
    out = np.full_like(arr, np.nan, dtype=np.float64)
    N, T = arr.shape
    for t in range(T):
        col = arr[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 5:
            continue
        mu = col[mask].mean()
        sigma = col[mask].std()
        if sigma == 0:
            continue
        out[mask, t] = (col[mask] - mu) / sigma
    return out


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------

class AssetEfficiencyComp(FundamentalFactorCalculator):
    """
    资产运营效率综合因子（ASSET_EFFICIENCY_COMP）

    参数
    ----
    min_components : int
        每股每日至少需要多少个分量的有效 z-score，不足则置 NaN，默认为 1。
    """

    def __init__(self, min_components: int = _DEFAULT_MIN_COMPONENTS):
        self._min_components = min_components

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "direction": FACTOR_DIRECTION,
            "min_components": self._min_components,
        }

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 ASSET_EFFICIENCY_COMP 日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为三个效率分量截面 z-score 的等权平均。
        """
        # ------------------------------------------------------------------
        # Step 1: 载入三个分量面板，对齐到 union symbol 集合
        # ------------------------------------------------------------------
        panels = {}  # component_name -> (values_NxT, symbols_array, dates_array)
        dates_ref = None
        all_symbols_set = set()

        for comp_name, field in _FIELDS.items():
            try:
                vals, syms, dates = fundamental_data.get_daily_panel(field)
            except Exception as exc:
                warnings.warn(
                    f"ASSET_EFFICIENCY_COMP: field '{field}' ({comp_name}) failed to load: {exc}. "
                    "This component will be skipped."
                )
                continue

            if vals.size == 0:
                warnings.warn(
                    f"ASSET_EFFICIENCY_COMP: field '{field}' ({comp_name}) returned empty array, skipping."
                )
                continue

            panels[comp_name] = (vals.astype(np.float64), syms, dates)
            all_symbols_set.update(syms.tolist())
            if dates_ref is None:
                dates_ref = dates

        if not panels:
            raise ValueError(
                "ASSET_EFFICIENCY_COMP: all three component fields failed to load. "
                "Please check lixinger database."
            )

        if len(panels) < self._min_components:
            raise ValueError(
                f"ASSET_EFFICIENCY_COMP: only {len(panels)} components loaded, "
                f"but min_components={self._min_components} required."
            )

        T = len(dates_ref)
        all_symbols = sorted(all_symbols_set)
        sym_idx = {s: i for i, s in enumerate(all_symbols)}
        N = len(all_symbols)

        # ------------------------------------------------------------------
        # Step 2: 将各分量 align 到 union symbol 集合
        # ------------------------------------------------------------------
        aligned: dict[str, np.ndarray] = {}
        for comp_name, (vals, syms, _) in panels.items():
            aligned_vals = np.full((N, T), np.nan, dtype=np.float64)
            for local_i, sym in enumerate(syms.tolist()):
                global_i = sym_idx.get(sym)
                if global_i is not None:
                    aligned_vals[global_i, :] = vals[local_i, :]
            aligned[comp_name] = aligned_vals

        # ------------------------------------------------------------------
        # Step 3: 截面 winsorize + z-score 标准化每个分量
        # ------------------------------------------------------------------
        zscores: list[np.ndarray] = []
        for comp_name, vals in aligned.items():
            wins = _cross_section_winsorize(vals)
            z = _cross_section_zscore(wins)
            zscores.append(z)
            nan_pct = np.isnan(z).mean()
            print(f"  - {comp_name}: NaN={nan_pct:.1%}")

        # ------------------------------------------------------------------
        # Step 4: 等权平均（NaN-tolerant），不足 min_components 个分量时置 NaN
        # ------------------------------------------------------------------
        stacked = np.stack(zscores, axis=0)  # shape: (n_components, N, T)
        n_valid = np.sum(~np.isnan(stacked), axis=0)  # shape: (N, T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            composite = np.nanmean(stacked, axis=0)   # shape: (N, T)
        composite[n_valid < self._min_components] = np.nan

        # ------------------------------------------------------------------
        # Step 5: 市值过滤（<50亿 或无数据的股票当日置 NaN）
        # ------------------------------------------------------------------
        try:
            mc_vals, mc_syms, _ = fundamental_data.get_market_cap_panel()
            mc_sym_idx = {s: i for i, s in enumerate(mc_syms.tolist())}

            mc_aligned = np.full((N, T), np.nan, dtype=np.float64)
            for i, sym in enumerate(all_symbols):
                j = mc_sym_idx.get(sym)
                if j is not None:
                    mc_aligned[i, :] = mc_vals[j, :]

            invalid_mc = np.isnan(mc_aligned) | (mc_aligned < MC_MIN_BILLION)
            composite[invalid_mc] = np.nan

            n_mc_masked = int(invalid_mc.sum())
            if n_mc_masked > 0:
                n_no_mc = int(np.isnan(mc_aligned).sum())
                n_below = n_mc_masked - n_no_mc
                print(
                    f"  - 市值过滤: 无市值数据 {n_no_mc} 个股-日, "
                    f"<{MC_MIN_BILLION:.0f}亿 {n_below} 个股-日"
                )
        except Exception as exc:
            warnings.warn(
                f"ASSET_EFFICIENCY_COMP: market-cap filter skipped due to error: {exc}. "
                "Factor values will NOT be market-cap filtered."
            )

        # ------------------------------------------------------------------
        # Step 6: 质量检查
        # ------------------------------------------------------------------
        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"ASSET_EFFICIENCY_COMP NaN ratio is very high ({nan_ratio:.1%}). "
                "Check efficiency fields in lixinger database."
            )
        else:
            print(f"  - 综合因子 NaN={nan_ratio:.1%} (有效股-日占比 {1 - nan_ratio:.1%})")

        return FactorData(
            values=composite,
            symbols=np.array(all_symbols),
            dates=dates_ref,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python asset_efficiency_comp.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("ASSET_EFFICIENCY_COMP factor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036",
                  "601166", "600276", "300750", "002594", "600900"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print("\n[Step 2] Calculate ASSET_EFFICIENCY_COMP factor")
    calc = AssetEfficiencyComp()
    result = calc.calculate(fd)

    print(f"\n[Step 3] Results")
    print(f"  Factor shape: {result.values.shape}  (N={len(result.symbols)}, T={len(result.dates)})")
    print(f"  Symbols: {result.symbols.tolist()}")
    print(f"  Date range: {result.dates[0]} ~ {result.dates[-1]}")
    print(f"  Value stats:")
    valid = result.values[~np.isnan(result.values)]
    if valid.size > 0:
        print(f"    mean={valid.mean():.4f}  std={valid.std():.4f}  "
              f"min={valid.min():.4f}  max={valid.max():.4f}")
    else:
        print("    (all NaN)")

    print("\n[Step 4] Sample values (last 5 dates)")
    df = pd.DataFrame(result.values, index=result.symbols,
                      columns=pd.DatetimeIndex(result.dates))
    print(df.iloc[:, -5:].T.to_string())

    print("\nDone.")
