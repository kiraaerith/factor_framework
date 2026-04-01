"""
NET_DEBT_TO_EQUITY 因子（净债务/净资产）

衡量公司有息债务净敞口相对净资产的规模，是综合财务杠杆风险的核心指标。

Formula:
    NET_DEBT_TO_EQUITY = (短期借款 + 长期借款 - 货币资金) / 净资产
                       = (q_bs_stl_t + q_bs_ltl_t - q_bs_cabb_t) / q_bs_toe_t

    其中净债务 = 有息债务 - 现金及等价物。净债务为负代表净现金头寸（持有现金多于借款）。

Data fields from lixinger.financial_statements:
    - q_bs_stl_t   : 短期借款，非空率 63.2%（NaN → 视为 0，即无短期银行贷款）
    - q_bs_ltl_t   : 长期借款，非空率 63.2%（NaN → 视为 0，即无长期银行贷款）
    - q_bs_cabb_t  : 货币资金，非空率 99.2%
    - q_bs_toe_t   : 所有者权益合计（净资产），非空率 93.7%

    注：q_bs_stl_t 和 q_bs_ltl_t 仅覆盖银行借款。对于数据缺失情形（63.2% 非空），
        NaN 通常表示该公司无相应银行借款（非金融公司常见），故以 0 填充。
        若两者均为 NaN 且货币资金有效，则净债务 = -cabb（纯持现公司）。

Factor direction: -1
    净债务/净资产越高 = 财务杠杆越重 = 预期超额收益越低。
    净现金头寸（比值 < 0）对应低杠杆、高安全边际，预期表现更好。

边界处理：
    - toe <= 0      → NaN（负净资产公司不适用）
    - cabb 为 NaN  → NaN（货币资金缺失，无法计算）
    - stl/ltl NaN  → 填充为 0（认为无银行借款，而非数据缺失）
    - 极端值由框架层 winsorization 处理

因子类别: 质量 - 财务安全边际 (Quality / Leverage & Safety)

市值过滤：mc < MC_MIN_BILLION 亿元时当日因子值置 NaN，与框架其他质量因子保持一致。

References:
    - Net Debt to Equity: 国际通用财务杠杆指标（Bloomberg、FactSet 标准定义）
    - Penman et al. (2007): 净金融负债在估值模型中的应用
    - A股去杠杆周期（2018-2019）中，高净债务/净资产股票跌幅明显大于市场
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

FACTOR_NAME = "NET_DEBT_TO_EQUITY"
FACTOR_DIRECTION = -1
MC_MIN_BILLION = 50.0  # 市值下限（亿元）

_FIELDS = {
    "stl":  "q_bs_stl_t",   # 短期借款，非空率 63.2%
    "ltl":  "q_bs_ltl_t",   # 长期借款，非空率 63.2%
    "cabb": "q_bs_cabb_t",  # 货币资金，非空率 99.2%
    "toe":  "q_bs_toe_t",   # 所有者权益合计，非空率 93.7%
}


class NetDebtToEquity(FundamentalFactorCalculator):
    """
    净债务/净资产因子：(短期借款 + 长期借款 - 货币资金) / 净资产

    参数
    ----
    无额外参数。因子由四个资产负债表字段直接计算，无需窗口/滞后设置。

    说明
    ----
    - 短期借款和长期借款 NaN 时填充为 0（认为无银行贷款，而非数据缺失）
    - 净资产 ≤ 0 的公司因子值置 NaN
    - 货币资金缺失时因子值置 NaN
    - 市值较小的公司（< 50 亿）因子值置 NaN，避免微市值公司噪音
    - 框架层会对极端值进行 winsorization 处理
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION}

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算净债务/净资产日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 float（含负数）或 NaN。
                     NaN 出现在：
                     - 数据缺失的场景
                     - 净资产 ≤ 0（负净资产公司）
                     - 货币资金数据缺失
                     - 市值 < 50 亿元
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 1: Load raw panels
        # ------------------------------------------------------------------
        all_symbols: set = set()
        raw_panels: dict = {}

        for key, field in _FIELDS.items():
            try:
                vals, syms, dts = fundamental_data.get_daily_panel(field)
                df = pd.DataFrame(
                    vals,
                    index=syms.tolist(),
                    columns=pd.DatetimeIndex(dts),
                )
                raw_panels[key] = df
                all_symbols.update(syms.tolist())
            except Exception as exc:
                warnings.warn(
                    f"NetDebtToEquity: failed to load field '{field}' "
                    f"(key={key}): {exc}. Factor will be all NaN."
                )
                raw_panels[key] = pd.DataFrame(dtype=np.float64)

        if not all_symbols:
            empty = np.empty((0, T), dtype=np.float64)
            return FactorData(
                values=empty,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype="datetime64[ns]"),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        all_symbols_sorted = sorted(all_symbols)
        N = len(all_symbols_sorted)
        td_idx = pd.DatetimeIndex(trading_dates)

        # ------------------------------------------------------------------
        # Step 2: Align to (all_symbols, trading_dates)
        # ------------------------------------------------------------------
        def _align(df: pd.DataFrame) -> np.ndarray:
            if df.empty:
                return np.full((N, T), np.nan, dtype=np.float64)
            aligned = df.reindex(index=all_symbols_sorted, columns=td_idx)
            return aligned.values.astype(np.float64)

        stl_arr  = _align(raw_panels["stl"])   # (N, T) 短期借款
        ltl_arr  = _align(raw_panels["ltl"])   # (N, T) 长期借款
        cabb_arr = _align(raw_panels["cabb"])  # (N, T) 货币资金
        toe_arr  = _align(raw_panels["toe"])   # (N, T) 净资产

        # ------------------------------------------------------------------
        # Step 3: Compute net debt = stl + ltl - cabb
        #   - stl/ltl NaN → 填充为 0（无银行借款，而非缺失）
        #   - cabb  NaN   → NaN（货币资金缺失，无法计算净债务）
        # ------------------------------------------------------------------
        stl_filled  = np.where(np.isnan(stl_arr),  0.0, stl_arr)
        ltl_filled  = np.where(np.isnan(ltl_arr),  0.0, ltl_arr)

        net_debt = np.where(
            ~np.isnan(cabb_arr),
            stl_filled + ltl_filled - cabb_arr,
            np.nan,
        )

        # ------------------------------------------------------------------
        # Step 4: Compute ratio = net_debt / toe
        #   - toe <= 0  → NaN（负净资产无意义）
        #   - toe is NaN → NaN
        # ------------------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                ~np.isnan(net_debt) & ~np.isnan(toe_arr) & (toe_arr > 0),
                net_debt / toe_arr,
                np.nan,
            )

        # ------------------------------------------------------------------
        # Step 5: Market-cap floor filter
        # ------------------------------------------------------------------
        mc_vals, mc_syms, _ = fundamental_data.get_market_cap_panel()
        mc_sym_idx = {s: i for i, s in enumerate(mc_syms.tolist())}
        mc_aligned = np.full((N, T), np.nan, dtype=np.float64)
        for i, sym in enumerate(all_symbols_sorted):
            j = mc_sym_idx.get(sym)
            if j is not None:
                mc_aligned[i, :] = mc_vals[j, :]

        invalid_mc = np.isnan(mc_aligned) | (mc_aligned < MC_MIN_BILLION)
        ratio[invalid_mc] = np.nan

        n_mc_masked = int(invalid_mc.sum())
        if n_mc_masked > 0:
            n_no_mc = int(np.isnan(mc_aligned).sum())
            n_below = n_mc_masked - n_no_mc
            print(
                f"  - 市值过滤: 无市值数据 {n_no_mc} 个股-日, "
                f"<{MC_MIN_BILLION:.0f}亿 {n_below} 个股-日"
            )

        nan_ratio = np.isnan(ratio).mean() if ratio.size > 0 else 1.0
        if nan_ratio > 0.9:
            warnings.warn(
                f"NetDebtToEquity NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if required fields exist in the lixinger database."
            )

        return FactorData(
            values=ratio,
            symbols=np.array(all_symbols_sorted),
            dates=np.array(trading_dates, dtype="datetime64[ns]"),
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (python net_debt_to_equity.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("NetDebtToEquity factor smoke test")
    print("=" * 60)

    TEST_START = "2019-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)

    print("\n[Step 2] Compute NetDebtToEquity factor")
    calculator = NetDebtToEquity()
    result = calculator.calculate(fd)

    print(f"\nFactor shape     : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range       : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    vals = result.values.ravel()
    valid = vals[~np.isnan(vals)]
    print(f"\nValid values     : {len(valid):,} / {len(vals):,} "
          f"({len(valid)/len(vals):.1%})")
    print(f"Net cash stocks  : {(valid < 0).sum():,} ({(valid < 0).mean():.1%}) "
          f"  [net_debt/equity < 0]")
    print(f"Mean             : {valid.mean():.4f}")
    print(f"Median           : {np.median(valid):.4f}")
    print(f"Std              : {valid.std():.4f}")
    print(f"P5 / P95         : {np.percentile(valid, 5):.4f} / "
          f"{np.percentile(valid, 95):.4f}")

    # Show a sample cross-section
    sample_date = pd.Timestamp("2023-06-30")
    date_idx = np.searchsorted(result.dates, np.datetime64(sample_date))
    if 0 <= date_idx < len(result.dates):
        col = result.values[:, date_idx]
        valid_col = col[~np.isnan(col)]
        print(f"\nSample date {sample_date.date()}: "
              f"{len(valid_col)} valid stocks")
        top5_idx = np.argsort(col)[::-1]
        top5_idx = top5_idx[~np.isnan(col[top5_idx])][:5]
        bot5_idx = np.argsort(col)
        bot5_idx = bot5_idx[~np.isnan(col[bot5_idx])][:5]
        print("  Top-5 (most leveraged):")
        for i in top5_idx:
            print(f"    {result.symbols[i]:20s}  {col[i]:+.4f}")
        print("  Bottom-5 (most net-cash):")
        for i in bot5_idx:
            print(f"    {result.symbols[i]:20s}  {col[i]:+.4f}")
