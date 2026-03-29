"""
FundamentalFactorCalculator: 基本面因子计算器抽象基类

所有基于 lixinger 财务数据的因子计算器继承此类，
实现 calculate(fundamental_data) -> FactorData 方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData


class FundamentalFactorCalculator(ABC):
    """
    基本面因子计算器抽象基类

    子类须实现：
    - name (property): 因子名称（含参数）
    - factor_type (property): 因子类型名（不含参数）
    - params (property): 参数字典
    - calculate(fundamental_data) -> FactorData: 因子计算逻辑

    示例::

        class ROE(FundamentalFactorCalculator):
            @property
            def name(self):
                return "ROE"

            @property
            def factor_type(self):
                return "ROE"

            @property
            def params(self):
                return {}

            def calculate(self, fd: FundamentalData) -> FactorData:
                values, symbols, dates = fd.get_daily_panel('q_m_roe_t')
                return FactorData(values=values, symbols=symbols, dates=dates, name=self.name, params=self.params)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """因子名称（含参数，如 'ROE_TTM'）"""
        pass

    @property
    @abstractmethod
    def factor_type(self) -> str:
        """因子类型名（不含参数，如 'ROE'）"""
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """因子参数字典"""
        pass

    @abstractmethod
    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算因子值

        Args:
            fundamental_data: FundamentalData 数据容器

        Returns:
            FactorData: 日频因子面板 (N股 × T日)，index=symbol，columns=交易日
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        return self.params

    # ------------------------------------------------------------------
    # 公共工具方法（子类可直接调用）
    # ------------------------------------------------------------------

    def _winsorize(
        self, df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
    ) -> pd.DataFrame:
        """
        横截面 Winsorize（按日期逐列截尾）

        Args:
            df: N×T DataFrame
            lower: 下分位数
            upper: 上分位数

        Returns:
            截尾后的 DataFrame
        """
        def _clip_col(col):
            q_lo = col.quantile(lower)
            q_hi = col.quantile(upper)
            return col.clip(q_lo, q_hi)

        return df.apply(_clip_col, axis=0)

    def _zscore_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        横截面 Z-Score 标准化（按日期逐列）

        Args:
            df: N×T DataFrame

        Returns:
            Z-Score 标准化后的 DataFrame
        """
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        return (df - mean) / std.replace(0, np.nan)

    def _rank_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        横截面排名（按日期逐列，pct=True 归一化到 [0, 1]）

        Args:
            df: N×T DataFrame

        Returns:
            排名百分比 DataFrame
        """
        return df.rank(axis=0, pct=True)
