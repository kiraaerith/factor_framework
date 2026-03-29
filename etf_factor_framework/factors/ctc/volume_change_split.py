"""
CTC量价因子 - 放量缩量切分因子

基于CTC Institute文章改造，将原始分钟线数据的放量/缩量切分逻辑
应用于日频OHLCV数据。

原始逻辑：使用成交量的一阶差分序列进行切分，取放量区间（差分大）和缩量区间（差分小）。
改造逻辑：使用过去N个交易日作为窗口，按成交量变化排序取放量/缩量日。
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

try:
    from ..ohlcv_calculator import OHLCVFactorCalculator
    from ...core.ohlcv_data import OHLCVData
    from ...core.factor_data import FactorData
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from factors.ohlcv_calculator import OHLCVFactorCalculator
    from core.ohlcv_data import OHLCVData
    from core.factor_data import FactorData


class HighVolChangeReturnSum(OHLCVFactorCalculator):
    """
    放量日收益率之和因子
    
    原理：取过去window日中成交量变化（一阶差分）最高的top_pct比例交易日，
         计算这些放量日的收益率之和。
    
    经济学解释：成交量放大通常表示市场情绪高涨或重要信息发布，
    放量日的价格变动可能反映资金流入的推动作用。
    
    公式：
        1. 计算成交量变化: vol_change_t = volume_t - volume_{t-1}
        2. 计算日收益率: r_t = close_t / close_{t-1} - 1
        3. 对每只股票，取过去window日数据
        4. 按成交量变化排序，取top_pct比例的放量日
        5. 计算这些放量日的收益率之和
    
    Attributes:
        window: 滚动窗口天数
        top_pct: 放量区间的比例（如0.2表示前20%）
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        # 获取数据
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        # 计算日收益率和成交量变化
        returns = close.pct_change(axis=1)
        volume_change = volume.diff(axis=1)
        
        # 使用rolling窗口计算放量日收益率之和
        result = self._rolling_high_vol_change_sum(volume_change, returns, self._window, self._top_pct)
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_change_sum(self, volume_change: pd.DataFrame, returns: pd.DataFrame, 
                                      window: int, top_pct: float) -> pd.DataFrame:
        """
        高效计算滚动窗口内放量日的收益率之和
        """
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        # 将数据转换为numpy数组以加速计算
        vol_change_array = volume_change.values
        ret_array = returns.values
        
        # 对每个时间点计算
        for t in range(window - 1, n_periods):
            # 获取过去window日的数据
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            # 计算每只股票在该窗口内的放量日收益率之和
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                ret_row = ret_window[i]
                
                # 检查有效数据
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(ret_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                # 按成交量变化排序，取放量日
                n_high = max(1, int(len(vol_change_valid) * top_pct))
                high_indices = np.argsort(vol_change_valid)[-n_high:]
                
                # 计算放量日的收益率之和
                result.iloc[i, t] = ret_valid[high_indices].sum()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolChangeReturnSum_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolChangeReturnSum"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolChangeReturnSum(OHLCVFactorCalculator):
    """
    缩量日收益率之和因子
    
    原理：取过去window日中成交量变化（一阶差分）最低的top_pct比例交易日，
         计算这些缩量日的收益率之和。
    
    经济学解释：成交量萎缩通常表示市场情绪降温或观望情绪浓厚，
    缩量日的价格变动可能反映市场缺乏方向性。
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        returns = close.pct_change(axis=1)
        volume_change = volume.diff(axis=1)
        result = self._rolling_low_vol_change_sum(volume_change, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_change_sum(self, volume_change: pd.DataFrame, returns: pd.DataFrame,
                                     window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内缩量日的收益率之和"""
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        vol_change_array = volume_change.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(ret_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                # 取缩量日（成交量变化最小的）
                n_low = max(1, int(len(vol_change_valid) * top_pct))
                low_indices = np.argsort(vol_change_valid)[:n_low]
                
                result.iloc[i, t] = ret_valid[low_indices].sum()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolChangeReturnSum_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolChangeReturnSum"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class HighVolChangeReturnStd(OHLCVFactorCalculator):
    """
    放量日收益率标准差因子
    
    原理：取过去window日中成交量变化（一阶差分）最高的top_pct比例交易日，
         计算这些放量日的收益率标准差。
    
    经济学解释：放量日的收益率波动反映了市场在高成交量变化时期的价格不确定性。
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        returns = close.pct_change(axis=1)
        volume_change = volume.diff(axis=1)
        result = self._rolling_high_vol_change_std(volume_change, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_change_std(self, volume_change: pd.DataFrame, returns: pd.DataFrame,
                                      window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内放量日的收益率标准差"""
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        vol_change_array = volume_change.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(ret_row)
                if valid_mask.sum() < 2:  # 标准差需要至少2个数据点
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                n_high = max(1, int(len(vol_change_valid) * top_pct))
                high_indices = np.argsort(vol_change_valid)[-n_high:]
                
                result.iloc[i, t] = ret_valid[high_indices].std()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolChangeReturnStd_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolChangeReturnStd"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolChangeReturnStd(OHLCVFactorCalculator):
    """
    缩量日收益率标准差因子
    
    原理：取过去window日中成交量变化（一阶差分）最低的top_pct比例交易日，
         计算这些缩量日的收益率标准差。
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        returns = close.pct_change(axis=1)
        volume_change = volume.diff(axis=1)
        result = self._rolling_low_vol_change_std(volume_change, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_change_std(self, volume_change: pd.DataFrame, returns: pd.DataFrame,
                                     window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内缩量日的收益率标准差"""
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        vol_change_array = volume_change.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(ret_row)
                if valid_mask.sum() < 2:
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                n_low = max(1, int(len(vol_change_valid) * top_pct))
                low_indices = np.argsort(vol_change_valid)[:n_low]
                
                result.iloc[i, t] = ret_valid[low_indices].std()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolChangeReturnStd_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolChangeReturnStd"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class HighVolChangeAmplitude(OHLCVFactorCalculator):
    """
    放量日振幅均值因子
    
    原理：取过去window日中成交量变化（一阶差分）最高的top_pct比例交易日，
         计算这些放量日的日振幅均值。
    
    振幅 = (high - low) / close
    
    经济学解释：放量日的振幅反映了当日价格的波动范围，
    可衡量市场情绪在成交量变化时的波动。
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        high = ohlcv_data.high
        low = ohlcv_data.low
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        # 计算日振幅
        amplitude = (high - low) / close
        # 计算成交量变化
        volume_change = volume.diff(axis=1)
        
        result = self._rolling_high_vol_change_mean(volume_change, amplitude, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_change_mean(self, volume_change: pd.DataFrame, values: pd.DataFrame,
                                       window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内放量日的特征均值"""
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        vol_change_array = volume_change.values
        val_array = values.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            val_window = val_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                val_row = val_window[i]
                
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(val_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                val_valid = val_row[valid_mask]
                
                n_high = max(1, int(len(vol_change_valid) * top_pct))
                high_indices = np.argsort(vol_change_valid)[-n_high:]
                
                result.iloc[i, t] = val_valid[high_indices].mean()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolChangeAmplitude_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolChangeAmplitude"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolChangeAmplitude(OHLCVFactorCalculator):
    """
    缩量日振幅均值因子
    
    原理：取过去window日中成交量变化（一阶差分）最低的top_pct比例交易日，
         计算这些缩量日的日振幅均值。
    """
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self._window = window
        self._top_pct = top_pct
        super().__init__()
    
    def _validate_params(self):
        if not isinstance(self._window, int) or self._window <= 0:
            raise ValueError(f"window must be positive integer, got {self._window}")
        if not isinstance(self._top_pct, (int, float)) or not (0 < self._top_pct <= 0.5):
            raise ValueError(f"top_pct must be between 0 and 0.5, got {self._top_pct}")
    
    def calculate(self, ohlcv_data: OHLCVData) -> FactorData:
        high = ohlcv_data.high
        low = ohlcv_data.low
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        amplitude = (high - low) / close
        volume_change = volume.diff(axis=1)
        result = self._rolling_low_vol_change_mean(volume_change, amplitude, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_change_mean(self, volume_change: pd.DataFrame, values: pd.DataFrame,
                                      window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内缩量日的特征均值"""
        n_stocks, n_periods = volume_change.shape
        result = pd.DataFrame(np.nan, index=volume_change.index, columns=volume_change.columns)
        
        vol_change_array = volume_change.values
        val_array = values.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_change_window = vol_change_array[:, start_idx:t+1]
            val_window = val_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_change_row = vol_change_window[i]
                val_row = val_window[i]
                
                valid_mask = ~np.isnan(vol_change_row) & ~np.isnan(val_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_change_valid = vol_change_row[valid_mask]
                val_valid = val_row[valid_mask]
                
                n_low = max(1, int(len(vol_change_valid) * top_pct))
                low_indices = np.argsort(vol_change_valid)[:n_low]
                
                result.iloc[i, t] = val_valid[low_indices].mean()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolChangeAmplitude_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolChangeAmplitude"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }
