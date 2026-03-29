"""
CTC量价因子 - 高低成交量切分因子

基于CTC Institute文章改造，将原始分钟线数据的高/低成交量切分逻辑
应用于日频OHLCV数据。

原始逻辑：将小时内60根分钟线按成交量排序，取前20%高成交量区间和后20%低成交量区间。
改造逻辑：使用过去N个交易日作为窗口，按成交量排序取高/低成交量日。
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


class HighVolReturnSum(OHLCVFactorCalculator):
    """
    高成交量日收益率之和因子
    
    原理：取过去window日中成交量最高的top_pct比例交易日，
         计算这些高成交量日的收益率之和。
    
    经济学解释：高成交量通常伴随重要信息释放，高成交量日的
    价格变动可能反映市场情绪或趋势强度。
    
    公式：
        1. 计算日收益率: r_t = close_t / close_{t-1} - 1
        2. 对每只股票，取过去window日数据
        3. 按成交量排序，取top_pct比例的高成交量日
        4. 计算这些高成交量日的收益率之和
    
    Attributes:
        window: 滚动窗口天数
        top_pct: 高成交量区间的比例（如0.2表示前20%）
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
        
        # 计算日收益率
        returns = close.pct_change(axis=1)
        
        # 使用rolling窗口计算高成交量日收益率之和
        # 对于每个时间点，我们需要在滚动窗口内按成交量排序
        result = self._rolling_high_vol_sum(volume, returns, self._window, self._top_pct)
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_sum(self, volume: pd.DataFrame, returns: pd.DataFrame, 
                               window: int, top_pct: float) -> pd.DataFrame:
        """
        高效计算滚动窗口内高成交量日的收益率之和
        
        使用向量化操作提高效率：
        1. 构建3D数组: (n_stocks, n_periods, window)
        2. 对最后一个维度按成交量排序
        3. 取top_pct比例的收益率并求和
        """
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        # 将数据转换为numpy数组以加速计算
        vol_array = volume.values
        ret_array = returns.values
        
        # 对每个时间点计算
        for t in range(window - 1, n_periods):
            # 获取过去window日的数据
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]  # (n_stocks, window)
            ret_window = ret_array[:, start_idx:t+1]  # (n_stocks, window)
            
            # 计算每只股票在该窗口内的高成交量日收益率之和
            for i in range(n_stocks):
                vol_row = vol_window[i]
                ret_row = ret_window[i]
                
                # 检查有效数据
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(ret_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_valid = vol_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                # 按成交量排序，取高成交量日
                n_high = max(1, int(len(vol_valid) * top_pct))
                high_indices = np.argsort(vol_valid)[-n_high:]
                
                # 计算高成交量日的收益率之和
                result.iloc[i, t] = ret_valid[high_indices].sum()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolReturnSum_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolReturnSum"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolReturnSum(OHLCVFactorCalculator):
    """
    低成交量日收益率之和因子
    
    原理：取过去window日中成交量最低的top_pct比例交易日，
         计算这些低成交量日的收益率之和。
    
    经济学解释：低成交量日的价格变动可能反映市场缺乏方向性，
    或趋势中的休整阶段。
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
        result = self._rolling_low_vol_sum(volume, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_sum(self, volume: pd.DataFrame, returns: pd.DataFrame,
                              window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内低成交量日的收益率之和"""
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        vol_array = volume.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_row = vol_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(ret_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_valid = vol_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                # 取低成交量日（与HighVolReturnSum的区别）
                n_low = max(1, int(len(vol_valid) * top_pct))
                low_indices = np.argsort(vol_valid)[:n_low]
                
                result.iloc[i, t] = ret_valid[low_indices].sum()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolReturnSum_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolReturnSum"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class HighVolReturnStd(OHLCVFactorCalculator):
    """
    高成交量日收益率标准差因子
    
    原理：取过去window日中成交量最高的top_pct比例交易日，
         计算这些高成交量日的收益率标准差。
    
    经济学解释：高成交量日的收益率波动反映了市场在高活跃度
    时期的价格不确定性或分歧程度。
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
        result = self._rolling_high_vol_std(volume, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_std(self, volume: pd.DataFrame, returns: pd.DataFrame,
                               window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内高成交量日的收益率标准差"""
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        vol_array = volume.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_row = vol_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(ret_row)
                if valid_mask.sum() < 2:  # 标准差需要至少2个数据点
                    continue
                
                vol_valid = vol_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                n_high = max(1, int(len(vol_valid) * top_pct))
                high_indices = np.argsort(vol_valid)[-n_high:]
                
                result.iloc[i, t] = ret_valid[high_indices].std()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolReturnStd_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolReturnStd"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolReturnStd(OHLCVFactorCalculator):
    """
    低成交量日收益率标准差因子
    
    原理：取过去window日中成交量最低的top_pct比例交易日，
         计算这些低成交量日的收益率标准差。
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
        result = self._rolling_low_vol_std(volume, returns, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_std(self, volume: pd.DataFrame, returns: pd.DataFrame,
                              window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内低成交量日的收益率标准差"""
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        vol_array = volume.values
        ret_array = returns.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]
            ret_window = ret_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_row = vol_window[i]
                ret_row = ret_window[i]
                
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(ret_row)
                if valid_mask.sum() < 2:
                    continue
                
                vol_valid = vol_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                n_low = max(1, int(len(vol_valid) * top_pct))
                low_indices = np.argsort(vol_valid)[:n_low]
                
                result.iloc[i, t] = ret_valid[low_indices].std()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolReturnStd_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolReturnStd"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class HighVolAmplitude(OHLCVFactorCalculator):
    """
    高成交量日振幅均值因子
    
    原理：取过去window日中成交量最高的top_pct比例交易日，
         计算这些高成交量日的日振幅均值。
    
    振幅 = (high - low) / close
    
    经济学解释：高成交量日的振幅反映了当日价格的波动范围，
    可衡量市场情绪或流动性。
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
        
        result = self._rolling_high_vol_mean(volume, amplitude, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_vol_mean(self, volume: pd.DataFrame, values: pd.DataFrame,
                                window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内高成交量日的特征均值"""
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        vol_array = volume.values
        val_array = values.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]
            val_window = val_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_row = vol_window[i]
                val_row = val_window[i]
                
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(val_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_valid = vol_row[valid_mask]
                val_valid = val_row[valid_mask]
                
                n_high = max(1, int(len(vol_valid) * top_pct))
                high_indices = np.argsort(vol_valid)[-n_high:]
                
                result.iloc[i, t] = val_valid[high_indices].mean()
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighVolAmplitude_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighVolAmplitude"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowVolAmplitude(OHLCVFactorCalculator):
    """
    低成交量日振幅均值因子
    
    原理：取过去window日中成交量最低的top_pct比例交易日，
         计算这些低成交量日的日振幅均值。
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
        result = self._rolling_low_vol_mean(volume, amplitude, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_vol_mean(self, volume: pd.DataFrame, values: pd.DataFrame,
                               window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内低成交量日的特征均值"""
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        vol_array = volume.values
        val_array = values.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]
            val_window = val_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                vol_row = vol_window[i]
                val_row = val_window[i]
                
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(val_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_valid = vol_row[valid_mask]
                val_valid = val_row[valid_mask]
                
                n_low = max(1, int(len(vol_valid) * top_pct))
                low_indices = np.argsort(vol_valid)[:n_low]
                
                result.iloc[i, t] = val_valid[low_indices].mean()
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowVolAmplitude_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowVolAmplitude"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }
