"""
CTC量价因子 - 不平衡度因子

基于CTC Institute文章改造，计算高/低区间特征的不平衡度。

原始逻辑：不平衡度 = (高区特征 - 低区特征) / (高区特征 + 低区特征)
改造逻辑：使用日频OHLCV数据计算各类特征的不平衡度。
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


class VolAmplitudeImbalance(OHLCVFactorCalculator):
    """
    高/低成交量日振幅不平衡度因子
    
    原理：取过去window日中成交量最高和最低的top_pct比例交易日，
         计算高成交量日振幅均值与低成交量日振幅均值之间的不平衡度。
    
    公式：
        imbalance = (high_vol_amplitude - low_vol_amplitude) / (high_vol_amplitude + low_vol_amplitude)
    
    经济学解释：
        不平衡度反映了高成交量日与低成交量日的振幅差异。
        - 正值：高成交量日振幅更大，表明高成交量伴随更大价格波动
        - 负值：低成交量日振幅更大，表明低成交量时价格波动更大
        - 绝对值越大，差异越显著
    
    Attributes:
        window: 滚动窗口天数
        top_pct: 高/低成交量区间的比例（如0.2表示前20%和后20%）
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
        high = ohlcv_data.high
        low = ohlcv_data.low
        close = ohlcv_data.close
        volume = ohlcv_data.volume
        
        # 计算日振幅
        amplitude = (high - low) / close
        
        # 计算高/低成交量日振幅不平衡度
        result = self._rolling_vol_amplitude_imbalance(volume, amplitude, self._window, self._top_pct)
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_vol_amplitude_imbalance(self, volume: pd.DataFrame, amplitude: pd.DataFrame,
                                          window: int, top_pct: float) -> pd.DataFrame:
        """
        高效计算滚动窗口内高/低成交量日振幅不平衡度
        """
        n_stocks, n_periods = volume.shape
        result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
        
        # 将数据转换为numpy数组以加速计算
        vol_array = volume.values
        amp_array = amplitude.values
        
        # 对每个时间点计算
        for t in range(window - 1, n_periods):
            # 获取过去window日的数据
            start_idx = t - window + 1
            vol_window = vol_array[:, start_idx:t+1]  # (n_stocks, window)
            amp_window = amp_array[:, start_idx:t+1]    # (n_stocks, window)
            
            # 计算每只股票的高/低成交量日振幅不平衡度
            for i in range(n_stocks):
                vol_row = vol_window[i]
                amp_row = amp_window[i]
                
                # 检查有效数据
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(amp_row)
                if valid_mask.sum() == 0:
                    continue
                
                vol_valid = vol_row[valid_mask]
                amp_valid = amp_row[valid_mask]
                
                # 按成交量排序
                n_extreme = max(1, int(len(vol_valid) * top_pct))
                
                # 取高成交量日（排序后最后n_extreme个）
                high_indices = np.argsort(vol_valid)[-n_extreme:]
                high_vol_amplitude = amp_valid[high_indices].mean()
                
                # 取低成交量日（排序后前n_extreme个）
                low_indices = np.argsort(vol_valid)[:n_extreme]
                low_vol_amplitude = amp_valid[low_indices].mean()
                
                # 计算不平衡度
                sum_amp = high_vol_amplitude + low_vol_amplitude
                if sum_amp > 0:
                    result.iloc[i, t] = (high_vol_amplitude - low_vol_amplitude) / sum_amp
        
        return result
    
    @property
    def name(self) -> str:
        return f"VolAmplitudeImbalance_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "VolAmplitudeImbalance"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class VolReturnStdImbalance(OHLCVFactorCalculator):
    """
    高/低成交量日收益率标准差不平衡度因子
    
    原理：取过去window日中成交量最高和最低的top_pct比例交易日，
         计算高成交量日收益率标准差与低成交量日收益率标准差之间的不平衡度。
    
    公式：
        imbalance = (high_vol_std - low_vol_std) / (high_vol_std + low_vol_std)
    
    经济学解释：
        不平衡度反映了高成交量日与低成交量日的收益率波动差异。
        - 正值：高成交量日波动更大，表明高成交量伴随更大不确定性
        - 负值：低成交量日波动更大，表明低成交量时价格更易波动
        - 绝对值越大，差异越显著
    
    Attributes:
        window: 滚动窗口天数
        top_pct: 高/低成交量区间的比例（如0.2表示前20%和后20%）
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
        
        # 计算高/低成交量日收益率标准差不平衡度
        result = self._rolling_vol_return_std_imbalance(volume, returns, self._window, self._top_pct)
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_vol_return_std_imbalance(self, volume: pd.DataFrame, returns: pd.DataFrame,
                                           window: int, top_pct: float) -> pd.DataFrame:
        """
        高效计算滚动窗口内高/低成交量日收益率标准差不平衡度
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
            
            # 计算每只股票的高/低成交量日收益率标准差不平衡度
            for i in range(n_stocks):
                vol_row = vol_window[i]
                ret_row = ret_window[i]
                
                # 检查有效数据（标准差需要至少2个数据点）
                valid_mask = ~np.isnan(vol_row) & ~np.isnan(ret_row)
                if valid_mask.sum() < 2:
                    continue
                
                vol_valid = vol_row[valid_mask]
                ret_valid = ret_row[valid_mask]
                
                # 按成交量排序
                n_extreme = max(1, int(len(vol_valid) * top_pct))
                
                # 取高成交量日（排序后最后n_extreme个）
                high_indices = np.argsort(vol_valid)[-n_extreme:]
                high_vol_std = ret_valid[high_indices].std()
                
                # 取低成交量日（排序后前n_extreme个）
                low_indices = np.argsort(vol_valid)[:n_extreme]
                low_vol_std = ret_valid[low_indices].std()
                
                # 计算不平衡度
                sum_std = high_vol_std + low_vol_std
                if sum_std > 0:
                    result.iloc[i, t] = (high_vol_std - low_vol_std) / sum_std
        
        return result
    
    @property
    def name(self) -> str:
        return f"VolReturnStdImbalance_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "VolReturnStdImbalance"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }
