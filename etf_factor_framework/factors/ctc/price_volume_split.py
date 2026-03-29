"""
CTC量价因子 - 高低价区间切分因子

基于CTC Institute文章改造，将原始分钟线数据的高/低价格区间切分逻辑
应用于日频OHLCV数据。

原始逻辑：将小时内60根分钟线按价格排序，取前20%高价区间和后20%低价区间。
改造逻辑：使用过去N个交易日作为窗口，按收盘价排序取高/低价日，计算成交量特征。
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


class HighPriceRelativeVolume(OHLCVFactorCalculator):
    """
    高价日相对成交量因子
    
    原理：取过去window日中收盘价最高的top_pct比例交易日（高价日），
         计算这些高价日的成交量均值与整体成交量均值的比值。
    
    经济学解释：高价日伴随的成交量特征可以反映价格在高位的市场参与度。
    若高价日成交量显著高于平均水平，可能表明高位抛压或追涨情绪浓厚。
    
    公式：
        1. 对每只股票，取过去window日数据
        2. 按收盘价排序，取top_pct比例的高价日
        3. 高价日相对成交量 = 高价日成交量均值 / 整体成交量均值
    
    Attributes:
        window: 滚动窗口天数
        top_pct: 高价区间的比例（如0.2表示前20%）
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
        
        # 使用rolling窗口计算高价日相对成交量
        result = self._rolling_high_price_rel_volume(close, volume, self._window, self._top_pct)
        
        # 验证输出
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_price_rel_volume(self, close: pd.DataFrame, volume: pd.DataFrame,
                                        window: int, top_pct: float) -> pd.DataFrame:
        """
        高效计算滚动窗口内高价日的相对成交量
        
        使用向量化操作提高效率：
        1. 对每个时间点，获取过去window日的收盘价和成交量
        2. 按收盘价排序，取top_pct比例的高价日
        3. 计算高价日成交量均值 / 整体成交量均值
        """
        n_stocks, n_periods = close.shape
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        # 将数据转换为numpy数组以加速计算
        close_array = close.values
        vol_array = volume.values
        
        # 对每个时间点计算
        for t in range(window - 1, n_periods):
            # 获取过去window日的数据
            start_idx = t - window + 1
            close_window = close_array[:, start_idx:t+1]  # (n_stocks, window)
            vol_window = vol_array[:, start_idx:t+1]      # (n_stocks, window)
            
            # 计算每只股票在该窗口内的高价日相对成交量
            for i in range(n_stocks):
                close_row = close_window[i]
                vol_row = vol_window[i]
                
                # 检查有效数据
                valid_mask = ~np.isnan(close_row) & ~np.isnan(vol_row)
                if valid_mask.sum() == 0:
                    continue
                
                close_valid = close_row[valid_mask]
                vol_valid = vol_row[valid_mask]
                
                # 按收盘价排序，取高价日
                n_high = max(1, int(len(close_valid) * top_pct))
                high_indices = np.argsort(close_valid)[-n_high:]
                
                # 计算高价日成交量均值 / 整体成交量均值
                high_vol_mean = vol_valid[high_indices].mean()
                overall_vol_mean = vol_valid.mean()
                
                if overall_vol_mean > 0:
                    result.iloc[i, t] = high_vol_mean / overall_vol_mean
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighPriceRelativeVolume_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighPriceRelativeVolume"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowPriceRelativeVolume(OHLCVFactorCalculator):
    """
    低价日相对成交量因子
    
    原理：取过去window日中收盘价最低的top_pct比例交易日（低价日），
         计算这些低价日的成交量均值与整体成交量均值的比值。
    
    经济学解释：低价日伴随的成交量特征可以反映价格在市场低位时的参与度。
    若低价日成交量显著高于平均水平，可能表明低位吸筹或恐慌性抛售。
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
        
        result = self._rolling_low_price_rel_volume(close, volume, self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_price_rel_volume(self, close: pd.DataFrame, volume: pd.DataFrame,
                                       window: int, top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内低价日的相对成交量"""
        n_stocks, n_periods = close.shape
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        close_array = close.values
        vol_array = volume.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            close_window = close_array[:, start_idx:t+1]
            vol_window = vol_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                close_row = close_window[i]
                vol_row = vol_window[i]
                
                valid_mask = ~np.isnan(close_row) & ~np.isnan(vol_row)
                if valid_mask.sum() == 0:
                    continue
                
                close_valid = close_row[valid_mask]
                vol_valid = vol_row[valid_mask]
                
                # 取低价日（与HighPriceRelativeVolume的区别）
                n_low = max(1, int(len(close_valid) * top_pct))
                low_indices = np.argsort(close_valid)[:n_low]
                
                # 计算低价日成交量均值 / 整体成交量均值
                low_vol_mean = vol_valid[low_indices].mean()
                overall_vol_mean = vol_valid.mean()
                
                if overall_vol_mean > 0:
                    result.iloc[i, t] = low_vol_mean / overall_vol_mean
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowPriceRelativeVolume_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowPriceRelativeVolume"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class HighPriceVolumeChange(OHLCVFactorCalculator):
    """
    高价日成交量变化因子
    
    原理：取过去window日中收盘价最高的top_pct比例交易日（高价日），
         计算这些高价日的成交量变化绝对值均值与整体成交量均值的比值。
    
    经济学解释：高价日的成交量变化反映了价格在高位时的资金流动剧烈程度。
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
        
        # 计算成交量变化绝对值
        volume_change_abs = volume.diff(axis=1).abs()
        
        result = self._rolling_high_price_vol_change(close, volume, volume_change_abs, 
                                                      self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_high_price_vol_change(self, close: pd.DataFrame, volume: pd.DataFrame,
                                        volume_change_abs: pd.DataFrame, window: int, 
                                        top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内高价日的成交量变化"""
        n_stocks, n_periods = close.shape
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        close_array = close.values
        vol_array = volume.values
        vol_change_array = volume_change_abs.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            close_window = close_array[:, start_idx:t+1]
            vol_window = vol_array[:, start_idx:t+1]
            vol_change_window = vol_change_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                close_row = close_window[i]
                vol_row = vol_window[i]
                vol_change_row = vol_change_window[i]
                
                valid_mask = (~np.isnan(close_row) & ~np.isnan(vol_row) & 
                             ~np.isnan(vol_change_row))
                if valid_mask.sum() == 0:
                    continue
                
                close_valid = close_row[valid_mask]
                vol_change_valid = vol_change_row[valid_mask]
                vol_valid = vol_row[valid_mask]
                
                # 按收盘价排序，取高价日
                n_high = max(1, int(len(close_valid) * top_pct))
                high_indices = np.argsort(close_valid)[-n_high:]
                
                # 计算高价日成交量变化均值 / 整体成交量均值
                high_vol_change_mean = vol_change_valid[high_indices].mean()
                overall_vol_mean = vol_valid.mean()
                
                if overall_vol_mean > 0:
                    result.iloc[i, t] = high_vol_change_mean / overall_vol_mean
        
        return result
    
    @property
    def name(self) -> str:
        return f"HighPriceVolumeChange_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "HighPriceVolumeChange"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }


class LowPriceVolumeChange(OHLCVFactorCalculator):
    """
    低价日成交量变化因子
    
    原理：取过去window日中收盘价最低的top_pct比例交易日（低价日），
         计算这些低价日的成交量变化绝对值均值与整体成交量均值的比值。
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
        
        volume_change_abs = volume.diff(axis=1).abs()
        
        result = self._rolling_low_price_vol_change(close, volume, volume_change_abs,
                                                     self._window, self._top_pct)
        
        self._validate_output(result, ohlcv_data)
        
        return FactorData(
            values=result,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params
        )
    
    def _rolling_low_price_vol_change(self, close: pd.DataFrame, volume: pd.DataFrame,
                                       volume_change_abs: pd.DataFrame, window: int,
                                       top_pct: float) -> pd.DataFrame:
        """高效计算滚动窗口内低价日的成交量变化"""
        n_stocks, n_periods = close.shape
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        close_array = close.values
        vol_array = volume.values
        vol_change_array = volume_change_abs.values
        
        for t in range(window - 1, n_periods):
            start_idx = t - window + 1
            close_window = close_array[:, start_idx:t+1]
            vol_window = vol_array[:, start_idx:t+1]
            vol_change_window = vol_change_array[:, start_idx:t+1]
            
            for i in range(n_stocks):
                close_row = close_window[i]
                vol_row = vol_window[i]
                vol_change_row = vol_change_window[i]
                
                valid_mask = (~np.isnan(close_row) & ~np.isnan(vol_row) &
                             ~np.isnan(vol_change_row))
                if valid_mask.sum() == 0:
                    continue
                
                close_valid = close_row[valid_mask]
                vol_change_valid = vol_change_row[valid_mask]
                vol_valid = vol_row[valid_mask]
                
                # 取低价日
                n_low = max(1, int(len(close_valid) * top_pct))
                low_indices = np.argsort(close_valid)[:n_low]
                
                # 计算低价日成交量变化均值 / 整体成交量均值
                low_vol_change_mean = vol_change_valid[low_indices].mean()
                overall_vol_mean = vol_valid.mean()
                
                if overall_vol_mean > 0:
                    result.iloc[i, t] = low_vol_change_mean / overall_vol_mean
        
        return result
    
    @property
    def name(self) -> str:
        return f"LowPriceVolumeChange_w{self._window}_p{self._top_pct}"
    
    @property
    def factor_type(self) -> str:
        return "LowPriceVolumeChange"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'window': self._window,
            'top_pct': self._top_pct
        }
