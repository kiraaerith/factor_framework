"""
配置验证模块

提供配置验证功能，确保配置的正确性和完整性。
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os

from .base_config import Config, DataConfig, FactorConfig, MapperConfig, EvaluationConfig, StorageConfig


class ValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigValidator:
    """
    配置验证器
    
    验证配置的正确性和完整性。
    
    Attributes:
        errors: 错误信息列表
        warnings: 警告信息列表
    """
    
    # 支持的因子类型
    SUPPORTED_FACTOR_TYPES = [
        'RSI', 'CloseOverMA', 'Momentum', 'MACD', 'BollingerBands', 'FutureReturn',
        # 动量因子（带 offset 参数的新版）
        'MomentumFactor',
        # CTC因子 - 高低成交量切分
        'HighVolReturnSum', 'LowVolReturnSum', 'HighVolReturnStd', 'LowVolReturnStd',
        'HighVolAmplitude', 'LowVolAmplitude',
        # CTC因子 - 放量缩量切分
        'HighVolChangeReturnSum', 'LowVolChangeReturnSum',
        'HighVolChangeReturnStd', 'LowVolChangeReturnStd',
        'HighVolChangeAmplitude', 'LowVolChangeAmplitude',
    ]
    
    # 支持的映射器类型
    SUPPORTED_MAPPER_TYPES = ['rank_based', 'direct', 'quantile', 'zscore']
    
    # 支持的权重方法
    SUPPORTED_WEIGHT_METHODS = ['equal', 'softmax', 'risk_parity', 'inverse_variance']
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: Config) -> Tuple[bool, List[str], List[str]]:
        """
        验证完整配置
        
        Args:
            config: 配置对象
            
        Returns:
            tuple: (是否通过, 错误列表, 警告列表)
        """
        self.errors = []
        self.warnings = []
        
        # 验证各部分
        self._validate_data_config(config.data)
        self._validate_factors_config(config.factors)
        self._validate_mapper_config(config.mapper)
        self._validate_evaluation_config(config.evaluation)
        self._validate_storage_config(config.storage)
        
        # 验证整体一致性
        self._validate_consistency(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_data_config(self, config: DataConfig):
        """验证数据配置（兼容 csv 和 duckdb 两种数据源）"""
        data_source = getattr(config, 'data_source', 'csv')

        if data_source == 'csv':
            # CSV 模式：验证 csv_path
            if not config.csv_path:
                self.errors.append("数据配置错误: csv 模式下 csv_path 不能为空")
            elif not os.path.exists(config.csv_path):
                # 警告而非错误，因为可能是相对路径
                self.warnings.append(f"数据配置警告: CSV文件不存在: {config.csv_path}")

            # 验证列名（仅 csv 模式需要）
            if not config.symbol_col:
                self.errors.append("数据配置错误: symbol_col不能为空")
            if not config.date_col:
                self.errors.append("数据配置错误: date_col不能为空")

            # 验证OHLCV列名
            required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
            for key in required_ohlcv:
                if key not in config.ohlcv_cols:
                    self.warnings.append(f"数据配置警告: OHLCV列名缺少 '{key}'")

        elif data_source == 'duckdb':
            # DuckDB 模式：验证必要字段
            if not config.start_date:
                self.errors.append("数据配置错误: duckdb 模式下 start_date 不能为空")
            if not config.end_date:
                self.errors.append("数据配置错误: duckdb 模式下 end_date 不能为空")
            db_path = getattr(config, 'db_path', None)
            if db_path and not os.path.exists(db_path):
                self.warnings.append(f"数据配置警告: DuckDB文件不存在: {db_path}")

        else:
            self.errors.append(f"数据配置错误: 不支持的 data_source '{data_source}'，"
                               "可选值: 'csv' 或 'duckdb'")

        # 验证非空比例（两种模式都验证）
        if not 0 <= config.min_non_null_ratio <= 1:
            self.errors.append(
                f"数据配置错误: min_non_null_ratio 必须在[0, 1]之间，"
                f"当前值: {config.min_non_null_ratio}"
            )
    
    def _validate_factors_config(self, factors: List[FactorConfig]):
        """验证因子配置"""
        if not factors:
            self.errors.append("因子配置错误: 至少需要配置一个因子")
            return
        
        factor_names = set()
        for i, factor in enumerate(factors):
            # 验证名称
            if not factor.name:
                self.errors.append(f"因子配置错误[#{i}]: 因子名称不能为空")
            elif factor.name in factor_names:
                self.errors.append(f"因子配置错误: 因子名称 '{factor.name}' 重复")
            else:
                factor_names.add(factor.name)
            
            # 验证类型
            if not factor.type:
                self.errors.append(f"因子配置错误[{factor.name}]: 因子类型不能为空")
            elif factor.type not in self.SUPPORTED_FACTOR_TYPES:
                self.warnings.append(
                    f"因子配置警告[{factor.name}]: 未知的因子类型 '{factor.type}'，"
                    f"支持的类型: {self.SUPPORTED_FACTOR_TYPES}"
                )
            
            # 验证参数
            self._validate_factor_params(factor.name, factor.type, factor.params)
    
    def _validate_factor_params(self, factor_name: str, factor_type: str, params: Dict[str, Any]):
        """验证因子参数"""
        if factor_type == 'RSI':
            period = params.get('period', 14)
            if not isinstance(period, int) or period < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: RSI period必须是正整数")
            if period < 5:
                self.warnings.append(f"因子参数警告[{factor_name}]: RSI period={period} 较小，可能产生较多噪声")
        
        elif factor_type == 'CloseOverMA':
            ma_period = params.get('ma_period', 20)
            if not isinstance(ma_period, int) or ma_period < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: CloseOverMA ma_period必须是正整数")
        
        elif factor_type in ('Momentum', 'MomentumFactor'):
            lookback = params.get('lookback', 20)
            if not isinstance(lookback, int) or lookback < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: {factor_type} lookback必须是正整数")
            if factor_type == 'MomentumFactor':
                offset = params.get('offset', 0)
                if not isinstance(offset, int) or offset < 0:
                    self.errors.append(f"因子参数错误[{factor_name}]: MomentumFactor offset必须是非负整数")
        
        elif factor_type == 'MACD':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            if fast >= slow:
                self.errors.append(f"因子参数错误[{factor_name}]: MACD fast({fast})必须小于slow({slow})")
        
        elif factor_type == 'BollingerBands':
            period = params.get('period', 20)
            num_std = params.get('num_std', 2.0)
            if not isinstance(period, int) or period < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: BollingerBands period必须是正整数")
            if num_std <= 0:
                self.errors.append(f"因子参数错误[{factor_name}]: BollingerBands num_std必须是正数")
        
        elif factor_type == 'HighVolReturnSum':
            window = params.get('window', 20)
            top_pct = params.get('top_pct', 0.2)
            if not isinstance(window, int) or window < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: HighVolReturnSum window必须是正整数")
            if not 0 < top_pct <= 1:
                self.errors.append(f"因子参数错误[{factor_name}]: HighVolReturnSum top_pct必须在(0, 1]范围内")
        
        elif factor_type in ('LowVolReturnSum', 'HighVolReturnStd', 'LowVolReturnStd', 'HighVolAmplitude', 'LowVolAmplitude'):
            window = params.get('window', 20)
            top_pct = params.get('top_pct', 0.2)
            if not isinstance(window, int) or window < 1:
                self.errors.append(f"因子参数错误[{factor_name}]: {factor_type} window必须是正整数")
            if not 0 < top_pct <= 0.5:
                self.errors.append(f"因子参数错误[{factor_name}]: {factor_type} top_pct必须在(0, 0.5]范围内")
    
    def _validate_mapper_config(self, config: MapperConfig):
        """验证映射器配置"""
        if not config.type:
            self.errors.append("映射器配置错误: type不能为空")
            return
        
        if config.type not in self.SUPPORTED_MAPPER_TYPES:
            self.warnings.append(
                f"映射器配置警告: 未知的映射器类型 '{config.type}'，"
                f"支持的类型: {self.SUPPORTED_MAPPER_TYPES}"
            )
        
        params = config.params
        
        # 验证各类型特定参数
        if config.type == 'rank_based':
            top_k = params.get('top_k', 5)
            if not isinstance(top_k, int) or top_k < 1:
                self.errors.append("映射器参数错误: rank_based的top_k必须是正整数")
            
            direction = params.get('direction', 1)
            if direction not in [1, -1]:
                self.errors.append("映射器参数错误: rank_based的direction必须是1或-1")
            
            weight_method = params.get('weight_method', 'equal')
            if weight_method not in self.SUPPORTED_WEIGHT_METHODS:
                self.warnings.append(
                    f"映射器参数警告: 未知的weight_method '{weight_method}'，"
                    f"支持的方法: {self.SUPPORTED_WEIGHT_METHODS}"
                )
        
        elif config.type == 'direct':
            target_sum = params.get('target_sum', 1.0)
            if target_sum <= 0:
                self.errors.append("映射器参数错误: direct的target_sum必须是正数")
        
        elif config.type == 'quantile':
            n_quantiles = params.get('n_quantiles', 5)
            if not isinstance(n_quantiles, int) or n_quantiles < 2:
                self.errors.append("映射器参数错误: quantile的n_quantiles必须是大于等于2的整数")
            
            long_quantile = params.get('long_quantile')
            short_quantile = params.get('short_quantile')
            
            if long_quantile is not None and (long_quantile < 0 or long_quantile >= n_quantiles):
                self.errors.append(f"映射器参数错误: long_quantile必须在[0, {n_quantiles-1}]范围内")
            
            if short_quantile is not None and (short_quantile < 0 or short_quantile >= n_quantiles):
                self.errors.append(f"映射器参数错误: short_quantile必须在[0, {n_quantiles-1}]范围内")
    
    def _validate_evaluation_config(self, config: EvaluationConfig):
        """验证评估器配置"""
        if config.forward_period < 1:
            self.errors.append(f"评估配置错误: forward_period必须大于等于1，当前值: {config.forward_period}")
        
        if config.periods_per_year < 1:
            self.errors.append(f"评估配置错误: periods_per_year必须大于等于1，当前值: {config.periods_per_year}")
        
        if not 0 <= config.risk_free_rate <= 1:
            self.warnings.append(f"评估配置警告: risk_free_rate通常应在[0, 1]范围内，当前值: {config.risk_free_rate}")
        
        if not 0 <= config.commission_rate <= 0.1:
            self.warnings.append(f"评估配置警告: commission_rate {config.commission_rate} 看起来不合理")
        
        if config.delay < 0:
            self.errors.append(f"评估配置错误: delay不能为负数，当前值: {config.delay}")
        
        if config.rebalance_freq < 1:
            self.errors.append(f"评估配置错误: rebalance_freq必须大于等于1，当前值: {config.rebalance_freq}")
    
    def _validate_storage_config(self, config: StorageConfig):
        """验证存储配置"""
        storage_mode = getattr(config, 'storage_mode', 'file')
        if storage_mode == 'file':
            if not config.base_path:
                self.errors.append("存储配置错误: 文件存储模式下 base_path 不能为空")
            # 验证至少有一个保存选项为True
            if not any([
                config.save_metrics,
                config.save_config,
                config.save_report,
                config.save_plots,
                config.save_comparison,
            ]):
                self.warnings.append("存储配置警告: 所有保存选项都为False，将不会保存任何结果")
    
    def _validate_consistency(self, config: Config):
        """验证配置各部分之间的一致性"""
        # 验证因子数量与映射器配置是否兼容
        if len(config.factors) > 1 and config.mapper.type == 'direct':
            self.warnings.append(
                "配置一致性警告: 配置了多个因子但使用direct映射器，"
                "direct映射器会直接使用因子值，可能需要先进行因子合成"
            )
        
        # 验证top_k是否合理
        if config.mapper.type == 'rank_based':
            top_k = config.mapper.params.get('top_k', 5)
            # 这里我们无法知道实际的ETF数量，只能给出警告
            if top_k > 50:
                self.warnings.append(f"配置一致性警告: top_k={top_k} 可能过大，请确保不超过ETF总数")
    
    def validate_strict(self, config: Config):
        """
        严格验证配置，如果有错误则抛出异常
        
        Args:
            config: 配置对象
            
        Raises:
            ValidationError: 如果验证失败
        """
        is_valid, errors, warnings = self.validate(config)
        
        if not is_valid:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValidationError(error_msg)
        
        return warnings


def validate_config(config: Config, strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    验证配置的便捷函数
    
    Args:
        config: 配置对象
        strict: 如果为True，验证失败时抛出异常
        
    Returns:
        tuple: (是否通过, 错误列表, 警告列表)
        
    Raises:
        ValidationError: 如果strict=True且验证失败
    """
    validator = ConfigValidator()
    
    if strict:
        warnings = validator.validate_strict(config)
        return True, [], warnings
    else:
        return validator.validate(config)
