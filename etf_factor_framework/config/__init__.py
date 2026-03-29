"""
配置系统模块

提供灵活的配置管理功能，支持YAML和JSON格式。

Usage:
    >>> from config import Config, load_config
    >>> config = load_config('config.yaml')
    >>> print(config.factor.name)
    >>> print(config.mapper.type)
"""

from .base_config import (
    Config,
    DataConfig,
    FactorConfig,
    MapperConfig,
    EvaluationConfig,
    StorageConfig,
)
from .yaml_parser import load_config, load_config_from_string, config_to_dict
from .config_validator import ConfigValidator, ValidationError

# 如果config_manager存在则导出
try:
    from .config_manager import ConfigManager
    __all__ = [
        # 配置类
        'Config',
        'DataConfig',
        'FactorConfig',
        'MapperConfig',
        'EvaluationConfig',
        'StorageConfig',
        # 解析函数
        'load_config',
        'load_config_from_string',
        'config_to_dict',
        # 验证
        'ConfigValidator',
        'ValidationError',
        # 管理工具
        'ConfigManager',
    ]
except ImportError:
    __all__ = [
        # 配置类
        'Config',
        'DataConfig',
        'FactorConfig',
        'MapperConfig',
        'EvaluationConfig',
        'StorageConfig',
        # 解析函数
        'load_config',
        'load_config_from_string',
        'config_to_dict',
        # 验证
        'ConfigValidator',
        'ValidationError',
    ]
