"""
YAML/JSON配置解析器

支持从YAML或JSON文件加载配置。
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import os
import sys

# 尝试导入yaml，如果没有则使用JSON
# 当前项目requirements中没有pyyaml，所以我们主要使用JSON
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .base_config import Config


def load_config(path: Union[str, Path]) -> Config:
    """
    从文件加载配置
    
    支持.json和.yaml/.yml格式。
    
    Args:
        path: 配置文件路径
        
    Returns:
        Config: 配置对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式不支持或解析失败
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    suffix = path.suffix.lower()
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("解析YAML需要安装pyyaml: pip install pyyaml")
        data = yaml.safe_load(content)
    elif suffix == '.json':
        data = json.loads(content)
    else:
        # 尝试自动检测格式
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if HAS_YAML:
                data = yaml.safe_load(content)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
    
    return Config.from_dict(data)


def load_config_from_string(content: str, format: str = 'auto') -> Config:
    """
    从字符串加载配置
    
    Args:
        content: 配置内容字符串
        format: 格式，'auto', 'json', 'yaml'
        
    Returns:
        Config: 配置对象
    """
    if format == 'auto':
        # 尝试JSON解析
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            if HAS_YAML:
                data = yaml.safe_load(content)
            else:
                raise ValueError("无法自动检测格式，请指定format参数或安装pyyaml")
    elif format == 'json':
        data = json.loads(content)
    elif format in ['yaml', 'yml']:
        if not HAS_YAML:
            raise ImportError("解析YAML需要安装pyyaml: pip install pyyaml")
        data = yaml.safe_load(content)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    return Config.from_dict(data)


def config_to_dict(config: Config) -> Dict[str, Any]:
    """
    将配置对象转换为字典
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 配置字典
    """
    return config.to_dict()


def save_config_yaml(config: Config, path: Union[str, Path]):
    """
    保存配置为YAML文件
    
    Args:
        config: 配置对象
        path: 保存路径
    """
    if not HAS_YAML:
        raise ImportError("保存YAML需要安装pyyaml: pip install pyyaml")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_config_json(config: Config, path: Union[str, Path], indent: int = 2):
    """
    保存配置为JSON文件
    
    Args:
        config: 配置对象
        path: 保存路径
        indent: JSON缩进
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config.save_json(str(path), indent=indent)


# 便捷函数，保持与Config类方法的一致性
def save_config(config: Config, path: Union[str, Path], format: str = 'auto'):
    """
    保存配置到文件
    
    Args:
        config: 配置对象
        path: 保存路径
        format: 格式，'auto', 'json', 'yaml'
    """
    path = Path(path)
    
    if format == 'auto':
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = 'json'
    
    if format == 'yaml':
        save_config_yaml(config, path)
    else:
        save_config_json(config, path)
