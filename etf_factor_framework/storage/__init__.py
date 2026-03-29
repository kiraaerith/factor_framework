"""
结果存储模块

提供系统化的因子评估结果存储功能。

支持两种存储模式：
- 文件存储模式：保存图片和指标结果到文件系统
- 数据库存储模式：只保存指标结果到 SQLite 数据库

两种模式互斥，通过 StorageConfig 的 storage_mode 配置控制。
"""

from .result_storage import ResultStorage, StorageConfig
from .database_storage import DatabaseStorage, create_database_storage

__all__ = [
    'ResultStorage',
    'StorageConfig',
    'DatabaseStorage',
    'create_database_storage',
]
