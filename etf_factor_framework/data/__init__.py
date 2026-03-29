"""
A股数据加载模块

提供从DuckDB加载A股行情数据的功能。
"""

try:
    from .stock_data_loader import StockDataLoader
    __all__ = ['StockDataLoader']
except ImportError:
    __all__ = []
