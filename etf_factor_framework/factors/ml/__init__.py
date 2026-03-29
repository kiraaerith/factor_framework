"""
ML因子模块

基于机器学习的因子实现，目前包含：
- MLCrossSectionalFactor：滚动LightGBM截面因子
"""

from .cross_section_factor import MLCrossSectionalFactor

__all__ = ['MLCrossSectionalFactor']
