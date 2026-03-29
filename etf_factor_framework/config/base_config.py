"""
基础配置类

定义配置系统的核心数据结构。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, ClassVar
from pathlib import Path
import json


@dataclass
class DataConfig:
    """
    数据加载配置

    支持两种数据源（通过 data_source 字段切换）：
      - 'csv'    : 原有ETF CSV加载方式（默认，向后兼容）
      - 'duckdb' : A股DuckDB数据库加载方式

    Attributes:
        csv_path: CSV文件路径（csv模式使用）
        symbol_col: 标的列名
        date_col: 日期列名
        ohlcv_cols: OHLCV列名映射
        start_date: 开始日期
        end_date: 结束日期
        min_non_null_ratio: 最小非空比例（数据清洗）
        data_source: 数据源类型，'csv' 或 'duckdb'
        db_path: DuckDB数据库路径（duckdb模式使用）
        use_adjusted: 是否使用后复权数据（duckdb模式）
        lookback_extra_days: 为因子计算预留的额外回望日历天数
        new_stock_filter_days: 新股过滤天数（0=不过滤）
        board_filter: 板块代码过滤列表，如 [10100101, 10100102]
    """
    # CSV模式字段（向后兼容：保持可选）
    csv_path: Optional[str] = None
    symbol_col: str = 'symbol'
    date_col: str = 'eob'
    ohlcv_cols: Dict[str, str] = field(default_factory=lambda: {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_non_null_ratio: float = 0.8
    # DuckDB / A股扩展字段
    data_source: str = 'csv'
    db_path: str = r'E:\code_project_v2\china_stock_data\china_stock.duckdb'
    use_adjusted: bool = True
    lookback_extra_days: int = 120
    new_stock_filter_days: int = 365
    board_filter: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """从字典创建（兼容旧配置：csv_path 在顶层字典中）"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FactorConfig:
    """
    因子计算配置
    
    Attributes:
        name: 因子名称
        type: 因子类型（如 'RSI', 'CloseOverMA', 'Momentum' 等）
        params: 因子参数
    """
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorConfig':
        """从字典创建"""
        return cls(
            name=data.get('name', 'Unknown'),
            type=data.get('type', 'Unknown'),
            params=data.get('params', {})
        )


@dataclass
class MapperConfig:
    """
    仓位映射器配置
    
    Attributes:
        type: 映射器类型（'rank_based', 'direct', 'quantile', 'zscore'）
        params: 映射器参数
    """
    type: str = 'rank_based'
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MapperConfig':
        """从字典创建"""
        return cls(
            type=data.get('type', 'rank_based'),
            params=data.get('params', {})
        )
    
    def get_mapper_params(self) -> Dict[str, Any]:
        """
        获取标准化的映射器参数
        
        根据type转换参数名称以匹配具体的映射器类。
        """
        params = self.params.copy()
        
        if self.type == 'rank_based':
            # RankBasedMapper参数映射
            return {
                'top_k': params.get('top_k', 5),
                'direction': params.get('direction', 1),
                'weight_method': params.get('weight_method', 'equal'),
                'temperature': params.get('temperature', 1.0),
            }
        elif self.type == 'direct':
            # DirMapper参数映射
            return {
                'normalize': params.get('normalize', True),
                'target_sum': params.get('target_sum', 1.0),
                'clip_range': params.get('clip_range'),
                'fill_na': params.get('fill_na', 0.0),
            }
        elif self.type == 'quantile':
            # QuantileMapper参数映射
            return {
                'n_quantiles': params.get('n_quantiles', 5),
                'long_quantile': params.get('long_quantile'),
                'short_quantile': params.get('short_quantile'),
                'equal_weight_within_group': params.get('equal_weight_within_group', True),
            }
        elif self.type == 'zscore':
            # ZScoreMapper参数映射
            return {
                'threshold': params.get('threshold'),
                'normalize': params.get('normalize', True),
            }
        else:
            return params


@dataclass
class EvaluationConfig:
    """
    评估器配置

    Attributes:
        forward_period: 前瞻期数（IC计算用）
        periods_per_year: 每年的周期数
        risk_free_rate: 无风险利率
        commission_rate: 通用手续费率（买卖双边对称，若设置了 buy/sell_commission_rate 则被覆盖）
        slippage_rate: 滑点率（买卖双边对称）
        delay: 调仓延迟（T日信号在 T+delay 日执行）
        rebalance_freq: 调仓频率（每N根K线调仓一次，默认1=每日调仓）
        metrics_to_calculate: 需要计算的指标列表
        execution_price: 执行价格类型，'close'（默认，兼容ETF）或 'open'（A股T+1开盘执行）
        buy_commission_rate: 买入佣金率（None=使用 commission_rate）
        sell_commission_rate: 卖出佣金率（None=使用 commission_rate）
        stamp_tax_rate: 印花税率（A股卖出时收取，默认 0.0005）
        suspended_value_mode: 停牌价值处理，'freeze' 或 'zero'
    """
    forward_period: int = 5
    periods_per_year: int = 252
    risk_free_rate: float = 0.03
    commission_rate: float = 0.0002
    slippage_rate: float = 0.0
    delay: int = 1
    rebalance_freq: int = 1
    metrics_to_calculate: List[str] = field(default_factory=lambda: [
        'ic', 'returns', 'risk', 'risk_adjusted', 'turnover'
    ])
    # A股扩展字段
    execution_price: str = 'close'
    buy_commission_rate: Optional[float] = None
    sell_commission_rate: Optional[float] = None
    stamp_tax_rate: float = 0.0
    suspended_value_mode: str = 'freeze'

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StorageConfig:
    """
    存储配置
    
    支持两种存储模式：
    - 'file': 文件存储模式（默认），保存图片和指标结果到文件系统
    - 'database': 数据库存储模式，只保存指标结果到 SQLite
    
    两种模式互斥，通过 storage_mode 配置控制。
    
    Attributes:
        storage_mode: 存储模式 ('file' 或 'database')
        base_path: 基础存储路径（文件模式使用）
        db_path: 数据库文件路径（数据库模式使用）
        save_metrics: 是否保存指标JSON（文件模式有效）
        save_config: 是否保存配置JSON（文件模式有效）
        save_report: 是否保存文本报告（文件模式有效）
        save_plots: 是否保存图表（文件模式有效）
        save_comparison: 是否保存对比表（文件模式有效）
    """
    
    # 类属性：文件命名模板（文件模式使用）
    METRICS_FILENAME: ClassVar[str] = "metrics.json"
    CONFIG_FILENAME: ClassVar[str] = "config.json"
    REPORT_FILENAME: ClassVar[str] = "report.txt"
    COMPARISON_FILENAME: ClassVar[str] = "comparison.csv"
    
    # 类属性：图表文件名（文件模式使用）
    CUMULATIVE_IC_FILENAME: ClassVar[str] = "cumulative_ic.png"
    IC_DISTRIBUTION_FILENAME: ClassVar[str] = "ic_distribution.png"
    FACTOR_DISTRIBUTION_FILENAME: ClassVar[str] = "factor_distribution.png"
    RETURNS_DRAWDOWN_FILENAME: ClassVar[str] = "returns_drawdown.png"
    RETURNS_DISTRIBUTION_FILENAME: ClassVar[str] = "returns_distribution.png"
    ROLLING_IC_FILENAME: ClassVar[str] = "rolling_ic.png"
    EVALUATION_REPORT_FILENAME: ClassVar[str] = "evaluation_report.png"
    
    # 实例属性
    storage_mode: str = 'file'  # 'file' 或 'database'
    base_path: str = r"E:\code_project\factor_eval_result\etf"
    db_path: str = r"E:\code_project\factor_eval_result\factor_eval.db"
    save_metrics: bool = True
    save_config: bool = True
    save_report: bool = True
    save_plots: bool = True
    save_comparison: bool = True
    
    def __post_init__(self):
        """验证配置"""
        if self.storage_mode not in ('file', 'database'):
            raise ValueError(f"storage_mode must be 'file' or 'database', got {self.storage_mode}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def is_file_mode(self) -> bool:
        """是否为文件存储模式"""
        return self.storage_mode == 'file'
    
    def is_database_mode(self) -> bool:
        """是否为数据库存储模式"""
        return self.storage_mode == 'database'


@dataclass
class Config:
    """
    主配置类
    
    整合所有子配置，提供统一的配置管理接口。
    
    Attributes:
        data: 数据配置
        factors: 因子配置列表（支持多因子）
        mapper: 仓位映射器配置
        evaluation: 评估器配置
        storage: 存储配置
        name: 配置名称（可选）
        version: 配置版本
        description: 配置描述
    """
    data: DataConfig
    factors: List[FactorConfig]
    mapper: MapperConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    name: Optional[str] = None
    version: str = '1.0'
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'data': self.data.to_dict(),
            'factors': [f.to_dict() for f in self.factors],
            'mapper': self.mapper.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'storage': self.storage.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, path: str):
        """保存为JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(
            name=data.get('name'),
            version=data.get('version', '1.0'),
            description=data.get('description'),
            data=DataConfig.from_dict(data['data']),
            factors=[FactorConfig.from_dict(f) for f in data.get('factors', [])],
            mapper=MapperConfig.from_dict(data.get('mapper', {})),
            evaluation=EvaluationConfig.from_dict(data.get('evaluation', {})),
            storage=StorageConfig.from_dict(data.get('storage', {})),
        )
    
    @classmethod
    def create_simple_config(
        cls,
        csv_path: str,
        factor_name: str,
        factor_type: str,
        factor_params: Optional[Dict[str, Any]] = None,
        mapper_type: str = 'rank_based',
        mapper_params: Optional[Dict[str, Any]] = None,
    ) -> 'Config':
        """
        快速创建简单配置
        
        Args:
            csv_path: CSV数据文件路径
            factor_name: 因子名称
            factor_type: 因子类型
            factor_params: 因子参数
            mapper_type: 映射器类型
            mapper_params: 映射器参数
            
        Returns:
            Config: 配置对象
        """
        return cls(
            name=f"{factor_name}_config",
            data=DataConfig(csv_path=csv_path),
            factors=[
                FactorConfig(
                    name=factor_name,
                    type=factor_type,
                    params=factor_params or {}
                )
            ],
            mapper=MapperConfig(
                type=mapper_type,
                params=mapper_params or {}
            ),
        )
    
    def get_factor(self, name: str) -> Optional[FactorConfig]:
        """根据名称获取因子配置"""
        for factor in self.factors:
            if factor.name == name:
                return factor
        return None
    
    def add_factor(self, factor: FactorConfig):
        """添加因子配置"""
        self.factors.append(factor)
    
    def remove_factor(self, name: str) -> bool:
        """根据名称移除因子配置"""
        for i, factor in enumerate(self.factors):
            if factor.name == name:
                del self.factors[i]
                return True
        return False
