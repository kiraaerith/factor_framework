"""
结果存储系统

系统化保存因子评估结果，支持按因子名称/参数/日期组织的目录结构，
以及基于 SQLite 数据库的高效存储。
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import shutil
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from storage.database_storage import DatabaseStorage, create_database_storage


class StorageConfig:
    """
    存储配置类
    
    定义结果存储的默认配置参数，支持文件存储和数据库两种模式。
    
    两种存储模式互斥：
    - 'file': 文件存储模式（默认），保存图片和指标结果到文件系统
    - 'database': 数据库存储模式，只保存指标结果到 SQLite
    """
    
    # 默认基础路径（文件模式）
    DEFAULT_BASE_PATH = r"E:\code_project\factor_eval_result\etf"
    
    # 默认数据库路径（数据库模式）
    DEFAULT_DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"
    
    # 文件命名模板
    METRICS_FILENAME = "metrics.json"
    CONFIG_FILENAME = "config.json"
    REPORT_FILENAME = "report.txt"
    COMPARISON_FILENAME = "comparison.csv"
    
    # 图表文件名
    CUMULATIVE_IC_FILENAME = "cumulative_ic.png"
    IC_DISTRIBUTION_FILENAME = "ic_distribution.png"
    FACTOR_DISTRIBUTION_FILENAME = "factor_distribution.png"
    RETURNS_DRAWDOWN_FILENAME = "returns_drawdown.png"
    RETURNS_DISTRIBUTION_FILENAME = "returns_distribution.png"
    ROLLING_IC_FILENAME = "rolling_ic.png"
    EVALUATION_REPORT_FILENAME = "evaluation_report.png"
    
    def __init__(
        self,
        storage_mode: str = 'file',
        base_path: Optional[str] = None,
        db_path: Optional[str] = None,
        save_metrics: bool = True,
        save_config: bool = True,
        save_report: bool = True,
        save_plots: bool = True,
        save_comparison: bool = True,
    ):
        """
        初始化存储配置
        
        Args:
            storage_mode: 存储模式 ('file' 或 'database')，默认 'file'
            base_path: 基础存储路径（文件模式使用），默认使用 DEFAULT_BASE_PATH
            db_path: 数据库文件路径（数据库模式使用），默认使用 DEFAULT_DB_PATH
            save_metrics: 是否保存指标JSON（文件模式有效）
            save_config: 是否保存配置JSON（文件模式有效）
            save_report: 是否保存文本报告（文件模式有效）
            save_plots: 是否保存图表（文件模式有效）
            save_comparison: 是否保存对比表（文件模式有效）
        """
        if storage_mode not in ('file', 'database'):
            raise ValueError(f"storage_mode must be 'file' or 'database', got {storage_mode}")
        
        self.storage_mode = storage_mode
        self.base_path = Path(base_path) if base_path else Path(self.DEFAULT_BASE_PATH)
        self.db_path = Path(db_path) if db_path else Path(self.DEFAULT_DB_PATH)
        self.save_metrics = save_metrics
        self.save_config = save_config
        self.save_report = save_report
        self.save_plots = save_plots
        self.save_comparison = save_comparison
    
    def is_file_mode(self) -> bool:
        """是否为文件存储模式"""
        return self.storage_mode == 'file'
    
    def is_database_mode(self) -> bool:
        """是否为数据库存储模式"""
        return self.storage_mode == 'database'


class ResultStorage:
    """
    结果存储类
    
    系统化保存因子评估结果，支持按因子名称/参数组织的目录结构（文件模式），
    以及基于 SQLite 数据库的高效存储（数据库模式）。
    
    两种存储模式互斥，通过 config.storage_mode 控制：
    - 'file': 文件存储模式（默认），保存到目录结构
    - 'database': 数据库存储模式，保存到 SQLite
    
    目录结构示例（文件模式）:
    ```
    E:/code_project/factor_eval_result/etf/
    ├── RSI/
    │   └── period=5_topk=5/
    │       ├── metrics.json          # 指标数值
    │       ├── config.json           # 配置信息
    │       ├── report.txt            # 文本报告
    │       ├── cumulative_ic.png     # 累积IC曲线
    │       └── ...
    ```
    
    数据库表结构（数据库模式）:
    - factor_evaluation_results: 因子评估结果主表
        - expression_name: 表达式名称（索引）
        - dataset_name: 数据集名称（索引）
        - expression_params: 表达式超参（JSON）
        - dataset_params: 数据集超参（JSON）
        - sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir
        - other_metrics: 其他绩效指标（JSON）
    
    Attributes:
        config: 存储配置
        base_path: 基础存储路径（文件模式）
        db_storage: 数据库存储器（数据库模式）
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        初始化结果存储器
        
        Args:
            config: 存储配置，如果为None则使用默认配置（文件模式）
        """
        self.config = config or StorageConfig()
        
        # 检查配置模式
        self._is_database_mode = self.config.is_database_mode() if hasattr(self.config, 'is_database_mode') else False
        
        if self._is_database_mode:
            # 数据库模式
            db_path = getattr(self.config, 'db_path', r"E:\code_project\factor_eval_result\factor_eval.db")
            self.db_storage = create_database_storage(db_path)
            self.base_path = None
        else:
            # 文件模式
            self.db_storage = None
            self.base_path = Path(self.config.base_path)
            self._ensure_base_path()
        
        # 存储记录，用于批量保存后生成对比表
        self._storage_records: List[Dict[str, Any]] = []
    
    def _ensure_base_path(self):
        """确保基础路径存在"""
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _build_factor_path(
        self,
        factor_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        构建因子存储路径
        
        Args:
            factor_name: 因子名称
            params: 因子参数字典
            
        Returns:
            Path: 因子存储目录路径
        """
        # 清理因子名称（去除非法字符）
        safe_factor_name = self._sanitize_filename(factor_name)
        
        # 构建参数子目录名
        if params:
            param_parts = []
            for key, value in sorted(params.items()):
                safe_key = self._sanitize_filename(str(key))
                safe_value = self._sanitize_filename(str(value))
                param_parts.append(f"{safe_key}={safe_value}")
            param_dir = "_".join(param_parts) if param_parts else "default"
        else:
            param_dir = "default"
        
        factor_path = self.base_path / safe_factor_name / param_dir
        factor_path.mkdir(parents=True, exist_ok=True)
        
        return factor_path
    
    def _sanitize_filename(self, name: str) -> str:
        """
        清理文件名，移除非法字符
        
        Args:
            name: 原始文件名
            
        Returns:
            str: 清理后的文件名
        """
        # 替换Windows文件系统中的非法字符
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    def save_evaluation_result(
        self,
        result: Dict[str, Any],
        figures: Optional[Dict[str, Any]] = None,
        report: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        factor_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        factor_type: Optional[str] = None,
        mapper_config: Optional[Dict[str, Any]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        neutralization_method: Optional[str] = None,
        top_k: Optional[int] = None,
        rebalance_freq: Optional[int] = None,
        forward_period: Optional[int] = None,
    ) -> Union[Dict[str, Path], int]:
        """
        保存单个因子的评估结果
        
        根据配置选择文件存储或数据库存储模式。两种模式互斥。
        
        Args:
            result: 评估结果字典（来自run_full_evaluation）
            figures: 图表对象字典（来自plot），仅在文件模式使用
            report: 文本报告（来自generate_report），仅在文件模式使用
            config: 额外配置信息，仅在文件模式使用
            factor_name: 因子/表达式名称（如果不提供则从result中提取）
            params: 表达式参数（如果不提供则从result中提取）
            dataset_name: 数据集名称（数据库模式使用，默认为'unknown_dataset'）
            dataset_params: 数据集参数（数据库模式使用）
            factor_type: 因子类型（如果不提供则从result中提取）
            mapper_config: 仓位映射配置（数据库模式使用）
            evaluation_config: 评估配置（数据库模式使用）
            
        Returns:
            Dict[str, Path]: 保存的文件路径字典（文件模式）
            int: 插入记录的ID（数据库模式）
        """
        # 提取因子名称、类型和参数
        if factor_name is None:
            factor_name = result.get('factor_name', 'unknown_factor')
        if factor_type is None:
            factor_type = result.get('factor_type', factor_name)  # 默认为factor_name
        if params is None:
            params = result.get('factor_params', {})
        
        if self._is_database_mode:
            # 数据库模式：只保存指标结果，不保存图片
            record_id = self.db_storage.save_evaluation_result(
                expression_name=factor_name,
                dataset_name=dataset_name or 'unknown_dataset',
                result=result,
                expression_params=params,
                dataset_params=dataset_params,
                factor_type=factor_type,
                mapper_config=mapper_config,
                evaluation_config=evaluation_config,
                neutralization_method=neutralization_method,
                top_k=top_k,
                rebalance_freq=rebalance_freq,
                forward_period=forward_period,
            )
            
            # 记录存储信息
            self._storage_records.append({
                'factor_name': factor_name,
                'dataset_name': dataset_name or 'unknown_dataset',
                'record_id': record_id,
                'timestamp': datetime.now().isoformat(),
            })
            
            return record_id
        else:
            # 文件模式：保存图片和指标结果到文件
            factor_path = self._build_factor_path(factor_name, params)
            
            saved_files = {}
            
            # 1. 保存指标JSON
            if self.config.save_metrics:
                metrics_path = factor_path / self.config.METRICS_FILENAME
                self._save_metrics_json(result, metrics_path)
                saved_files['metrics'] = metrics_path
            
            # 2. 保存配置JSON
            if self.config.save_config:
                config_path = factor_path / self.config.CONFIG_FILENAME
                self._save_config_json(result, config, config_path)
                saved_files['config'] = config_path
            
            # 3. 保存文本报告
            if self.config.save_report and report:
                report_path = factor_path / self.config.REPORT_FILENAME
                self._save_text_report(report, report_path)
                saved_files['report'] = report_path
            
            # 4. 保存图表
            if self.config.save_plots and figures:
                plot_paths = self._save_plots(figures, factor_path)
                saved_files.update(plot_paths)
            
            # 记录存储信息
            self._storage_records.append({
                'factor_name': factor_name,
                'params': params,
                'path': factor_path,
                'saved_files': saved_files,
                'timestamp': datetime.now().isoformat(),
            })
            
            return saved_files
    
    def _save_metrics_json(self, result: Dict[str, Any], path: Path):
        """保存指标为JSON文件"""
        # 处理numpy类型和特殊类型的序列化
        def json_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        # 添加元数据
        output = {
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
            },
            'result': result
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=json_serializer, ensure_ascii=False)
    
    def _save_config_json(
        self,
        result: Dict[str, Any],
        extra_config: Optional[Dict[str, Any]],
        path: Path
    ):
        """保存配置为JSON文件"""
        config = {
            'factor_name': result.get('factor_name', 'unknown'),
            'factor_params': result.get('factor_params', {}),
            'forward_period': result.get('forward_period', 1),
            'saved_at': datetime.now().isoformat(),
        }
        
        if extra_config:
            config['extra_config'] = extra_config
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str, ensure_ascii=False)
    
    def _save_text_report(self, report: str, path: Path):
        """保存文本报告"""
        # 添加时间戳头
        header = f"""{'='*60}
因子评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + report)
    
    def _save_plots(self, figures: Dict[str, Any], factor_path: Path) -> Dict[str, Path]:
        """保存图表"""
        saved = {}
        
        # 图表文件名映射
        filename_map = {
            'cumulative_ic': self.config.CUMULATIVE_IC_FILENAME,
            'ic_distribution': self.config.IC_DISTRIBUTION_FILENAME,
            'factor_distribution': self.config.FACTOR_DISTRIBUTION_FILENAME,
            'returns_drawdown': self.config.RETURNS_DRAWDOWN_FILENAME,
            'returns_distribution': self.config.RETURNS_DISTRIBUTION_FILENAME,
            'rolling_ic': self.config.ROLLING_IC_FILENAME,
            'evaluation_report': self.config.EVALUATION_REPORT_FILENAME,
        }
        
        for key, figure in figures.items():
            if figure is None:
                continue
                
            if key in filename_map:
                filepath = factor_path / filename_map[key]
                try:
                    # 如果figure有savefig方法（matplotlib）
                    if hasattr(figure, 'savefig'):
                        figure.savefig(filepath, dpi=150, bbox_inches='tight')
                        saved[key] = filepath
                    # 如果figure是文件路径
                    elif isinstance(figure, (str, Path)):
                        shutil.copy(figure, filepath)
                        saved[key] = filepath
                except Exception as e:
                    print(f"Warning: Failed to save plot '{key}': {e}")
        
        return saved
    
    def save_comparison_table(
        self,
        comparison_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        保存多因子对比表
        
        Args:
            comparison_df: 对比表DataFrame
            filename: 文件名（默认使用config中的COMPARISON_FILENAME）
            
        Returns:
            Path: 保存的文件路径
        """
        if filename is None:
            filename = self.config.COMPARISON_FILENAME
        
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = filename.rsplit('.', 1)
        filename_with_timestamp = f"{name}_{timestamp}.{ext}"
        
        path = self.base_path / filename_with_timestamp
        comparison_df.to_csv(path, index=False, encoding='utf-8-sig')
        
        return path
    
    def get_storage_summary(self) -> pd.DataFrame:
        """
        获取存储摘要
        
        Returns:
            DataFrame: 存储记录摘要
        """
        if not self._storage_records:
            return pd.DataFrame()
        
        rows = []
        for record in self._storage_records:
            row = {
                'factor_name': record['factor_name'],
                'params': str(record['params']),
                'path': str(record['path']),
                'timestamp': record['timestamp'],
                'n_files': len(record['saved_files']),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def clear_storage_records(self):
        """清空存储记录"""
        self._storage_records = []
    
    def list_saved_factors(self) -> List[Dict[str, Any]]:
        """
        列出所有已保存的因子
        
        Returns:
            List[Dict]: 已保存因子的信息列表
        """
        factors = []
        
        if not self.base_path.exists():
            return factors
        
        for factor_dir in self.base_path.iterdir():
            if factor_dir.is_dir():
                factor_name = factor_dir.name
                param_dirs = []
                
                for param_dir in factor_dir.iterdir():
                    if param_dir.is_dir():
                        # 检查是否有metrics.json
                        has_metrics = (param_dir / self.config.METRICS_FILENAME).exists()
                        param_dirs.append({
                            'param_dir': param_dir.name,
                            'has_metrics': has_metrics,
                            'path': str(param_dir),
                        })
                
                factors.append({
                    'factor_name': factor_name,
                    'param_variants': param_dirs,
                    'n_variants': len(param_dirs),
                })
        
        return factors
    
    def load_metrics(self, factor_name: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        加载已保存的指标
        
        Args:
            factor_name: 因子名称
            params: 因子参数
            
        Returns:
            Dict: 指标字典，如果不存在则返回None
        """
        factor_path = self._build_factor_path(factor_name, params)
        metrics_path = factor_path / self.config.METRICS_FILENAME
        
        if not metrics_path.exists():
            return None
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('result')
    
    def delete_factor(self, factor_name: str, params: Optional[Dict[str, Any]] = None):
        """
        删除因子存储
        
        Args:
            factor_name: 因子名称
            params: 因子参数，如果为None则删除该因子的所有参数变体
        """
        if params is None:
            # 删除整个因子目录
            factor_dir = self.base_path / self._sanitize_filename(factor_name)
            if factor_dir.exists():
                shutil.rmtree(factor_dir)
        else:
            # 删除特定参数变体
            factor_path = self._build_factor_path(factor_name, params)
            if factor_path.exists():
                shutil.rmtree(factor_path)


def create_default_storage(base_path: Optional[str] = None) -> ResultStorage:
    """
    创建默认的结果存储器
    
    Args:
        base_path: 基础存储路径，默认使用 E:\code_project\factor_eval_result\etf
        
    Returns:
        ResultStorage: 结果存储器实例
    """
    config = StorageConfig(base_path=base_path)
    return ResultStorage(config)


# 便捷函数
def save_evaluation_result(
    result: Dict[str, Any],
    figures: Optional[Dict[str, Any]] = None,
    report: Optional[str] = None,
    base_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Path]:
    """
    便捷函数：保存评估结果
    
    Args:
        result: 评估结果字典
        figures: 图表对象字典
        report: 文本报告
        base_path: 基础存储路径
        **kwargs: 其他参数传递给save_evaluation_result
        
    Returns:
        Dict[str, Path]: 保存的文件路径字典
    """
    storage = create_default_storage(base_path)
    return storage.save_evaluation_result(
        result=result,
        figures=figures,
        report=report,
        **kwargs
    )
