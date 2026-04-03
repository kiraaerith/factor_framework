"""
SQLite 数据库存储模块

提供基于 SQLite 的因子评估结果存储功能，支持海量因子评估结果的高效存储和查询。
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from contextlib import contextmanager


class DatabaseStorage:
    """
    SQLite 数据库存储类
    
    用于存储和查询因子评估结果，支持海量数据的高效存储。
    
    数据库表结构:
    - factor_evaluation_results: 因子评估结果主表
        - id: 主键
        - expression_name: 表达式名称 (索引)
        - dataset_name: 数据集名称 (索引)
        - expression_params: 表达式超参 (JSON)
        - dataset_params: 数据集超参 (JSON)
        - sharpe: 夏普比率
        - max_drawdown: 最大回撤
        - calmar: Calmar比率
        - ic: IC均值
        - icir: IC IR
        - rank_ic: Rank IC均值
        - rank_icir: Rank IC IR
        - other_metrics: 其他绩效指标 (JSON)
        - created_at: 创建时间
    
    Attributes:
        db_path: 数据库文件路径
    """
    
    _DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "factor_eval_result" / "factor_eval.db"

    def __init__(self, db_path: Union[str, Path] = None):
        """
        初始化数据库存储

        Args:
            db_path: 数据库文件路径，默认使用项目根目录下 factor_eval_result/factor_eval.db
        """
        if db_path is None:
            db_path = self._DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """确保数据库目录存在"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库表结构"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS factor_evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expression_name TEXT NOT NULL,
            factor_type TEXT NOT NULL,         -- 因子类型（不包含参数的通用类型名）
            dataset_name TEXT NOT NULL,
            expression_params TEXT NOT NULL,  -- JSON格式
            dataset_params TEXT NOT NULL,     -- JSON格式
            mapper_config TEXT,               -- JSON格式，仓位映射配置
            evaluation_config TEXT,           -- JSON格式，评估配置
            neutralization_method TEXT DEFAULT 'raw',  -- 中性化方式: raw / industry / size
            top_k INTEGER,                    -- 持仓股数
            rebalance_freq INTEGER,           -- 调仓频率（交易日数）
            forward_period INTEGER,           -- IC 前瞻期（交易日数）
            sharpe REAL,
            max_drawdown REAL,
            calmar REAL,
            ic REAL,
            icir REAL,
            rank_ic REAL,
            rank_icir REAL,
            other_metrics TEXT,               -- JSON格式
            excess_ret_csi300 REAL,          -- 相对沪深300超额年化收益
            excess_ret_csi500 REAL,          -- 相对中证500超额年化收益
            excess_ret_csi2000 REAL,         -- 相对中证2000超额年化收益
            ir_csi300 REAL,                  -- 相对沪深300信息比率
            ir_csi500 REAL,                  -- 相对中证500信息比率
            ir_csi2000 REAL,                 -- 相对中证2000信息比率
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # 对已有 DB 做向后兼容的 ALTER TABLE（每列单独 try/except）
        new_columns = [
            "ALTER TABLE factor_evaluation_results ADD COLUMN neutralization_method TEXT DEFAULT 'raw'",
            "ALTER TABLE factor_evaluation_results ADD COLUMN top_k INTEGER",
            "ALTER TABLE factor_evaluation_results ADD COLUMN rebalance_freq INTEGER",
            "ALTER TABLE factor_evaluation_results ADD COLUMN forward_period INTEGER",
            "ALTER TABLE factor_evaluation_results ADD COLUMN excess_ret_csi300 REAL",
            "ALTER TABLE factor_evaluation_results ADD COLUMN excess_ret_csi500 REAL",
            "ALTER TABLE factor_evaluation_results ADD COLUMN excess_ret_csi2000 REAL",
            "ALTER TABLE factor_evaluation_results ADD COLUMN ir_csi300 REAL",
            "ALTER TABLE factor_evaluation_results ADD COLUMN ir_csi500 REAL",
            "ALTER TABLE factor_evaluation_results ADD COLUMN ir_csi2000 REAL",
        ]

        create_return_curves_sql = """
        CREATE TABLE IF NOT EXISTS factor_return_curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            expression_name TEXT NOT NULL,
            neutralization_method TEXT,
            top_k INTEGER,
            rebalance_freq INTEGER,
            dates TEXT NOT NULL,
            daily_returns TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES factor_evaluation_results(id)
        );
        """

        create_decile_curves_sql = """
        CREATE TABLE IF NOT EXISTS factor_decile_return_curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            expression_name TEXT NOT NULL,
            decile INTEGER NOT NULL,
            dates TEXT NOT NULL,
            daily_returns TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES factor_evaluation_results(id)
        );
        """

        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_expression_name ON factor_evaluation_results(expression_name);",
            "CREATE INDEX IF NOT EXISTS idx_factor_type ON factor_evaluation_results(factor_type);",
            "CREATE INDEX IF NOT EXISTS idx_dataset_name ON factor_evaluation_results(dataset_name);",
            "CREATE INDEX IF NOT EXISTS idx_sharpe ON factor_evaluation_results(sharpe);",
            "CREATE INDEX IF NOT EXISTS idx_calmar ON factor_evaluation_results(calmar);",
            "CREATE INDEX IF NOT EXISTS idx_rank_ic ON factor_evaluation_results(rank_ic);",
            "CREATE INDEX IF NOT EXISTS idx_rank_icir ON factor_evaluation_results(rank_icir);",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON factor_evaluation_results(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_expression_dataset ON factor_evaluation_results(expression_name, dataset_name);",
            "CREATE INDEX IF NOT EXISTS idx_factor_type_dataset ON factor_evaluation_results(factor_type, dataset_name);",
            "CREATE INDEX IF NOT EXISTS idx_neutralization ON factor_evaluation_results(neutralization_method);",
            "CREATE INDEX IF NOT EXISTS idx_top_k ON factor_evaluation_results(top_k);",
            "CREATE INDEX IF NOT EXISTS idx_rebalance_freq ON factor_evaluation_results(rebalance_freq);",
            "CREATE INDEX IF NOT EXISTS idx_decile_result_id ON factor_decile_return_curves(result_id);",
            "CREATE INDEX IF NOT EXISTS idx_decile_expression ON factor_decile_return_curves(expression_name);",
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_return_curves_sql)
            cursor.execute(create_decile_curves_sql)
            # 尝试补列（已存在时 SQLite 会报错，忽略即可）
            for alter_sql in new_columns:
                try:
                    cursor.execute(alter_sql)
                except Exception:
                    pass
            for sql in create_indexes_sql:
                cursor.execute(sql)
            conn.commit()
    
    def _extract_key_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从评估结果中提取关键指标
        
        Args:
            result: 评估结果字典
            
        Returns:
            dict: 提取的关键指标
        """
        metrics = {}
        
        # IC 指标
        ic_metrics = result.get('ic_metrics', {})
        metrics['ic'] = ic_metrics.get('ic_mean')
        metrics['icir'] = ic_metrics.get('ic_ir')
        metrics['rank_ic'] = ic_metrics.get('rank_ic_mean')
        metrics['rank_icir'] = ic_metrics.get('rank_ic_ir')
        
        # 风险调整指标
        risk_adj_metrics = result.get('risk_adjusted_metrics', {})
        metrics['sharpe'] = risk_adj_metrics.get('sharpe_ratio')
        metrics['calmar'] = risk_adj_metrics.get('calmar_ratio')
        
        # 风险指标
        risk_metrics = result.get('risk_metrics', {})
        metrics['max_drawdown'] = risk_metrics.get('max_drawdown')

        # 基准超额收益指标
        benchmark_metrics = result.get('benchmark_metrics', {})
        bm_mapping = {'csi300': 'csi300', 'csi500': 'csi500', 'csi2000': 'csi2000'}
        for bm_key, col_suffix in bm_mapping.items():
            bm = benchmark_metrics.get(bm_key, {})
            metrics[f'excess_ret_{col_suffix}'] = bm.get('excess_annual_return')
            metrics[f'ir_{col_suffix}'] = bm.get('information_ratio')

        return metrics
    
    def _build_other_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建其他绩效指标字典
        
        Args:
            result: 评估结果字典
            
        Returns:
            dict: 其他绩效指标
        """
        other_metrics = {}
        
        # 复制所有指标，排除已单独存储的关键指标
        for key, value in result.items():
            if key in ('factor_name', 'factor_params', 'forward_period',
                       'ic_metrics', 'risk_adjusted_metrics', 'risk_metrics',
                       'daily_returns', 'decile_daily_returns'):
                continue
            other_metrics[key] = value
        
        # 添加关键指标的详细信息
        if 'ic_metrics' in result:
            other_metrics['ic_metrics'] = result['ic_metrics']
        if 'risk_adjusted_metrics' in result:
            other_metrics['risk_adjusted_metrics'] = result['risk_adjusted_metrics']
        if 'risk_metrics' in result:
            other_metrics['risk_metrics'] = result['risk_metrics']
        if 'returns_metrics' in result:
            other_metrics['returns_metrics'] = result['returns_metrics']
        if 'turnover_metrics' in result:
            other_metrics['turnover_metrics'] = result['turnover_metrics']
        
        return other_metrics
    
    def _json_serializer(self, obj) -> Any:
        """处理 numpy 类型和特殊类型的序列化"""
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
    
    def _save_return_curve(
        self,
        conn,
        result_id: int,
        expression_name: str,
        result: Dict[str, Any],
        neutralization_method: Optional[str] = None,
        top_k: Optional[int] = None,
        rebalance_freq: Optional[int] = None,
    ):
        """
        Save daily return curve to factor_return_curves table.

        Extracts 'daily_returns' (pd.Series with DatetimeIndex) from result dict.
        Silently skips if daily_returns is not present.
        """
        daily_returns = result.get('daily_returns')
        if daily_returns is None:
            return
        if isinstance(daily_returns, pd.Series) and len(daily_returns) > 0:
            dates_json = json.dumps(
                [d.strftime('%Y-%m-%d') for d in daily_returns.index],
                ensure_ascii=False,
            )
            returns_json = json.dumps(
                [round(float(v), 8) for v in daily_returns.values],
                ensure_ascii=False,
            )
            conn.execute(
                """INSERT INTO factor_return_curves
                   (result_id, expression_name, neutralization_method, top_k, rebalance_freq,
                    dates, daily_returns)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (result_id, expression_name, neutralization_method or 'raw',
                 top_k, rebalance_freq, dates_json, returns_json),
            )

    def _save_decile_return_curves(
        self,
        conn,
        result_id: int,
        expression_name: str,
        result: Dict[str, Any],
    ):
        """
        Save decile daily return curves to factor_decile_return_curves table.

        Extracts 'decile_daily_returns' (Dict[int, pd.Series]) from result dict.
        Silently skips if not present.
        """
        decile_returns = result.get('decile_daily_returns')
        if not decile_returns:
            return
        # Loop: n_groups 次 (典型 10)
        for decile, series in decile_returns.items():
            if not isinstance(series, pd.Series) or len(series) == 0:
                continue
            dates_json = json.dumps(
                [d.strftime('%Y-%m-%d') for d in series.index],
                ensure_ascii=False,
            )
            returns_json = json.dumps(
                [round(float(v), 8) for v in series.values],
                ensure_ascii=False,
            )
            conn.execute(
                """INSERT INTO factor_decile_return_curves
                   (result_id, expression_name, decile, dates, daily_returns)
                   VALUES (?, ?, ?, ?, ?)""",
                (result_id, expression_name, int(decile), dates_json, returns_json),
            )

    def save_evaluation_result(
        self,
        expression_name: str,
        dataset_name: str,
        result: Dict[str, Any],
        expression_params: Optional[Dict[str, Any]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        factor_type: Optional[str] = None,
        mapper_config: Optional[Dict[str, Any]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        neutralization_method: Optional[str] = None,
        top_k: Optional[int] = None,
        rebalance_freq: Optional[int] = None,
        forward_period: Optional[int] = None,
    ) -> int:
        """
        保存评估结果到数据库
        
        Args:
            expression_name: 表达式名称
            dataset_name: 数据集名称
            result: 评估结果字典
            expression_params: 表达式超参
            dataset_params: 数据集超参
            factor_type: 因子类型（不包含参数的通用类型名）
            mapper_config: 仓位映射配置（JSON格式）
            evaluation_config: 评估配置（JSON格式）
            
        Returns:
            int: 插入记录的ID
        """
        # 提取关键指标
        key_metrics = self._extract_key_metrics(result)
        
        # 构建其他指标
        other_metrics = self._build_other_metrics(result)
        
        # 序列化 JSON 字段
        expression_params_json = json.dumps(
            expression_params or result.get('factor_params', {}), 
            default=self._json_serializer, 
            ensure_ascii=False
        )
        dataset_params_json = json.dumps(
            dataset_params or {}, 
            default=self._json_serializer, 
            ensure_ascii=False
        )
        other_metrics_json = json.dumps(
            other_metrics, 
            default=self._json_serializer, 
            ensure_ascii=False
        )
        mapper_config_json = json.dumps(
            mapper_config or {}, 
            default=self._json_serializer, 
            ensure_ascii=False
        )
        evaluation_config_json = json.dumps(
            evaluation_config or {}, 
            default=self._json_serializer, 
            ensure_ascii=False
        )
        
        # 获取因子类型
        factor_type_value = factor_type or result.get('factor_type', 'unknown')
        
        # 从 dataset_params 自动提取 top_k / rebalance_freq / forward_period（若未显式传入）
        _dp = dataset_params or {}
        top_k_val = top_k if top_k is not None else _dp.get('top_k')
        rebalance_freq_val = rebalance_freq if rebalance_freq is not None else _dp.get('rebalance_freq')
        forward_period_val = forward_period if forward_period is not None else _dp.get('forward_period')
        neutralization_method_val = neutralization_method or 'raw'

        insert_sql = """
        INSERT INTO factor_evaluation_results
        (expression_name, factor_type, dataset_name, expression_params, dataset_params,
         mapper_config, evaluation_config,
         neutralization_method, top_k, rebalance_freq, forward_period,
         sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, other_metrics,
         excess_ret_csi300, excess_ret_csi500, excess_ret_csi2000,
         ir_csi300, ir_csi500, ir_csi2000)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?)
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, (
                expression_name,
                factor_type_value,
                dataset_name,
                expression_params_json,
                dataset_params_json,
                mapper_config_json,
                evaluation_config_json,
                neutralization_method_val,
                top_k_val,
                rebalance_freq_val,
                forward_period_val,
                key_metrics.get('sharpe'),
                key_metrics.get('max_drawdown'),
                key_metrics.get('calmar'),
                key_metrics.get('ic'),
                key_metrics.get('icir'),
                key_metrics.get('rank_ic'),
                key_metrics.get('rank_icir'),
                other_metrics_json,
                key_metrics.get('excess_ret_csi300'),
                key_metrics.get('excess_ret_csi500'),
                key_metrics.get('excess_ret_csi2000'),
                key_metrics.get('ir_csi300'),
                key_metrics.get('ir_csi500'),
                key_metrics.get('ir_csi2000'),
            ))
            record_id = cursor.lastrowid
            # Save daily return curve (same transaction)
            self._save_return_curve(
                conn, record_id, expression_name, result,
                neutralization_method=neutralization_method_val,
                top_k=top_k_val,
                rebalance_freq=rebalance_freq_val,
            )
            # Save decile return curves (same transaction)
            self._save_decile_return_curves(
                conn, record_id, expression_name, result,
            )
            conn.commit()
            return record_id
    
    def query_by_expression_name(
        self, 
        expression_name: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按表达式名称查询
        
        Args:
            expression_name: 表达式名称
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        sql = """
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               mapper_config, evaluation_config, sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        WHERE expression_name = ?
        ORDER BY created_at DESC
        """
        params = [expression_name]
        
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        return df
    
    def query_by_dataset_name(
        self, 
        dataset_name: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按数据集名称查询
        
        Args:
            dataset_name: 数据集名称
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        sql = """
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               mapper_config, evaluation_config, sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        WHERE dataset_name = ?
        ORDER BY created_at DESC
        """
        params = [dataset_name]
        
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        return df
    
    def query_by_expression_and_params(
        self,
        expression_name: str,
        expression_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按表达式名称和超参组合查询
        
        Args:
            expression_name: 表达式名称
            expression_params: 表达式超参（部分匹配）
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        sql = """
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               mapper_config, evaluation_config, sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        WHERE expression_name = ?
        """
        params = [expression_name]
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        # 在 Python 中进行参数匹配（因为 JSON 查询比较复杂）
        if expression_params is not None and not df.empty:
            mask = df['expression_params'].apply(
                lambda x: self._match_params(json.loads(x), expression_params)
            )
            df = df[mask]
        
        if limit is not None:
            df = df.head(limit)
        
        return df.reset_index(drop=True)
    
    def query_by_dataset_and_params(
        self,
        dataset_name: str,
        dataset_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按数据集名称和超参组合查询
        
        Args:
            dataset_name: 数据集名称
            dataset_params: 数据集超参（部分匹配）
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        sql = """
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        WHERE dataset_name = ?
        """
        params = [dataset_name]
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        # 在 Python 中进行参数匹配
        if dataset_params is not None and not df.empty:
            mask = df['dataset_params'].apply(
                lambda x: self._match_params(json.loads(x), dataset_params)
            )
            df = df[mask]
        
        if limit is not None:
            df = df.head(limit)
        
        return df.reset_index(drop=True)
    
    def _match_params(self, stored_params: Dict, query_params: Dict) -> bool:
        """
        检查存储的参数是否匹配查询参数
        
        Args:
            stored_params: 数据库中存储的参数
            query_params: 查询参数
            
        Returns:
            bool: 是否匹配
        """
        for key, value in query_params.items():
            if key not in stored_params:
                return False
            if stored_params[key] != value:
                return False
        return True
    
    def query_by_metric_range(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按指标值范围查询
        
        Args:
            metric_name: 指标名称 ('sharpe', 'max_drawdown', 'calmar', 'ic', 'icir', 'rank_ic', 'rank_icir')
            min_value: 最小值（包含）
            max_value: 最大值（包含）
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        valid_metrics = {'sharpe', 'max_drawdown', 'calmar', 'ic', 'icir', 'rank_ic', 'rank_icir'}
        if metric_name not in valid_metrics:
            raise ValueError(f"metric_name must be one of {valid_metrics}, got {metric_name}")
        
        conditions = []
        params = []
        
        if min_value is not None:
            conditions.append(f"{metric_name} >= ?")
            params.append(min_value)
        
        if max_value is not None:
            conditions.append(f"{metric_name} <= ?")
            params.append(max_value)
        
        sql = f"""
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               mapper_config, evaluation_config, sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        """
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += f" ORDER BY {metric_name} DESC"
        
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        return df
    
    def get_distinct_expressions(self) -> List[str]:
        """
        获取所有不同的表达式名称
        
        Returns:
            list: 表达式名称列表
        """
        sql = "SELECT DISTINCT expression_name FROM factor_evaluation_results ORDER BY expression_name"
    
    def get_distinct_factor_types(self) -> List[str]:
        """
        获取所有不同的因子类型
        
        Returns:
            list: 因子类型列表
        """
        sql = "SELECT DISTINCT factor_type FROM factor_evaluation_results ORDER BY factor_type"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [row[0] for row in cursor.fetchall()]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [row[0] for row in cursor.fetchall()]
    
    def query_by_factor_type(
        self, 
        factor_type: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        按因子类型查询
        
        Args:
            factor_type: 因子类型（不包含参数的通用类型名，如 'RSI'）
            limit: 返回记录数限制
            
        Returns:
            DataFrame: 查询结果
        """
        sql = """
        SELECT id, expression_name, factor_type, dataset_name, expression_params, dataset_params,
               mapper_config, evaluation_config, sharpe, max_drawdown, calmar, ic, icir, rank_ic, rank_icir, 
               other_metrics, created_at
        FROM factor_evaluation_results
        WHERE factor_type = ?
        ORDER BY created_at DESC
        """
        params = [factor_type]
        
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        
        return df
    
    def get_distinct_datasets(self) -> List[str]:
        """
        获取所有不同的数据集名称
        
        Returns:
            list: 数据集名称列表
        """
        sql = "SELECT DISTINCT dataset_name FROM factor_evaluation_results ORDER BY dataset_name"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [row[0] for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            dict: 统计信息
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM factor_evaluation_results")
            total_records = cursor.fetchone()[0]
            
            # 不同表达式数
            cursor.execute("SELECT COUNT(DISTINCT expression_name) FROM factor_evaluation_results")
            distinct_expressions = cursor.fetchone()[0]
            
            # 不同数据集数
            cursor.execute("SELECT COUNT(DISTINCT dataset_name) FROM factor_evaluation_results")
            distinct_datasets = cursor.fetchone()[0]
            
            # 指标统计
            metrics_stats = {}
            for metric in ['sharpe', 'calmar', 'rank_ic', 'rank_icir']:
                cursor.execute(f"""
                    SELECT COUNT(*), AVG({metric}), MAX({metric}), MIN({metric})
                    FROM factor_evaluation_results
                    WHERE {metric} IS NOT NULL
                """)
                row = cursor.fetchone()
                metrics_stats[metric] = {
                    'count': row[0],
                    'avg': row[1],
                    'max': row[2],
                    'min': row[3],
                }
        
        return {
            'total_records': total_records,
            'distinct_expressions': distinct_expressions,
            'distinct_datasets': distinct_datasets,
            'metrics_statistics': metrics_stats,
        }
    
    def delete_by_expression(self, expression_name: str) -> int:
        """
        按表达式名称删除记录
        
        Args:
            expression_name: 表达式名称
            
        Returns:
            int: 删除的记录数
        """
        sql = "DELETE FROM factor_evaluation_results WHERE expression_name = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (expression_name,))
            conn.commit()
            return cursor.rowcount
    
    def delete_by_dataset(self, dataset_name: str) -> int:
        """
        按数据集名称删除记录
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            int: 删除的记录数
        """
        sql = "DELETE FROM factor_evaluation_results WHERE dataset_name = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (dataset_name,))
            conn.commit()
            return cursor.rowcount
    
    def execute_custom_query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        执行自定义 SQL 查询
        
        Args:
            sql: SQL 查询语句
            params: 查询参数
            
        Returns:
            DataFrame: 查询结果
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params or [])
        return df


def create_database_storage(db_path: Optional[str] = None) -> DatabaseStorage:
    """
    创建默认的数据库存储器
    
    Args:
        db_path: 数据库文件路径，默认使用项目根目录下 factor_eval_result/factor_eval.db

    Returns:
        DatabaseStorage: 数据库存储器实例
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent.parent / "factor_eval_result" / "factor_eval.db"
    return DatabaseStorage(db_path)
