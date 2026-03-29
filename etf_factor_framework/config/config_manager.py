"""
配置管理器

提供大规模配置的搜索、对比、创建和管理功能。
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from .base_config import Config
from .yaml_parser import load_config, save_config


class ConfigManager:
    """
    配置管理器
    
    用于管理大量配置文件，支持搜索、对比、批量操作。
    
    Attributes:
        config_root: 配置根目录
        index: 配置索引
        
    Example:
        >>> manager = ConfigManager("config")
        >>> configs = manager.list_configs(category="rsi")
        >>> results = manager.find_config(factor_type="RSI")
    """
    
    def __init__(self, config_root: Union[str, Path] = "config"):
        """
        初始化配置管理器
        
        Args:
            config_root: 配置根目录路径
        """
        self.config_root = Path(config_root)
        self.index_path = self.config_root / "index.json"
        self.index = self._load_or_create_index()
    
    def _load_or_create_index(self) -> Dict[str, Any]:
        """加载或创建索引文件"""
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建空索引
            index = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "categories": {},
                "tags": {}
            }
            self._save_index(index)
            return index
    
    def _save_index(self, index: Dict[str, Any] = None):
        """保存索引文件"""
        if index is None:
            index = self.index
        index["last_updated"] = datetime.now().isoformat()
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def scan_configs(self) -> List[Dict[str, Any]]:
        """
        扫描所有配置文件
        
        Returns:
            配置信息列表
        """
        configs = []
        for config_file in self.config_root.rglob("*.json"):
            # 跳过索引文件和私有文件
            if config_file.name in ["index.json"] or config_file.name.startswith("_"):
                continue
            
            try:
                config = load_config(config_file)
                rel_path = config_file.relative_to(self.config_root)
                
                # 提取关键信息
                info = {
                    "path": str(rel_path).replace("\\", "/"),
                    "name": config.name if config.name else config_file.stem,
                    "factor_count": len(config.factors),
                    "factor_types": [f.type for f in config.factors],
                    "mapper_type": config.mapper.type if config.mapper else None,
                }
                configs.append(info)
            except Exception as e:
                print(f"警告: 无法加载配置 {config_file}: {e}")
        
        return configs
    
    def list_configs(
        self,
        category: str = None,
        tags: List[str] = None,
        factor_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        列出配置
        
        Args:
            category: 按类别筛选
            tags: 按标签筛选
            factor_type: 按因子类型筛选
            
        Returns:
            配置信息列表
        """
        configs = self.scan_configs()
        
        # 按因子类型筛选
        if factor_type:
            configs = [c for c in configs if factor_type in c.get("factor_types", [])]
        
        # 按类别筛选
        if category and category in self.index.get("categories", {}):
            category_paths = {
                item["path"] for item in self.index["categories"][category].get("configs", [])
            }
            configs = [c for c in configs if c["path"] in category_paths]
        
        # 按标签筛选
        if tags:
            tagged_paths = set()
            for tag in tags:
                tagged_paths.update(self.index.get("tags", {}).get(tag, []))
            configs = [c for c in configs if c["path"] in tagged_paths]
        
        return configs
    
    def find_config(self, **criteria) -> List[Dict[str, Any]]:
        """
        按条件查找配置
        
        支持的条件:
        - name_contains: 名称包含
        - factor_type: 因子类型
        - mapper_type: 映射器类型
        - path_contains: 路径包含
        
        Args:
            **criteria: 查询条件
            
        Returns:
            匹配的配置列表
        """
        configs = self.scan_configs()
        results = []
        
        for config_info in configs:
            match = True
            config_path = self.config_root / config_info["path"]
            
            try:
                config = load_config(config_path)
                
                # 检查名称
                if "name_contains" in criteria:
                    if criteria["name_contains"].lower() not in config.name.lower():
                        match = False
                
                # 检查因子类型
                if "factor_type" in criteria and match:
                    factor_types = [f.type for f in config.factors]
                    if criteria["factor_type"] not in factor_types:
                        match = False
                
                # 检查映射器类型
                if "mapper_type" in criteria and match:
                    if config.mapper.type != criteria["mapper_type"]:
                        match = False
                
                # 检查路径
                if "path_contains" in criteria and match:
                    if criteria["path_contains"] not in config_info["path"]:
                        match = False
                
                if match:
                    results.append({
                        "path": str(config_info["path"]),
                        "config": config
                    })
            
            except Exception as e:
                print(f"警告: 处理配置时出错 {config_info['path']}: {e}")
        
        return results
    
    def compare_configs(self, config_paths: List[str]) -> pd.DataFrame:
        """
        对比多个配置
        
        Args:
            config_paths: 配置路径列表（相对config_root）
            
        Returns:
            对比表格
        """
        comparison = []
        
        for path in config_paths:
            full_path = self.config_root / path
            try:
                config = load_config(full_path)
                
                row = {
                    "配置路径": path,
                    "配置名称": config.name,
                    "因子数量": len(config.factors),
                }
                
                # 添加第一个因子的信息
                if config.factors:
                    factor = config.factors[0]
                    row["因子类型"] = factor.type
                    row["因子参数"] = str(factor.params)
                
                # 添加映射器信息
                if config.mapper:
                    row["映射器类型"] = config.mapper.type
                    row["映射器参数"] = str(config.mapper.params)
                
                # 添加评估信息
                if config.evaluation:
                    row["前瞻期数"] = config.evaluation.forward_period
                    row["手续费率"] = config.evaluation.commission_rate
                
                comparison.append(row)
            
            except Exception as e:
                print(f"警告: 无法加载配置 {path}: {e}")
        
        return pd.DataFrame(comparison)
    
    def create_from_template(
        self,
        template_name: str,
        new_name: str,
        output_dir: str,
        **overrides
    ) -> str:
        """
        基于模板创建新配置
        
        Args:
            template_name: 模板名称（不含路径和扩展名）
            new_name: 新配置名称
            output_dir: 输出目录（相对config_root）
            **overrides: 要覆盖的参数
            
        Returns:
            新配置文件的相对路径
        """
        template_path = self.config_root / "_templates" / f"{template_name}.json"
        
        if not template_path.exists():
            raise FileNotFoundError(f"模板不存在: {template_path}")
        
        # 加载模板
        config = load_config(template_path)
        
        # 应用覆盖
        config.name = new_name
        
        # 处理参数覆盖（简单实现）
        for key, value in overrides.items():
            if "." in key:
                # 支持嵌套参数，如 "factors.0.params.period"
                parts = key.split(".")
                if parts[0] == "factors" and len(parts) >= 3:
                    idx = int(parts[1])
                    if idx < len(config.factors):
                        if parts[2] == "params":
                            config.factors[idx].params[parts[3]] = value
            else:
                # 直接设置属性
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 保存
        output_path = self.config_root / output_dir / f"{new_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, output_path)
        
        return str(output_path.relative_to(self.config_root)).replace("\\", "/")
    
    def add_to_category(self, config_path: str, category: str, description: str = None):
        """
        将配置添加到类别
        
        Args:
            config_path: 配置相对路径
            category: 类别名称
            description: 类别描述（仅在新类别时有效）
        """
        if category not in self.index["categories"]:
            self.index["categories"][category] = {
                "description": description or f"{category}相关配置",
                "configs": []
            }
        
        # 检查是否已存在
        existing = [c for c in self.index["categories"][category]["configs"] if c["path"] == config_path]
        if not existing:
            self.index["categories"][category]["configs"].append({
                "path": config_path,
                "name": Path(config_path).stem
            })
            self._save_index()
    
    def add_tag(self, config_path: str, tag: str):
        """
        为配置添加标签
        
        Args:
            config_path: 配置相对路径
            tag: 标签名称
        """
        if tag not in self.index["tags"]:
            self.index["tags"][tag] = []
        
        if config_path not in self.index["tags"][tag]:
            self.index["tags"][tag].append(config_path)
            self._save_index()
    
    def archive_config(self, config_path: str, reason: str = None):
        """
        归档配置
        
        Args:
            config_path: 配置相对路径
            reason: 归档原因
        """
        src = self.config_root / config_path
        archive_dir = self.config_root / "experiments" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        dst = archive_dir / src.name
        
        # 移动文件
        shutil.move(str(src), str(dst))
        
        # 记录日志
        log_path = archive_dir / "archive_log.json"
        log_entry = {
            "original_path": config_path,
            "archived_path": str(dst.relative_to(self.config_root)).replace("\\", "/"),
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        
        logs = []
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        print(f"配置已归档: {config_path} -> {dst}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取配置统计信息
        
        Returns:
            统计信息字典
        """
        configs = self.scan_configs()
        
        stats = {
            "total_configs": len(configs),
            "by_factor_type": {},
            "by_mapper_type": {},
            "categories": len(self.index.get("categories", {})),
            "tags": len(self.index.get("tags", {}))
        }
        
        for config in configs:
            # 统计因子类型
            for ft in config.get("factor_types", []):
                stats["by_factor_type"][ft] = stats["by_factor_type"].get(ft, 0) + 1
            
            # 统计映射器类型
            mt = config.get("mapper_type")
            if mt:
                stats["by_mapper_type"][mt] = stats["by_mapper_type"].get(mt, 0) + 1
        
        return stats
    
    def print_summary(self):
        """打印配置摘要"""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("配置管理系统摘要")
        print("=" * 60)
        print(f"\n总配置数: {stats['total_configs']}")
        print(f"类别数: {stats['categories']}")
        print(f"标签数: {stats['tags']}")
        
        if stats["by_factor_type"]:
            print("\n按因子类型分布:")
            for ft, count in sorted(stats["by_factor_type"].items(), key=lambda x: -x[1]):
                print(f"  {ft}: {count}")
        
        if stats["by_mapper_type"]:
            print("\n按映射器类型分布:")
            for mt, count in sorted(stats["by_mapper_type"].items(), key=lambda x: -x[1]):
                print(f"  {mt}: {count}")
        
        if self.index.get("categories"):
            print("\n已定义类别:")
            for cat_name, cat_info in self.index["categories"].items():
                config_count = len(cat_info.get("configs", []))
                print(f"  {cat_name}: {config_count}个配置")
