#!/usr/bin/env python
"""
配置管理命令行工具

Usage:
    python config_manager_cli.py list
    python config_manager_cli.py list --factor-type RSI
    python config_manager_cli.py search rsi
    python config_manager_cli.py compare config/factors/rsi/rsi_period_5.json config/factors/rsi/rsi_period_14.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import pandas as pd
from pathlib import Path

from config import ConfigManager, load_config


@click.group()
@click.option('--config-root', default='config', help='配置根目录')
@click.pass_context
def cli(ctx, config_root):
    """ETF因子框架配置管理工具"""
    ctx.ensure_object(dict)
    ctx.obj['manager'] = ConfigManager(config_root)


@cli.command()
@click.option('--factor-type', '-f', help='按因子类型筛选')
@click.option('--category', '-c', help='按类别筛选')
@click.option('--tag', '-t', multiple=True, help='按标签筛选')
@click.pass_context
def list(ctx, factor_type, category, tag):
    """列出所有配置"""
    manager = ctx.obj['manager']
    
    configs = manager.list_configs(
        factor_type=factor_type,
        category=category,
        tags=list(tag) if tag else None
    )
    
    if not configs:
        print("未找到匹配的配置")
        return
    
    print(f"\n共找到 {len(configs)} 个配置:\n")
    print(f"{'名称':<30} {'因子类型':<20} {'映射器':<15} {'路径'}")
    print("-" * 100)
    
    for cfg in configs:
        factor_types = ", ".join(cfg.get('factor_types', []))[:18]
        print(f"{cfg['name']:<30} {factor_types:<20} {str(cfg.get('mapper_type')):<15} {cfg['path']}")


@cli.command()
@click.argument('keyword')
@click.pass_context
def search(ctx, keyword):
    """搜索配置"""
    manager = ctx.obj['manager']
    
    results = manager.find_config(name_contains=keyword)
    
    if not results:
        print(f"未找到包含 '{keyword}' 的配置")
        return
    
    print(f"\n找到 {len(results)} 个匹配配置:\n")
    for r in results:
        config = r['config']
        print(f"  {r['path']}")
        print(f"    名称: {config.name}")
        if config.factors:
            print(f"    因子: {config.factors[0].type} {config.factors[0].params}")
        print()


@cli.command()
@click.argument('config_paths', nargs=-1)
@click.pass_context
def compare(ctx, config_paths):
    """对比多个配置"""
    if len(config_paths) < 2:
        print("请提供至少2个配置路径进行对比")
        return
    
    manager = ctx.obj['manager']
    df = manager.compare_configs(list(config_paths))
    
    print("\n配置对比:\n")
    print(df.to_string(index=False))


@cli.command()
@click.argument('template')
@click.argument('name')
@click.option('--output-dir', '-o', default='experiments', help='输出目录')
@click.option('--param', '-p', multiple=True, help='参数覆盖，格式: key=value')
@click.pass_context
def create(ctx, template, name, output_dir, param):
    """基于模板创建新配置"""
    manager = ctx.obj['manager']
    
    # 解析参数
    overrides = {}
    for p in param:
        if '=' not in p:
            print(f"警告: 忽略无效参数 {p}")
            continue
        key, value = p.split('=', 1)
        # 尝试转换为数字
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        overrides[key] = value
    
    try:
        new_path = manager.create_from_template(template, name, output_dir, **overrides)
        print(f"配置已创建: {new_path}")
    except Exception as e:
        print(f"创建失败: {e}")


@cli.command()
@click.pass_context
def stats(ctx):
    """显示配置统计信息"""
    manager = ctx.obj['manager']
    manager.print_summary()


@cli.command()
@click.argument('config_path')
@click.argument('category')
@click.option('--description', '-d', help='类别描述')
@click.pass_context
def categorize(ctx, config_path, category, description):
    """将配置添加到类别"""
    manager = ctx.obj['manager']
    manager.add_to_category(config_path, category, description)
    print(f"已添加 {config_path} 到类别 {category}")


@cli.command()
@click.argument('config_path')
@click.argument('tag')
@click.pass_context
def tag(ctx, config_path, tag):
    """为配置添加标签"""
    manager = ctx.obj['manager']
    manager.add_tag(config_path, tag)
    print(f"已为 {config_path} 添加标签 {tag}")


@cli.command()
@click.pass_context
def rebuild_index(ctx):
    """重建配置索引"""
    manager = ctx.obj['manager']
    
    print("正在扫描所有配置...")
    configs = manager.scan_configs()
    
    print(f"找到 {len(configs)} 个配置")
    
    # 自动分类：按因子类型
    for cfg in configs:
        for ft in cfg.get('factor_types', []):
            manager.add_to_category(cfg['path'], ft.lower())
            manager.add_tag(cfg['path'], ft.lower())
    
    # 自动分类：按映射器类型
    for cfg in configs:
        mt = cfg.get('mapper_type')
        if mt:
            manager.add_tag(cfg['path'], mt.lower())
    
    print("索引重建完成")
    manager.print_summary()


@cli.command()
@click.argument('config_path')
@click.option('--reason', '-r', help='归档原因')
@click.pass_context
def archive(ctx, config_path, reason):
    """归档配置"""
    manager = ctx.obj['manager']
    manager.archive_config(config_path, reason)


@cli.command()
@click.argument('config_path')
@click.pass_context
def show(ctx, config_path):
    """显示配置详情"""
    full_path = Path(ctx.obj['manager'].config_root) / config_path
    
    try:
        config = load_config(full_path)
        
        print(f"\n配置详情: {config_path}")
        print("=" * 60)
        print(f"名称: {config.name}")
        print(f"版本: {config.version}")
        if config.description:
            print(f"描述: {config.description}")
        
        print(f"\n数据配置:")
        print(f"  CSV路径: {config.data.csv_path}")
        
        print(f"\n因子配置 ({len(config.factors)}个):")
        for i, f in enumerate(config.factors, 1):
            print(f"  {i}. {f.name} ({f.type})")
            print(f"     参数: {f.params}")
        
        print(f"\n映射器配置:")
        print(f"  类型: {config.mapper.type}")
        print(f"  参数: {config.mapper.params}")
        
        print(f"\n评估配置:")
        print(f"  前瞻期数: {config.evaluation.forward_period}")
        print(f"  年化周期: {config.evaluation.periods_per_year}")
        print(f"  手续费率: {config.evaluation.commission_rate}")
        
    except Exception as e:
        print(f"无法加载配置: {e}")


if __name__ == '__main__':
    cli()
