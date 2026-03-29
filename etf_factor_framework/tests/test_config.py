"""
配置系统单元测试

测试配置类的创建、解析、验证功能。
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import (
    Config,
    DataConfig,
    FactorConfig,
    MapperConfig,
    EvaluationConfig,
    StorageConfig,
    load_config,
    load_config_from_string,
    config_to_dict,
    ConfigValidator,
    ValidationError,
)


class TestDataConfig:
    """测试数据配置"""
    
    def test_default_values(self):
        """测试默认值"""
        config = DataConfig(csv_path="test.csv")
        assert config.csv_path == "test.csv"
        assert config.symbol_col == "symbol"
        assert config.date_col == "eob"
        assert "open" in config.ohlcv_cols
        assert config.min_non_null_ratio == 0.8
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = DataConfig(csv_path="test.csv")
        data = config.to_dict()
        assert data["csv_path"] == "test.csv"
        assert data["symbol_col"] == "symbol"
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "csv_path": "data.csv",
            "symbol_col": "code",
            "date_col": "date"
        }
        config = DataConfig.from_dict(data)
        assert config.csv_path == "data.csv"
        assert config.symbol_col == "code"


class TestFactorConfig:
    """测试因子配置"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        config = FactorConfig(
            name="RSI_10",
            type="RSI",
            params={"period": 10}
        )
        assert config.name == "RSI_10"
        assert config.type == "RSI"
        assert config.params["period"] == 10
    
    def test_default_params(self):
        """测试默认参数"""
        config = FactorConfig(name="Test", type="RSI")
        assert config.params == {}
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "CloseOverMA",
            "type": "CloseOverMA",
            "params": {"ma_period": 20}
        }
        config = FactorConfig.from_dict(data)
        assert config.name == "CloseOverMA"
        assert config.params["ma_period"] == 20


class TestMapperConfig:
    """测试映射器配置"""
    
    def test_default_values(self):
        """测试默认值"""
        config = MapperConfig()
        assert config.type == "rank_based"
        assert config.params == {}
    
    def test_rank_based_params(self):
        """测试rank_based参数转换"""
        config = MapperConfig(
            type="rank_based",
            params={"top_k": 5, "direction": -1}
        )
        params = config.get_mapper_params()
        assert params["top_k"] == 5
        assert params["direction"] == -1
        assert params["weight_method"] == "equal"
    
    def test_direct_params(self):
        """测试direct参数转换"""
        config = MapperConfig(
            type="direct",
            params={"normalize": True, "target_sum": 1.0}
        )
        params = config.get_mapper_params()
        assert params["normalize"] is True
        assert params["target_sum"] == 1.0
    
    def test_quantile_params(self):
        """测试quantile参数转换"""
        config = MapperConfig(
            type="quantile",
            params={"n_quantiles": 5, "long_quantile": 4}
        )
        params = config.get_mapper_params()
        assert params["n_quantiles"] == 5
        assert params["long_quantile"] == 4


class TestEvaluationConfig:
    """测试评估配置"""
    
    def test_default_values(self):
        """测试默认值"""
        config = EvaluationConfig()
        assert config.forward_period == 5
        assert config.periods_per_year == 252
        assert config.risk_free_rate == 0.03
        assert config.commission_rate == 0.0002
        assert config.delay == 1
    
    def test_custom_values(self):
        """测试自定义值"""
        config = EvaluationConfig(
            forward_period=10,
            risk_free_rate=0.05,
            commission_rate=0.001
        )
        assert config.forward_period == 10
        assert config.risk_free_rate == 0.05


class TestStorageConfig:
    """测试存储配置"""
    
    def test_default_values(self):
        """测试默认值"""
        config = StorageConfig()
        assert config.save_metrics is True
        assert config.save_config is True
        assert config.save_report is True
        assert config.save_plots is True
    
    def test_custom_path(self):
        """测试自定义路径"""
        config = StorageConfig(base_path="/custom/path")
        assert config.base_path == "/custom/path"


class TestConfig:
    """测试主配置类"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[FactorConfig(name="RSI", type="RSI")],
            mapper=MapperConfig()
        )
        assert config.name is None
        assert config.version == "1.0"
        assert len(config.factors) == 1
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = Config(
            name="Test",
            data=DataConfig(csv_path="test.csv"),
            factors=[FactorConfig(name="RSI", type="RSI")],
            mapper=MapperConfig()
        )
        data = config.to_dict()
        assert data["name"] == "Test"
        assert "data" in data
        assert "factors" in data
    
    def test_to_json(self):
        """测试转换为JSON"""
        config = Config(
            name="Test",
            data=DataConfig(csv_path="test.csv"),
            factors=[FactorConfig(name="RSI", type="RSI")],
            mapper=MapperConfig()
        )
        json_str = config.to_json()
        assert "Test" in json_str
        assert "RSI" in json_str
        # 验证是有效的JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "Test"
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "TestConfig",
            "version": "2.0",
            "data": {"csv_path": "data.csv"},
            "factors": [
                {"name": "RSI", "type": "RSI", "params": {"period": 10}}
            ],
            "mapper": {"type": "rank_based", "params": {}},
            "evaluation": {},
            "storage": {}
        }
        config = Config.from_dict(data)
        assert config.name == "TestConfig"
        assert config.version == "2.0"
        assert len(config.factors) == 1
    
    def test_create_simple_config(self):
        """测试快速创建简单配置"""
        config = Config.create_simple_config(
            csv_path="data.csv",
            factor_name="RSI",
            factor_type="RSI",
            factor_params={"period": 10}
        )
        assert config.data.csv_path == "data.csv"
        assert config.factors[0].name == "RSI"
        assert config.factors[0].params["period"] == 10
    
    def test_get_factor(self):
        """测试获取因子配置"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[
                FactorConfig(name="RSI_10", type="RSI"),
                FactorConfig(name="RSI_20", type="RSI")
            ],
            mapper=MapperConfig()
        )
        factor = config.get_factor("RSI_10")
        assert factor is not None
        assert factor.name == "RSI_10"
        
        missing = config.get_factor("NOT_EXIST")
        assert missing is None
    
    def test_add_factor(self):
        """测试添加因子"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[FactorConfig(name="RSI", type="RSI")],
            mapper=MapperConfig()
        )
        config.add_factor(FactorConfig(name="MA", type="CloseOverMA"))
        assert len(config.factors) == 2
    
    def test_remove_factor(self):
        """测试移除因子"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[
                FactorConfig(name="RSI", type="RSI"),
                FactorConfig(name="MA", type="CloseOverMA")
            ],
            mapper=MapperConfig()
        )
        result = config.remove_factor("RSI")
        assert result is True
        assert len(config.factors) == 1
        
        result = config.remove_factor("NOT_EXIST")
        assert result is False
    
    def test_save_and_load_json(self):
        """测试保存和加载JSON"""
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "config.json")
            
            original = Config.create_simple_config(
                csv_path="data.csv",
                factor_name="RSI",
                factor_type="RSI",
                factor_params={"period": 10}
            )
            original.name = "TestSaveLoad"
            
            # 保存
            original.save_json(config_path)
            assert os.path.exists(config_path)
            
            # 加载
            loaded = load_config(config_path)
            assert loaded.name == "TestSaveLoad"
            assert loaded.factors[0].name == "RSI"
        finally:
            shutil.rmtree(temp_dir)


class TestLoadConfig:
    """测试配置加载"""
    
    def test_load_json_file(self):
        """测试加载JSON文件"""
        temp_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(temp_dir, "test.json")
            config_data = {
                "name": "Test",
                "data": {"csv_path": "data.csv"},
                "factors": [{"name": "RSI", "type": "RSI", "params": {}}],
                "mapper": {"type": "rank_based", "params": {}}
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            config = load_config(config_path)
            assert config.name == "Test"
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_from_string(self):
        """测试从字符串加载"""
        json_str = json.dumps({
            "name": "StringTest",
            "data": {"csv_path": "data.csv"},
            "factors": [{"name": "RSI", "type": "RSI", "params": {}}],
            "mapper": {"type": "rank_based", "params": {}}
        })
        
        config = load_config_from_string(json_str, format='json')
        assert config.name == "StringTest"
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = Config.create_simple_config(
            csv_path="data.csv",
            factor_name="RSI",
            factor_type="RSI"
        )
        data = config_to_dict(config)
        assert "data" in data
        assert "factors" in data


class TestConfigValidator:
    """测试配置验证器"""
    
    def test_valid_config(self):
        """测试有效配置"""
        config = Config.create_simple_config(
            csv_path="etf_rotation_daily.csv",
            factor_name="RSI",
            factor_type="RSI",
            factor_params={"period": 10}
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_empty_csv_path(self):
        """测试空CSV路径"""
        config = Config(
            data=DataConfig(csv_path=""),
            factors=[FactorConfig(name="RSI", type="RSI")],
            mapper=MapperConfig()
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("csv_path不能为空" in e for e in errors)
    
    def test_no_factors(self):
        """测试没有因子"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[],
            mapper=MapperConfig()
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("至少需要配置一个因子" in e for e in errors)
    
    def test_duplicate_factor_names(self):
        """测试重复因子名称"""
        config = Config(
            data=DataConfig(csv_path="test.csv"),
            factors=[
                FactorConfig(name="RSI", type="RSI"),
                FactorConfig(name="RSI", type="RSI")
            ],
            mapper=MapperConfig()
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("重复" in e for e in errors)
    
    def test_invalid_rsi_period(self):
        """测试无效的RSI周期"""
        config = Config.create_simple_config(
            csv_path="test.csv",
            factor_name="RSI",
            factor_type="RSI",
            factor_params={"period": -1}
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("period必须是正整数" in e for e in errors)
    
    def test_invalid_mapper_top_k(self):
        """测试无效的top_k"""
        config = Config.create_simple_config(
            csv_path="test.csv",
            factor_name="RSI",
            factor_type="RSI",
            mapper_params={"top_k": 0}
        )
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("top_k必须是正整数" in e for e in errors)
    
    def test_invalid_evaluation_params(self):
        """测试无效的评估参数"""
        config = Config.create_simple_config(
            csv_path="test.csv",
            factor_name="RSI",
            factor_type="RSI"
        )
        config.evaluation.forward_period = -1
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate(config)
        assert is_valid is False
        assert any("forward_period必须大于等于1" in e for e in errors)
    
    def test_strict_validation_pass(self):
        """测试严格验证通过"""
        config = Config.create_simple_config(
            csv_path="test.csv",
            factor_name="RSI",
            factor_type="RSI"
        )
        validator = ConfigValidator()
        warnings = validator.validate_strict(config)
        assert isinstance(warnings, list)
    
    def test_strict_validation_fail(self):
        """测试严格验证失败"""
        config = Config.create_simple_config(
            csv_path="",
            factor_name="RSI",
            factor_type="RSI"
        )
        validator = ConfigValidator()
        with pytest.raises(ValidationError):
            validator.validate_strict(config)
    
    def test_validate_config_function(self):
        """测试便捷验证函数"""
        config = Config.create_simple_config(
            csv_path="test.csv",
            factor_name="RSI",
            factor_type="RSI"
        )
        is_valid, errors, warnings = validator_module_validate(config)
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


def validator_module_validate(config):
    """便捷函数导入"""
    from config import validate_config
    return validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
