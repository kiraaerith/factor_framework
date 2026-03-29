# Factor Framework

## 背景

本目录是 `etf_factor_framework` 的独立部署位置。

`etf_factor_framework` 原本位于 `etf_cross_ml-master` 项目内部，是一个基于横截面数据结构的因子计算与评估框架。虽然名称中带有"ETF"，但它已经支持 A 股股票的因子测评。为了将因子框架从 `etf_cross_ml-master` 中解耦、便于独立维护和复用，将其迁移备份到此目录。


## 备份与恢复

迁移脚本位于本目录下：`move_factor_framework.py`

脚本使用相对路径（基于 `__file__` 推算），不依赖具体的机器或目录名称。


## 目录结构要求

迁移脚本通过自身位置（`__file__`）相对推算路径，因此必须满足以下结构条件：

```
{任意根目录}/
├── etf_cross_ml-master/
│   └── etf_factor_framework/
└── factor_framework/                       # 迁移目标在这里（与 etf_cross_ml-master 同级）
    ├── move_factor_framework.py            # 脚本在这里
    └── etf_factor_framework/
```

关键约束：
- `factor_framework` 必须与 `etf_cross_ml-master` 处于**同一父目录**下
- 脚本必须位于 `factor_framework/` 下，不能移动到其他层级
- 父目录名称不限（不要求叫 `code_project_v2`），但上述相对层级关系不能变


### 迁移备份（将源目录复制到此处）

在任意位置执行（路径可用绝对或相对路径指向脚本）：

```bash
python move_factor_framework.py move
```

或从父目录执行：

```bash
python factor_framework/move_factor_framework.py move
```

执行后会：
1. 将 `etf_cross_ml-master/etf_factor_framework` 完整复制到本目录下的 `etf_factor_framework/`
2. 在父目录生成 `factor_framework_move_backup.json` 元数据文件，记录迁移时间、路径等信息

源目录不会被删除。

### 恢复备份（从此处恢复到源位置）

```bash
python move_factor_framework.py restore
```

或从父目录执行：

```bash
python factor_framework/move_factor_framework.py restore
```

执行后会：
1. 根据元数据文件将本目录的内容复制回源位置
2. 询问是否删除本目录及清理元数据文件

所有删除操作均需手动输入 `Yes` 确认。

