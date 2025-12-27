# Optuna 超参数搜索配置说明

## 统一配置文件格式

所有配置都在一个 JSON 文件中，包含：
- `search_space`: 定义需要优化的超参数搜索空间
- 其他字段: 固定参数（不参与优化）

### 配置文件结构

```json
{
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "low": 1e-5,
      "high": 5e-4
    },
    "num_train_epochs": {
      "type": "int",
      "low": 2,
      "high": 5
    }
  },
  "model_name": "/tmp2/share_data/google--gemma-3-12b-it",
  "input_data": "data/input/ocr_non_test_data=500.json",
  "max_seq_length": 8192,
  "lora_r": 16,
  "lora_alpha": 16
}
```

## 搜索空间参数类型

### 1. `loguniform` - 对数均匀分布
适用于学习率等跨越多个数量级的参数。

```json
{
  "learning_rate": {
    "type": "loguniform",
    "low": 1e-5,
    "high": 5e-4
  }
}
```

### 2. `uniform` - 线性均匀分布
适用于 warmup_ratio 等在固定范围内的小数参数。

```json
{
  "warmup_ratio": {
    "type": "uniform",
    "low": 0.0,
    "high": 0.1
  }
}
```

### 3. `int` - 整数范围
适用于 epoch 数等整数参数。

```json
{
  "num_train_epochs": {
    "type": "int",
    "low": 2,
    "high": 5,
    "step": 1
  }
}
```

### 4. `categorical` - 离散选择
适用于预定义的几个固定值。

```json
{
  "gradient_accumulation_steps": {
    "type": "categorical",
    "choices": [2, 4, 8]
  }
}
```

## 使用方法

### 基本用法（使用默认配置）

```bash
python core/hyperparameter_search.py \
  --n-trials 20 \
  --study-name my-study
```

这会使用代码中定义的默认搜索空间（learning_rate 和 num_train_epochs）。

### 使用自定义配置文件

```bash
python core/hyperparameter_search.py \
  --n-trials 20 \
  --study-name my-study \
  --config configs/hyperparm_configs/optuna/config.json
```

### 使用不同搜索策略

#### 贝叶斯优化（推荐，默认）
```bash
python core/hyperparameter_search.py \
  --search-mode bayesian \
  --n-trials 20 \
  --config configs/hyperparm_configs/optuna/config.json
```

#### 网格搜索（穷举所有组合）
```bash
python core/hyperparameter_search.py \
  --search-mode grid \
  --config configs/hyperparm_configs/optuna/config.json
```

注意：网格搜索只支持 `int` 和 `categorical` 类型。

#### 随机搜索
```bash
python core/hyperparameter_search.py \
  --search-mode random \
  --n-trials 50 \
  --config configs/hyperparm_configs/optuna/config.json
```

### 继续之前的研究
```bash
python core/hyperparameter_search.py \
  --study-name my-study \
  --resume
```

## 配置文件示例

### 示例 1: 只优化学习率和 epoch (config.json)

```json
{
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "low": 1e-5,
      "high": 5e-4
    },
    "num_train_epochs": {
      "type": "int",
      "low": 2,
      "high": 5
    }
  },
  "model_name": "/tmp2/share_data/google--gemma-3-12b-it",
  "input_data": "data/input/ocr_non_test_data=500.json",
  "lora_r": 16,
  "lora_alpha": 16,
  "gradient_accumulation_steps": 4
}
```

### 示例 2: 优化 LoRA 参数

```json
{
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "low": 1e-5,
      "high": 5e-4
    },
    "lora_r": {
      "type": "categorical",
      "choices": [8, 16, 32, 64]
    },
    "lora_alpha": {
      "type": "categorical",
      "choices": [8, 16, 32, 64]
    }
  },
  "model_name": "/tmp2/share_data/google--gemma-3-12b-it",
  "input_data": "data/input/ocr_non_test_data=500.json",
  "num_train_epochs": 3,
  "gradient_accumulation_steps": 4
}
```

### 示例 3: 完整超参数搜索 (config_full_example.json)

```json
{
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "low": 1e-6,
      "high": 5e-4
    },
    "num_train_epochs": {
      "type": "int",
      "low": 2,
      "high": 10
    },
    "gradient_accumulation_steps": {
      "type": "categorical",
      "choices": [2, 4, 8]
    },
    "lora_r": {
      "type": "categorical",
      "choices": [16, 32, 64]
    },
    "lora_alpha": {
      "type": "categorical",
      "choices": [16, 32, 64]
    },
    "warmup_ratio": {
      "type": "uniform",
      "low": 0.0,
      "high": 0.15
    }
  },
  "model_name": "/tmp2/share_data/google--gemma-3-12b-it",
  "input_data": "data/input/ocr_non_test_data=500.json",
  "per_device_train_batch_size": 1
}
```

## 可用的固定参数

除了 `search_space`，配置文件还可以包含以下固定参数：

### 数据相关
- `model_name`: 模型路径
- `input_data`: 数据文件路径
- `instruction`: 指令文本
- `train_split`: 训练集划分
- `val_split`: 验证集划分

### 模型相关
- `max_seq_length`: 最大序列长度
- `load_in_4bit`: 是否使用 4-bit 量化
- `lora_r`: LoRA rank（如果不在 search_space 中）
- `lora_alpha`: LoRA alpha（如果不在 search_space 中）

### 训练相关
- `num_train_epochs`: 训练轮数（如果不在 search_space 中）
- `per_device_train_batch_size`: 每设备批次大小
- `gradient_accumulation_steps`: 梯度累积步数（如果不在 search_space 中）
- `learning_rate`: 学习率（如果不在 search_space 中）
- `warmup_ratio`: 预热比例（如果不在 search_space 中）
- `eval_steps`: 评估步数
- `save_steps`: 保存步数
- `base_output_dir`: 输出目录

## 输出结果

优化完成后，结果保存在 `optuna_studies/` 目录：

- `{study_name}.db` - SQLite 数据库（所有试验历史）
- `{study_name}_best_params.json` - 最佳参数及元数据
- `{study_name}_best_config.json` - 可直接用于训练的最佳配置

每个试验的详细结果：
`models/hyperparam_search/trial_{N}/trial_results.json`

## 快速开始

1. **使用默认配置运行**
   ```bash
   python core/hyperparameter_search.py --n-trials 10 --study-name test
   ```

2. **使用提供的配置文件**
   ```bash
   python core/hyperparameter_search.py \
     --n-trials 20 \
     --study-name my-search \
     --config configs/hyperparm_configs/optuna/config.json
   ```

3. **自定义您的配置**
   - 复制 `config.json` 并根据需要修改
   - 在 `search_space` 中定义要优化的参数
   - 其他字段设置为固定值
