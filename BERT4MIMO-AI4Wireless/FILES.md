# 文件说明 - BERT4MIMO 项目文件结构

## 📋 核心文件

### 1. `model.py` - CSIBERT 模型定义
**行数**：~500行  
**语言**：Python  
**依赖**：PyTorch, Transformers, NumPy

**主要内容**：
- `CSIBERT` 类：主要 Transformer 模型
- `forward()` 方法：前向推理
- 工具函数：数据预处理、评估指标

**使用**：
```python
from model import CSIBERT

# 轻量化配置
model = CSIBERT(
    feature_dim=1024, 
    hidden_size=256, 
    num_hidden_layers=4
)

# 标准配置（推荐）
model = CSIBERT(
    feature_dim=1024, 
    hidden_size=512, 
    num_hidden_layers=8
)

# 原始配置
model = CSIBERT(
    feature_dim=1024, 
    hidden_size=768, 
    num_hidden_layers=12
)
```

**核心参数对比**：

| 参数 | 轻量化 ⚡ | 标准 ⭐ | 原始 🚀 | 说明 |
|------|---------|--------|--------|------|
| `hidden_size` | 256 | 512 | 768 | 隐层维度 |
| `num_hidden_layers` | 4 | 8 | 12 | Transformer层数 |
| `num_attention_heads` | 4 | 8 | 12 | 注意力头数 |
| `intermediate_size` | 1024 | 2048 | 3072 | FFN隐层维度 |
| `max_position_embeddings` | 2048 | 4096 | 4096 | 最大序列长度 |

---

### 2. `train.py` - 模型训练脚本
**行数**：~300行  
**语言**：Python  
**依赖**：PyTorch, Transformers, Scipy

**主要功能**：
- 数据加载和预处理
- 模型初始化和配置
- 完整的训练循环
- 模型保存和评估

**关键功能**：
- 支持 GPU/CPU 自动选择
- 支持不同的优化器
- 支持检查点保存
- 支持命令行参数配置

**三级配置使用**：

```bash
# 轻量化配置（快速测试）
python train.py \
    --hidden_size 256 \
    --num_layers 4 \
    --batch_size 16 \
    --max_epochs 10

# 标准配置（推荐，平衡）
python train.py \
    --hidden_size 512 \
    --num_layers 8 \
    --batch_size 32 \
    --max_epochs 50

# 原始配置（高精度）
python train.py \
    --hidden_size 768 \
    --num_layers 12 \
    --batch_size 64 \
    --max_epochs 200
```

**训练参数对比**：

| 参数 | 轻量化 ⚡ | 标准 ⭐ | 原始 🚀 |
|------|---------|--------|--------|
| `batch_size` | 16 | 32 | 64 |
| `max_epochs` | 10 | 50 | 200 |
| `learning_rate` | 1e-4 | 1e-4 | 1e-4 |
| `early_stopping_patience` | 5 | 10 | 200 |

**输出**：
- 训练日志
- 检查点：`checkpoints/best_model.pt`
- 性能指标

---

### 3. `model_validation.py` - 模型验证脚本
**行数**：~840行  
**语言**：Python  
**依赖**：PyTorch, Matplotlib, Scikit-learn

**主要功能**：
5 个基础验证测试：
1. 重构误差测试
2. 预测准确度测试
3. SNR 鲁棒性测试
4. 压缩率测试
5. 推理速度测试

**核心类**：
- `CSIBERTValidator`：验证器主类
- 各测试方法：`test_*`

**使用方式**：
```bash
python model_validation.py
```

**输出**：
- 性能指标（JSON）
- 验证图表（PNG）
- 详细报告（Markdown）

---

### 4. `run_all_experiments.py` - 统一实验运行器
**行数**：~190行  
**语言**：Python  
**依赖**：model_validation.py, experiments_extended.py

**主要功能**：
- 加载模型和数据
- 运行所有测试
- 生成综合报告
- 结果保存

**三种运行模式**：
- `basic`：仅基础验证 (2-5分钟)
- `advanced`：仅高级实验 (8-25分钟)
- `all`：全部测试 (10-30分钟)

**使用方式**：
```bash
python run_all_experiments.py --mode all
```

**输出**：
- 13+ 性能图表
- JSON 详细数据
- Markdown 报告

---

### 5. `experiments_extended.py` - 高级实验套件
**行数**：~590行  
**语言**：Python  
**依赖**：PyTorch, Matplotlib, Scikit-learn

**主要功能**：
8 个高级实验（来自 Jupyter Notebook）：
1. 掩码比率敏感性分析
2. 场景性能分析
3. 子载波性能测试
4. 多普勒移位鲁棒性
5. 跨场景泛化测试
6. 基线模型对比
7. 注意力机制可视化
8. 错误分布分析

**核心类**：
- `AdvancedCSIBERTExperiments`：实验套件主类

**使用方式**：
```python
from experiments_extended import AdvancedCSIBERTExperiments
experiments = AdvancedCSIBERTExperiments(model, data, ...)
results = experiments.experiment_masking_ratio_sensitivity()
```

---

### 6. `data_generator.m` - 数据生成脚本
**行数**：~200行  
**语言**：MATLAB  
**依赖**：MATLAB R2020a+ (Communications Toolbox 可选)

**主要功能**：
- 生成大规模 MIMO CSI 数据
- 支持多小区场景
- 支持多用户配置
- 生成 ~1.4GB 的训练数据

**输出**：
- 文件名：`foundation_model_data/csi_data_massive_mimo.mat`
- 大小：~1.4 GB
- 格式：MATLAB `.mat` 文件

**使用方式**：
```bash
matlab -batch "run('data_generator.m')"
```

**配置参数**（在 MATLAB 中修改）：
- 小区数
- 用户数
- 天线数
- 子载波数

---

## 📁 目录结构

### 核心模块
```
├── model.py                          # CSIBERT 模型定义
├── train.py                          # 训练脚本
├── model_validation.py               # 模型验证
├── run_all_experiments.py            # 统一实验运行器
├── experiments_extended.py           # 高级实验套件
└── data_generator.m                  # MATLAB 数据生成
```

### WebUI 模块
```
├── webui/
│   ├── app.py                        # Gradio 应用主程序
│   ├── README.md                     # WebUI 使用说明
│   ├── training_config.json          # 训练配置模板
│   └── TRAINING_PARAMETERS.md        # 参数说明
```

### 启动脚本
```
├── RUN.bat                           # Windows 一键启动
├── RUN.sh                            # Linux/Mac 一键启动
└── setup_environment.py              # 环境初始化脚本
```

### 数据与结果
```
├── foundation_model_data/            # 训练数据目录
│   ├── README.md
│   └── csi_data_massive_mimo.mat     # 生成的数据 (~1.4GB)
├── checkpoints/                      # 模型检查点
│   └── best_model.pt                 # 最优模型
└── validation_results/               # 验证结果
    ├── *.png                         # 性能图表
    ├── *.json                        # 详细数据
    └── VALIDATION_REPORT.md          # 报告
```

### 文档
```
├── README.md                         # 项目介绍
├── USAGE.md                          # 使用指南
├── FILES.md                          # 文件说明（本文件）
├── TESTS.md                          # 测试说明
└── requirements.txt                  # 依赖列表
```

---

## 🔄 文件之间的关系

```
数据流：
data_generator.m 
    ↓
foundation_model_data/csi_data_massive_mimo.mat
    ↓
train.py (加载数据，训练模型)
    ↓
checkpoints/best_model.pt
    ↓
model_validation.py (单个测试)
run_all_experiments.py (统一运行所有测试)
    ↓
validation_results/ (输出结果)

导入关系：
model.py (模型定义)
    ↓ 导入
train.py (训练，导入 model.py)
model_validation.py (验证，导入 model.py)
experiments_extended.py (实验，导入 model.py)
run_all_experiments.py (统一运行，导入上面所有)
```

---

## 📊 文件大小和性能

| 文件 | 大小 | 加载时间 | 执行时间 |
|------|------|---------|---------|
| model.py | ~20KB | <1秒 | - |
| train.py | ~15KB | <1秒 | 30-60分钟 |
| model_validation.py | ~40KB | <1秒 | 2-5分钟 |
| experiments_extended.py | ~25KB | <1秒 | 8-25分钟 |
| data_generator.m | ~15KB | <1秒 | 5-10分钟 |
| checkpoints/best_model.pt | ~87MB | 3-5秒 | - |
| csi_data_massive_mimo.mat | ~1.4GB | 10-20秒 | - |

---

## 🔧 文件修改指南

### 修改模型架构
编辑 `model.py` 中的 `CSIBERT` 类

### 修改训练参数
编辑 `train.py` 中的默认参数或使用命令行参数

### 修改验证测试
编辑 `model_validation.py` 中的 `CSIBERTValidator` 类

### 修改实验项目
编辑 `experiments_extended.py` 中的 `AdvancedCSIBERTExperiments` 类

### 修改数据生成
编辑 `data_generator.m` 中的配置参数

---

## 📚 相关文档

- **README.md** - 项目介绍
- **USAGE.md** - 使用指南
- **TESTS.md** - 测试说明
