# BERT4MIMO 项目从零构建指南
# Project Setup Guide from Scratch

> **最后更新**: 2025年11月18日  
> **适用对象**: 新团队成员、项目复现者、研究人员

---

## 目录 / Table of Contents

1. [环境准备](#1-环境准备)
2. [项目克隆与初始化](#2-项目克隆与初始化)
3. [数据生成](#3-数据生成)
4. [模型训练](#4-模型训练)
5. [模型验证](#5-模型验证)
6. [Web界面使用](#6-web界面使用)
7. [常见问题排查](#7-常见问题排查)
8. [进阶配置](#8-进阶配置)

---

## 1. 环境准备

### 1.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10/11, Linux, macOS | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.9+ | 3.11 或 3.13 |
| **GPU** | NVIDIA GPU (可选) | CUDA 11.8+ 兼容 GPU |
| **内存** | 8GB RAM | 16GB+ RAM |
| **存储** | 5GB 可用空间 | 10GB+ SSD |
| **MATLAB** | R2019b+ (数据生成) | R2023a+ |

### 1.2 软件安装

#### Step 1: 安装 Python 3.11+
```bash
# Windows - 从官网下载安装
https://www.python.org/downloads/

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# macOS (使用 Homebrew)
brew install python@3.11
```

#### Step 2: 安装 CUDA Toolkit (可选，用于GPU加速)
```bash
# 访问 NVIDIA 官网下载
https://developer.nvidia.com/cuda-downloads

# 验证安装
nvcc --version
nvidia-smi
```

#### Step 3: 安装 Git
```bash
# Windows - 从官网下载
https://git-scm.com/download/win

# Linux
sudo apt install git

# macOS
brew install git
```

#### Step 4: 安装 MATLAB (用于数据生成)
- 从 MathWorks 官网下载并安装
- 确保包含 Communications Toolbox

---

## 2. 项目克隆与初始化

### 2.1 克隆项目

```bash
# 克隆 GitHub 仓库
git clone https://github.com/hsms4710-pixel/AI_TeleProject.git

# 进入项目目录
cd AI_TeleProject/BERT4MIMO-AI4Wireless
```

### 2.2 创建虚拟环境

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows CMD
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 安装依赖

```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 验证关键库
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**预期输出**:
```
PyTorch: 2.x.x+cu118 (或 cu121)
CUDA Available: True (如果有 GPU)
```

### 2.4 目录结构检查

```bash
# 确认目录结构
tree /F  # Windows
ls -R    # Linux/macOS
```

**应包含以下关键文件**:
```
BERT4MIMO-AI4Wireless/
├── model.py                    # 模型定义
├── train.py                    # 训练脚本
├── model_validation.py         # 验证脚本
├── experiments_extended.py     # 高级实验
├── data_generator.m            # MATLAB 数据生成器
├── requirements.txt            # Python 依赖
├── START.bat                   # Windows 快速启动
├── webui/
│   └── app.py                  # Gradio Web 界面
├── docs/                       # 文档目录
└── foundation_model_data/      # 数据存储目录 (待创建)
```

---

## 3. 数据生成

### 3.1 方案 A: 使用 MATLAB 生成数据 (推荐)

#### Step 1: 打开 MATLAB
```matlab
% 在 MATLAB 中导航到项目目录
cd('C:\path\to\BERT4MIMO-AI4Wireless')
```

#### Step 2: 运行数据生成脚本
```matlab
% 执行数据生成
run('data_generator.m')
```

**生成参数**:
- **小区数量**: 10 cells
- **用户数**: 200 UEs per cell (共 2000 用户)
- **子载波数**: 64 subcarriers
- **基站天线数**: 64 antennas
- **用户天线数**: 4 antennas
- **输出文件**: `foundation_model_data/csi_data_massive_mimo.mat`

**预计生成时间**: 5-15 分钟 (取决于 CPU 性能)

#### Step 3: 验证数据
```matlab
% 加载并检查数据
load('foundation_model_data/csi_data_massive_mimo.mat');
disp(size(csi_data));  % 应显示类似 [10×1 cell]
```

### 3.2 方案 B: 使用预生成数据

如果无法访问 MATLAB，可以从以下来源获取预生成数据：
1. 联系项目维护者获取数据文件
2. 从项目发布页面下载 (如有提供)
3. 使用 Python 替代脚本生成 (需自行实现)

### 3.3 数据文件放置

确保生成的 `.mat` 文件位于正确位置：
```
BERT4MIMO-AI4Wireless/
└── foundation_model_data/
    └── csi_data_massive_mimo.mat  ✓
```

---

## 4. 模型训练

### 4.1 配置训练参数

编辑 `train.py` 中的超参数（可选）:
```python
# 关键参数
hidden_size = 256        # 隐藏层维度
num_layers = 4           # Transformer 层数
num_heads = 4            # 注意力头数
batch_size = 64          # 批次大小
num_epochs = 100         # 训练轮数
learning_rate = 0.0001   # 学习率
patience = 15            # 早停耐心值
```

### 4.2 启动训练

```bash
# 激活虚拟环境后执行
python train.py
```

**训练过程输出示例**:
```
Loading and preprocessing data...
Data loaded: 2000 samples
Train: 1400, Val: 200, Test: 400 samples

Initializing CSIBERT model...
Model: 12,644,608 parameters
Device: cuda

Starting training...
Epoch 1/100
Train Loss: 0.0234 | Val Loss: 0.0189
Best model saved!

Epoch 2/100
Train Loss: 0.0156 | Val Loss: 0.0145
Best model saved!
...
```

### 4.3 训练输出

训练完成后会生成以下文件：
```
checkpoints/
└── best_model.pt              # 最佳模型权重 (~50MB)

validation_data/
└── test_data.npy              # 测试集 (~338MB, 已 gitignore)

logs/
└── training_log.txt           # 训练日志
```

### 4.4 训练时间估算

| 硬件配置 | 预计时间 |
|---------|---------|
| CPU (16 cores) | 4-8 小时 |
| GPU (RTX 3060) | 30-60 分钟 |
| GPU (RTX 4090) | 10-20 分钟 |

### 4.5 训练监控

```bash
# 实时监控训练日志
# Windows PowerShell
Get-Content logs/training_log.txt -Wait -Tail 20

# Linux/macOS
tail -f logs/training_log.txt
```

---

## 5. 模型验证

### 5.1 运行验证脚本

```bash
# 方式 1: 使用验证脚本
python model_validation.py

# 方式 2: 使用高级实验脚本
python experiments_extended.py
```

### 5.2 验证测试项

| 测试项 | 描述 | 输出文件 |
|--------|------|---------|
| **重构误差测试** | CSI 重构精度 | `reconstruction_error.png` |
| **预测准确率测试** | 序列预测性能 | `prediction_accuracy.png` |
| **SNR 鲁棒性测试** | 不同信噪比下表现 | `snr_robustness.png` |
| **压缩比测试** | 数据压缩效率 | `compression_analysis.png` |
| **推理速度测试** | 模型推理性能 | `inference_speed.json` |

### 5.3 验证结果

所有结果保存在：
```
validation_results/
├── reconstruction_error.png
├── prediction_accuracy.png
├── snr_robustness.png
├── compression_analysis.png
├── inference_speed.json
├── validation_report.json
└── VALIDATION_REPORT.md      # 完整报告
```

### 5.4 性能基准

**预期性能指标**:
- 重构误差 (MSE): < 0.01
- 预测准确率: > 85%
- 压缩比: 4:1 - 8:1
- 推理速度 (GPU): < 10ms per sample

---

## 6. Web界面使用

### 6.1 启动 WebUI

#### 方式 1: 使用快速启动脚本 (推荐)
```bash
# Windows - 双击运行
START.bat

# 或在 PowerShell 中
.\START.bat
```

#### 方式 2: 手动启动
```bash
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/macOS

# 启动 WebUI
python webui/app.py
```

### 6.2 访问界面

启动成功后，在浏览器中访问：
```
http://127.0.0.1:7861
```

**界面预览**:
```
╔══════════════════════════════════════╗
║  BERT4MIMO Web Interface            ║
╠══════════════════════════════════════╣
║  📊 Training          - 模型训练     ║
║  🔬 Advanced Experiments - 高级实验  ║
║  ✅ Validation        - 模型验证     ║
║  💾 Model Management  - 模型管理     ║
║  ❓ Help              - 使用帮助     ║
╚══════════════════════════════════════╝
```

### 6.3 功能说明

#### Tab 1: Training (训练)
- 配置超参数
- 启动/停止训练
- 实时查看训练曲线
- 下载训练日志

#### Tab 2: Advanced Experiments (高级实验)
1. **SNR 鲁棒性测试**: 测试不同信噪比下的性能
2. **时域相关性分析**: 分析 CSI 时间序列特征
3. **多用户干扰测试**: 评估多用户场景性能
4. **压缩重构权衡分析**: 压缩比与质量的平衡
5. **通道估计性能**: 信道估计精度评估

#### Tab 3: Validation (验证)
- 一键运行所有验证测试
- 查看验证报告
- 下载结果图表

#### Tab 4: Model Management (模型管理)
- 查看模型信息
- 加载不同检查点
- 导出模型

#### Tab 5: Help (帮助)
- 快速开始指南
- API 文档
- 常见问题

---

## 7. 常见问题排查

### 7.1 环境问题

#### Q1: `ModuleNotFoundError: No module named 'torch'`
**解决方案**:
```bash
# 确认虚拟环境已激活
.\.venv\Scripts\Activate.ps1

# 重新安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Q2: `CUDA out of memory`
**解决方案**:
```python
# 在 train.py 中减小 batch_size
batch_size = 32  # 从 64 改为 32
```

#### Q3: 虚拟环境无法激活
**解决方案 (Windows)**:
```powershell
# 设置 PowerShell 执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 7.2 数据问题

#### Q4: `FileNotFoundError: foundation_model_data/csi_data_massive_mimo.mat`
**解决方案**:
1. 确认已运行 MATLAB 数据生成脚本
2. 检查文件路径是否正确
3. 确认文件大小 > 0 bytes

#### Q5: 数据加载失败或格式错误
**解决方案**:
```python
# 验证数据格式
import scipy.io as sio
data = sio.loadmat('foundation_model_data/csi_data_massive_mimo.mat')
print(data.keys())
print(type(data['csi_data']))
```

### 7.3 训练问题

#### Q6: 训练损失不下降
**检查项**:
- 学习率是否过大/过小
- 数据是否正确归一化
- 模型架构是否合理

**尝试调整**:
```python
learning_rate = 0.0001  # 尝试 0.001, 0.0001, 0.00001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
```

#### Q7: 显存不足但 GPU 空闲
**解决方案**:
```bash
# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 7.4 WebUI 问题

#### Q8: WebUI 无法启动
**检查步骤**:
```bash
# 1. 确认虚拟环境已激活
# 2. 检查 Gradio 是否安装
pip show gradio

# 3. 测试端口是否被占用
netstat -ano | findstr :7861  # Windows
lsof -i :7861                 # Linux/macOS
```

#### Q9: 模型加载失败
**解决方案**:
- 确认 `checkpoints/best_model.pt` 存在
- 检查文件大小是否正常 (~50MB)
- 确认模型架构参数一致

### 7.5 Git 问题

#### Q10: 推送被拒绝 (文件过大)
**解决方案**:
```bash
# 检查大文件
git ls-files | ForEach-Object {Get-Item $_ | Where-Object {$_.length -gt 100MB}}

# 从 Git 历史中移除
git rm --cached path/to/large/file
git commit --amend --no-edit
git push --force
```

---

## 8. 进阶配置

### 8.1 自定义模型架构

编辑 `model.py`:
```python
class CSIBERT(nn.Module):
    def __init__(
        self,
        input_size=256,      # 输入维度
        hidden_size=512,     # 增大隐藏层
        num_layers=6,        # 增加层数
        num_heads=8,         # 增加注意力头
        dropout=0.1
    ):
        # ... 模型定义
```

### 8.2 数据增强

在 `train.py` 中添加数据增强：
```python
def add_noise(data, snr_db=20):
    """添加高斯噪声"""
    signal_power = np.mean(np.abs(data)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*data.shape) + 1j*np.random.randn(*data.shape))
    return data + noise
```

### 8.3 分布式训练

使用 PyTorch DDP:
```bash
# 单机多卡训练
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### 8.4 实验跟踪

集成 Weights & Biases:
```bash
pip install wandb

# 在 train.py 中添加
import wandb
wandb.init(project="bert4mimo", config=config)
wandb.log({"loss": loss, "epoch": epoch})
```

### 8.5 模型导出

导出为 ONNX 格式：
```python
import torch.onnx

dummy_input = torch.randn(1, seq_length, input_size)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11
)
```

---

## 9. 项目时间表

### 完整构建时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| **Day 0** | 环境准备 + 软件安装 | 1-2 小时 |
| **Day 1** | 项目克隆 + 依赖安装 | 30 分钟 |
| **Day 1** | 数据生成 (MATLAB) | 15 分钟 |
| **Day 1-2** | 模型训练 (GPU) | 1-2 小时 |
| **Day 2** | 模型验证 | 30 分钟 |
| **Day 2** | WebUI 测试 | 30 分钟 |
| **总计** | | **4-6 小时** (含 GPU) |

---

## 10. 资源链接

### 官方文档
- **项目仓库**: https://github.com/hsms4710-pixel/AI_TeleProject
- **快速开始**: `docs/QUICK_START.md`
- **WebUI 指南**: `docs/WEBUI_GUIDE.md`

### 相关论文
- BERT: Pre-training of Deep Bidirectional Transformers
- CSI Feedback with Deep Learning
- Massive MIMO Channel Estimation

### 技术栈
- PyTorch: https://pytorch.org/
- Gradio: https://gradio.app/
- MATLAB Communications Toolbox: https://www.mathworks.com/products/communications.html

### 社区支持
- Issue 追踪: https://github.com/hsms4710-pixel/AI_TeleProject/issues
- 讨论区: https://github.com/hsms4710-pixel/AI_TeleProject/discussions

---

## 11. 检查清单

完成以下检查确保项目正确搭建：

- [ ] Python 3.9+ 已安装并可运行
- [ ] Git 已安装并配置
- [ ] MATLAB 已安装（可选但推荐）
- [ ] 项目已克隆到本地
- [ ] 虚拟环境已创建并激活
- [ ] requirements.txt 依赖已全部安装
- [ ] PyTorch CUDA 可用（如有 GPU）
- [ ] 数据文件 `csi_data_massive_mimo.mat` 已生成
- [ ] 训练脚本可正常运行
- [ ] 模型文件 `best_model.pt` 已生成
- [ ] 验证测试全部通过
- [ ] WebUI 可正常启动和访问
- [ ] 所有图表和报告已生成

---

## 12. 下一步行动

项目构建完成后，可以：

1. **研究实验**: 运行 5 个高级实验，分析结果
2. **模型优化**: 调整超参数，提升性能
3. **论文撰写**: 使用验证报告撰写论文
4. **代码贡献**: 向项目提交改进 PR
5. **应用部署**: 将模型部署到生产环境

---

## 附录 A: 命令速查表

### 常用命令

```bash
# 激活环境
.\.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate          # Linux/macOS

# 训练模型
python train.py

# 运行验证
python model_validation.py

# 启动 WebUI
python webui/app.py

# 查看 GPU 状态
nvidia-smi

# 检查磁盘空间
Get-PSDrive                        # Windows
df -h                              # Linux/macOS
```

---

## 附录 B: 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| CSI | Channel State Information | 信道状态信息 |
| MIMO | Multiple-Input Multiple-Output | 多输入多输出 |
| BERT | Bidirectional Encoder Representations from Transformers | 双向编码器表示 |
| SNR | Signal-to-Noise Ratio | 信噪比 |
| UE | User Equipment | 用户设备 |
| BS | Base Station | 基站 |
| MSE | Mean Squared Error | 均方误差 |
| NMSE | Normalized Mean Squared Error | 归一化均方误差 |

---

**文档维护**: GitHub Copilot  
**项目版本**: v1.0  
**联系方式**: 见 GitHub Issues

---

