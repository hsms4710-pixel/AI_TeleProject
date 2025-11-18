# BERT4MIMO-AI4Wireless

**基于 BERT 的大规模 MIMO 信道状态信息 (CSI) 预测与重构框架**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 项目简介

BERT4MIMO 是一个利用 Transformer 架构处理无线通信中信道状态信息 (CSI) 的深度学习项目。通过自监督学习，模型能够：

- 📡 **预测未来 CSI**：减少导频开销，提升频谱效率
- 🗜️ **压缩 CSI 反馈**：显著降低反馈开销
- 🔊 **去噪与增强**：提升信道估计精度
- 📊 **多场景适应**：支持多小区、多用户、大规模天线阵列

**最新版本**: v2.0.0 - 集成 WebUI，支持可视化训练和实验管理

---

## ⚡ 快速开始（3步启动）

### 1️⃣ 安装依赖

```bash
# 克隆项目
git clone https://github.com/hsms4710-pixel/AI_TeleProject.git
cd AI_TeleProject/BERT4MIMO-AI4Wireless

# 安装依赖
pip install -r requirements.txt
```

**环境要求**：Python 3.8+ | PyTorch 2.0+ | CUDA (可选)

### 2️⃣ 启动 WebUI（推荐）

```bash
# Windows 用户 - 双击运行
START.bat

# 或命令行启动
python webui/app.py
```

**浏览器访问**：http://127.0.0.1:7861

### 3️⃣ 开始使用

通过 WebUI 一键完成：
- ✅ 训练模型（选择配置方案或自定义参数）
- ✅ 运行实验（5项基础测试 + 5项高级实验）
- ✅ 查看结果（实时图表和报告）
- ✅ 管理模型（自动扫描和加载）

---

## 🖥️ 命令行模式（高级用户）

如果你更喜欢命令行操作：

### 训练模型

**使用默认参数：**

```bash
python train.py
```

**自定义参数：**
cd BERT4MIMO-AI4Wireless

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装PyTorch（请根据你的系统选择合适版本）
# 访问 https://pytorch.org/ 获取安装命令

# 安装其他依赖
pip install -r requirements.txt
```

### 开始训练

**方式1：使用默认参数快速训练**

```bash
python train.py
```

**方式2：使用 Windows 批处理脚本**

```bash
RUN.bat
```

**方式3：使用 Web 界面（推荐）**

```bash
# 启动 WebUI
RUN_WEBUI.bat

# 或者直接运行
python webui/app.py
```

然后在浏览器中打开 http://127.0.0.1:7861，通过图形界面进行：
- 🎯 一键训练
- 📊 实时监控训练进度
- 🧪 运行各种实验
- 📈 可视化结果分析

**方式4：自定义参数训练**

```bash
python train.py \
    --hidden_size 256 \
    --num_hidden_layers 4 \
    --num_attention_heads 4 \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --num_epochs 50
```

### 验证和实验

**基础验证（5项测试）：**

```bash
python model_validation.py
```

**高级实验（5项分析）：**

```bash
# 运行所有高级实验
python experiments_extended.py

# 或运行单个实验
python experiments_extended.py --experiment 1  # 掩码比率敏感性
python experiments_extended.py --experiment 2  # 误差分布分析
python experiments_extended.py --experiment 3  # 预测步长分析
python experiments_extended.py --experiment 4  # 基线方法对比
python experiments_extended.py --experiment 5  # 注意力可视化
```

**结果目录：**
- `validation_results/` - 基础验证结果（图表、报告）
- `advanced_experiments/` - 高级实验结果（CSV、JSON、PNG）

---

## 📁 项目结构

```
BERT4MIMO-AI4Wireless/
├── START.bat                       # 主启动脚本（启动WebUI）
├── model.py                        # CSIBERT 模型定义
├── train.py                        # 训练脚本（含数据拆分）
├── model_validation.py             # 基础验证脚本（5项测试）
├── experiments_extended.py         # 高级实验脚本（5项实验）
├── data_generator.m                # MATLAB 数据生成脚本
├── requirements.txt                # Python 依赖列表
├── setup_environment.py            # 环境配置脚本
│
├── foundation_model_data/          # 训练数据
│   └── csi_data_massive_mimo.mat   # MATLAB 格式的 CSI 数据（可用 data_generator.m 生成）
│
├── checkpoints/                    # 模型检查点
│   └── best_model.pt               # 最佳模型（训练后生成）
│
├── validation_data/                # 验证数据
│   └── test_data.npy               # 测试集（训练时生成）
│
├── validation_results/             # 基础验证结果
│   ├── validation_report.json      # 详细指标（JSON）
│   ├── VALIDATION_REPORT.md        # 可视化报告（Markdown）
│   └── *.png                       # 性能曲线图
│
├── advanced_experiments/           # 高级实验结果
│   ├── exp1_masking_ratio.csv/png  # 掩码比率敏感性
│   ├── exp2_error_stats.json/png   # 误差分布分析
│   ├── exp3_prediction_horizon.*   # 预测步长分析
│   ├── exp4_baseline_comparison.*  # 基线方法对比
│   └── exp5_attention_sample_*.png # 注意力可视化
│
├── webui/                          # 🌐 Web 图形界面
│   ├── __init__.py
│   └── app.py                      # Gradio 界面应用
│
├── docs/                           # 📚 文档
│   └── WEBUI_GUIDE.md              # WebUI 使用指南
│
└── .github/
    └── copilot-instructions.md     # AI 编程助手指南
```

---

## 🌐 WebUI 功能特性

### 🎯 四大功能模块

1. **一键训练** - 自动化训练流程
   - 3种预设配置（轻量级/标准/高性能）
   - 自定义参数设置
   - 实时训练日志
   - 损失曲线可视化

2. **导入数据训练** - 使用现有数据
   - 自动数据加载
   - 配置方案选择
   - 训练进度监控

3. **生成数据** - 合成CSI数据集
   - 9种参数配置
   - 自定义场景参数

4. **实验管理** - 性能评估与分析
   - 5项基础测试（重构/预测/SNR/压缩/速度）
   - 5项高级实验（掩码敏感性/误差分布/步长分析/基线对比/注意力可视化）
   - 自动生成图表和报告
   - 结果实时展示

### 💡 使用优势

- ✅ **零编程** - 图形界面操作，无需命令行
- ✅ **实时反馈** - 训练进度、损失曲线实时更新
- ✅ **一键实验** - 批量运行所有测试
- ✅ **智能管理** - 自动扫描和加载已训练模型
- ✅ **完整文档** - 内置使用说明和配置建议

---

## 🚀 工作流程

### 1️⃣ 数据准备

**选项A：使用已有数据**
- **位置**：`foundation_model_data/csi_data_massive_mimo.mat`
- **格式**：复数矩阵 (时间 × 子载波 × 天线)
- **场景**：多小区、多用户、大规模MIMO

**选项B：生成新数据（需要 MATLAB）**

使用 `data_generator.m` 生成 Massive MIMO CSI 数据：

```matlab
% 在 MATLAB 中运行
data_generator.m
```

**数据生成参数**：
- 基站数量：10 个小区
- 用户数量：200 UE/小区
- 子载波数：64
- 基站天线：64（Massive MIMO）
- 用户天线：4
- 采样率：30.72 MHz（5G NR）
- 信噪比：25 dB
- 移动速度：120 km/h（高速场景）
- 载波频率：3.5 GHz

生成的数据将保存为 `csi_data_massive_mimo.mat`，包含多小区、多用户的完整 CSI 数据。

### 2️⃣ 模型训练

训练脚本 (`train.py`) 自动完成：

1. **数据加载与预处理**
   - 分离实部/虚部
   - 标准化
   - 序列填充

2. **数据拆分**
   - 训练集：70%
   - 验证集：10%
   - 测试集：20%（保存供后续验证）

3. **模型训练**
   - 自监督学习（掩码重构）
   - 验证集监控
   - 早停机制（patience=15）
   - 保存最佳模型

4. **输出**
   - `checkpoints/best_model.pt`：最佳模型权重
   - `training_validation_loss.png`：训练/验证损失曲线
   - `validation_data/test_data.npy`：测试集数据

### 3️⃣ 模型验证与实验

#### 基础验证（5项测试）

验证脚本 (`model_validation.py`) 在独立测试集上评估：

**评估指标：**

| 测试项 | 指标 | 说明 |
|--------|------|------|
| **重构误差** | MSE, NMSE, MAE | 掩码重构精度 |
| **预测准确度** | NMSE vs 预测步长 | 时序预测能力（1/3/5/10步） |
| **SNR 鲁棒性** | NMSE vs SNR | 噪声环境下的性能（-10~30dB） |
| **压缩质量** | NMSE vs 压缩率 | 压缩-质量权衡（10x~50x） |
| **推理速度** | 吞吐量 vs Batch Size | 实时性能（1/8/16/32） |

**生成的报告：**
- `validation_results/validation_report.json`
- `validation_results/VALIDATION_REPORT.md`
- 各类性能曲线图

#### 高级实验（5项实验）

高级实验脚本 (`experiments_extended.py`) 提供深度分析：

| 实验项 | 输出 | 说明 |
|--------|------|------|
| **掩码比率敏感性** | CSV + PNG | 测试15种掩码比率（0-70%） |
| **误差分布分析** | JSON + PNG | 直方图、箱线图、Q-Q图 |
| **预测步长分析** | JSON + PNG | 测试1-20步预测能力 |
| **基线方法对比** | JSON + PNG | 与零填充、均值填充比较 |
| **注意力可视化** | PNG | 注意力权重热力图 |

**生成的结果：**
- `advanced_experiments/exp1_*` - 实验1结果
- `advanced_experiments/exp2_*` - 实验2结果
- `advanced_experiments/exp3_*` - 实验3结果
- `advanced_experiments/exp4_*` - 实验4结果
- `advanced_experiments/exp5_*` - 实验5结果

---

## 🔧 模型架构

### CSIBERT

**核心组件：**

```
输入 CSI (T, F)
    ↓
[时间嵌入] + [特征嵌入]
    ↓
BERT Encoder (多层 Transformer)
    ↓
输出层 (全连接)
    ↓
重构 CSI (T, F)
```

- **T**: 时间步数（序列长度）
- **F**: 特征维度（子载波 × 天线 × 实虚部）

### 配置方案

| 方案 | Hidden | Layers | Heads | 场景 | 显存 | 训练时间 |
|------|--------|--------|-------|------|------|----------|
| 轻量级⚡ | 256 | 4 | 4 | 快速体验 | 4GB | 5分钟 |
| 标准🎯 | 256 | 6 | 8 | 推荐使用 | 8GB | 15分钟 |
| 高性能💪 | 512 | 8 | 8 | 最佳效果 | 16GB | 30分钟 |

4. **输出**
   - `checkpoints/best_model.pt`：最佳模型权重
   - `training_validation_loss.png`：训练/验证损失曲线
   - `validation_data/test_data.npy`：测试集数据

### 3️⃣ 模型验证与实验

#### 基础验证（5项测试）

验证脚本 (`model_validation.py`) 在独立测试集上评估：

**评估指标：**

| 测试项 | 指标 | 说明 |
|--------|------|------|
| **重构误差** | MSE, NMSE, MAE | 掩码重构精度 |
| **预测准确度** | NMSE vs 预测步长 | 时序预测能力（1/3/5/10步） |
| **SNR 鲁棒性** | NMSE vs SNR | 噪声环境下的性能（-10~30dB） |
| **压缩质量** | NMSE vs 压缩率 | 压缩-质量权衡（10x~50x） |
| **推理速度** | 吞吐量 vs Batch Size | 实时性能（1/8/16/32） |

**生成的报告：**
- `validation_results/validation_report.json`
- `validation_results/VALIDATION_REPORT.md`
- 各类性能曲线图

#### 高级实验（5项实验）

高级实验脚本 (`experiments_extended.py`) 提供深度分析：

| 实验项 | 输出 | 说明 |
|--------|------|------|
| **掩码比率敏感性** | CSV + PNG | 测试15种掩码比率（0-70%） |
| **误差分布分析** | JSON + PNG | 直方图、箱线图、Q-Q图 |
| **预测步长分析** | JSON + PNG | 测试1-20步预测能力 |
| **基线方法对比** | JSON + PNG | 与零填充、均值填充比较 |
| **注意力可视化** | PNG | 注意力权重热力图 |

**生成的结果：**
- `advanced_experiments/exp1_*` - 实验1结果
- `advanced_experiments/exp2_*` - 实验2结果
- `advanced_experiments/exp3_*` - 实验3结果
- `advanced_experiments/exp4_*` - 实验4结果
- `advanced_experiments/exp5_*` - 实验5结果

---

## 🔧 模型架构

### CSIBERT

**核心组件：**

```
输入 CSI (T, F)
    ↓
[时间嵌入] + [特征嵌入]
    ↓
BERT Encoder (多层 Transformer)
    ↓
输出层 (全连接)
    ↓
重构 CSI (T, F)
```

- **T**: 时间步数（序列长度）
- **F**: 特征维度（子载波 × 天线 × 实虚部）

---

## 📊 实验结果示例

典型性能指标（在标准配置下）：
- **重构误差**：NMSE < -20 dB
- **压缩性能**：支持 10x - 50x 压缩率
- **预测能力**：1-10步准确预测
- **推理速度**：GPU 上可达数千 samples/s

> 完整结果请运行 `model_validation.py` 或通过 WebUI 查看

---

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [QUICK_START.md](QUICK_START.md) | 30秒快速入门指南 |
| [docs/WEBUI_GUIDE.md](docs/WEBUI_GUIDE.md) | WebUI 详细使用说明 |
| [README.md](README.md) | 项目完整文档（本文件） |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发指南

1. Fork 本项目
2. 创建新分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目基于以下开源库：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Transformer 模型
- [Gradio](https://gradio.app/) - Web 界面框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具

---

## 📞 支持与反馈

- **GitHub Issues**: [提交问题](https://github.com/hsms4710-pixel/AI_TeleProject/issues)
- **项目主页**: [AI_TeleProject](https://github.com/hsms4710-pixel/AI_TeleProject)

---

**🌟 如果这个项目对你有帮助，请给个 Star！**

[![Star History Chart](https://api.star-history.com/svg?repos=hsms4710-pixel/AI_TeleProject&type=Date)](https://star-history.com/#hsms4710-pixel/AI_TeleProject&Date)
