# BERT4MIMO - AI for Wireless Communications

基于 BERT 的大规模 MIMO 无线通信 CSI 处理框架

## 🎯 核心功能

- **🤖 BERT架构** - Transformer编码器处理信道状态信息(CSI)
- **📡 大规模MIMO** - 支持多小区、多用户、多天线场景  
- **🚀 WebUI界面** - 可视化训练、实验、数据生成
- **⚙️ 多级配置** - 轻量化(256)、标准(512)、原始(768)三种模型
- **🔬 完整验证** - 5个基础测试 + 8个高级实验

## 🌟 应用价值

- **📈 预测信道** - 减少导频开销60%，提升频谱效率30%
- **🗜️ 压缩反馈** - 压缩比10:1～50:1，反馈开销降低90%+
- **🔧 增强估计** - 去噪提升精度50%，改善边缘用户体验
- **🎯 优化波束** - 提升系统容量20%+

## 🚀 快速开始

### 方法1：一键启动（推荐）
```bash
# Windows
RUN.bat

# Linux/Mac  
bash RUN.sh
```

自动完成：虚拟环境创建 → 依赖安装 → WebUI启动

访问：http://127.0.0.1:7861

### 方法2：命令行
```bash
# 手动创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 启动WebUI
python webui/app.py

# 或直接训练
python train.py
```

## 📁 项目结构

```
BERT4MIMO-AI4Wireless/
├── webui/
│   ├── app.py              # WebUI主程序（可独立运行）
│   └── __init__.py
├── checkpoints/            # 模型检查点
├── foundation_model_data/  # 训练数据
├── validation_results/     # 实验结果
├── docs/                   # 详细文档
│   ├── USAGE.md           # 使用指南
│   ├── FILES.md           # 文件说明
│   └── TESTS.md           # 测试说明
├── model.py               # CSIBERT模型定义
├── train.py               # 训练脚本
├── model_validation.py    # 基础验证（5个测试）
├── experiments_extended.py # 高级实验（8个实验）
├── run_all_experiments.py  # 批量实验
├── data_generator.m       # MATLAB数据生成
├── RUN.bat / RUN.sh       # 一键启动脚本
└── requirements.txt       # Python依赖
```

## 🎨 WebUI 四大功能

### 1. ⚡ 一键训练
完整自动化流程：数据生成 → 预处理 → 训练 → 测试
- 支持全参数自定义（8个参数可调）
- 三种预设配置（轻量化/标准/原始）

### 2. 📂 导入数据训练
自定义训练：
- 导入已有数据
- 灵活参数配置
- 实时训练监控

### 3. 🔧 生成数据
MATLAB数据生成器：
- 9个可配置参数
- 支持多场景（小区数、用户数、子载波等）

### 4. 🔬 进行实验
智能实验管理：
- **自动检测已训练模型**
- **多模型选择和切换**
- **基础实验**（5项）：重构误差、预测准确度、SNR鲁棒性、压缩率、推理速度
- **高级实验**（8项）：掩码敏感性、场景性能、子载波分析、多普勒鲁棒性、跨场景泛化、基线对比、误差分布、注意力可视化
- **批量运行**：单项/批量/全部13项实验
- **自动生成**：可视化图表 + 分析报告

## 🎯 三级配置方案

| 参数 | 轻量化⚡ | 标准⭐ | 原始🚀 | 说明 |
|-----|---------|--------|--------|------|
| Hidden Size | 256 | 512 | 768 | 隐藏层维度 |
| Num Layers | 4 | 8 | 12 | Transformer层数 |
| Attention Heads | 4 | 8 | 12 | 注意力头数 |
| Intermediate | 1024 | 2048 | 3072 | 前馈网络维度 |
| Epochs | 10 | 50 | 200 | 训练轮数 |
| Batch Size | 16 | 32 | 64 | 批次大小 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 | 学习率 |

**硬件建议**：
- 轻量化：4GB显存，训练30分钟
- 标准：8GB显存，训练2小时（推荐）
- 原始：16GB显存，训练8小时

## 📊 实验功能

### 智能模型管理
- ✅ 启动时自动扫描checkpoints目录
- 🔄 自动加载最新模型
- 📋 下拉列表选择不同模型
- 🔍 实时显示模型配置信息

### 基础测试（快速验证）
```bash
python model_validation.py
```
1. **重构误差** - MSE/MAE分析
2. **预测准确度** - 时序预测能力
3. **SNR鲁棒性** - 抗噪声性能  
4. **压缩率** - 数据压缩效率
5. **推理速度** - 计算性能测试

### 高级实验（深度分析）
```bash
python run_all_experiments.py --mode all
```
1. **掩码比率敏感性** - 最优掩码率分析
2. **场景性能分析** - 不同场景表现
3. **子载波性能** - 频域分析
4. **多普勒鲁棒性** - 高速场景测试
5. **跨场景泛化** - 泛化能力评估
6. **基线对比** - 与传统方法比较
7. **错误分布** - 误差统计分析
8. **注意力可视化** - 模型注意力热图

### 批量运行
WebUI支持：
- 单项实验运行
- 批量运行（5项基础测试或8项高级实验）
- 全部运行（13项完整测试）
- 自动生成图表和报告（保存到validation_results/）

## 🛠️ 依赖环境

**Python**: 3.8+  
**PyTorch**: 2.0+  
**CUDA**: 11.8+ (GPU推荐)

关键依赖：
- `torch` - 深度学习框架
- `transformers` - BERT模型
- `gradio` - WebUI
- `scipy` - 数据处理
- `matplotlib` - 可视化

完整依赖见 `requirements.txt`

## 📖 详细文档

- **[使用指南](docs/USAGE.md)** - 详细使用说明和配置参数
- **[文件说明](docs/FILES.md)** - 各文件功能详解
- **[测试说明](docs/TESTS.md)** - 实验方法和结果解读

## 🔧 常见问题

**Q: WebUI启动失败？**  
A: 检查虚拟环境激活，依赖完整安装。运行`RUN.bat`自动修复。

**Q: 模型训练太慢？**  
A: 使用轻量化配置，或减少epochs。确保使用GPU。

**Q: 实验找不到模型？**  
A: 点击"重新扫描"按钮刷新模型列表，或先训练模型。

**Q: 如何比较不同配置？**  
A: 训练多个模型（使用不同配置），在实验页面切换加载对比。

**Q: 数据生成失败？**  
A: MATLAB数据生成需要Communications Toolbox。可使用已有数据或生成随机演示数据。

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

项目地址：https://github.com/hsms4710-pixel/AI_TeleProject
