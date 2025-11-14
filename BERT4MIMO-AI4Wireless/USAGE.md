# 使用指南 - BERT4MIMO 模块使用方法

## 🚀 快速开始

### 一键启动（推荐）
```bash
# Windows
RUN.bat

# Linux/Mac
bash RUN.sh
```

自动完成：虚拟环境创建 → 依赖安装 → WebUI 启动

---

## 📚 核心模块使用

### 1. 模型定义 - `model.py`

**用途**：CSIBERT 模型定义和工具函数

**基本使用**：
```python
from model import CSIBERT

# 创建模型
model = CSIBERT(
    feature_dim=1024,      # 输入特征维度
    hidden_size=256,       # 隐层维度
    num_hidden_layers=4,   # Transformer 层数
    num_attention_heads=4  # 注意力头数
)

# 移到设备
model = model.to('cuda')

# 前向传播
outputs = model(input_ids, attention_mask)
```

**主要函数**：
- `CSIBERT` - 主模型类
- `CSIBERT.forward()` - 前向推理
- 工具函数：数据预处理、评估指标计算

---

### 2. 模型训练 - `train.py`

**用途**：完整的训练脚本，支持 WebUI 和命令行

**命令行使用**：
```bash
# 基本训练
python train.py

# 自定义参数
python train.py \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_epochs 200 \
    --hidden_size 256 \
    --num_layers 4
```

**参数说明**：
- `--batch_size`: 批处理大小 (默认 32)
- `--learning_rate`: 学习率 (默认 2e-4)
- `--max_epochs`: 最大训练轮数 (默认 200)
- `--hidden_size`: 隐层维度 (默认 256)
- `--num_layers`: Transformer 层数 (默认 4)
- `--device`: 计算设备 (默认 'cuda' 或 'cpu')

**WebUI 使用**：
```bash
python -m webui.app
```
访问 http://127.0.0.1:7861，选择训练方案并配置参数

---

### 3. 数据生成 - `data_generator.m`

**用途**：使用 MATLAB 生成 CSI 训练数据

**运行方式**：
```bash
# 方法 1：直接运行
matlab -batch "run('data_generator.m')"

# 方法 2：MATLAB GUI 中打开后运行

# 方法 3：WebUI 中选择"仅生成数据"方案
```

**生成内容**：
- 输出文件：`foundation_model_data/csi_data_massive_mimo.mat`
- 数据大小：~1.4 GB
- 包含数据：多小区、多用户、多场景的 CSI 数据

**依赖**：
- MATLAB R2020a 或更新版本
- Communications Toolbox (可选)

---

### 4. 模型验证 - `model_validation.py`

**用途**：快速验证模型性能

**基础验证**（5个测试，2-5分钟）：
```bash
python model_validation.py
```

**测试内容**：
1. 重构误差 - MSE, NMSE, MAE
2. 预测准确度 - 时序预测能力
3. SNR 鲁棒性 - 不同信噪比下的性能
4. 压缩率 - 掩码比率与质量权衡
5. 推理速度 - 不同批大小下的速度

**输出**：
- 控制台：文本格式的性能指标
- 图表：`validation_results/` 目录

---

### 5. 完整实验套件 - `run_all_experiments.py`

**用途**：运行所有验证和实验方法

**运行模式**：
```bash
# 运行所有（基础+高级，10-30分钟）
python run_all_experiments.py --mode all

# 仅运行基础验证（2-5分钟）
python run_all_experiments.py --mode basic

# 仅运行高级实验（8-25分钟）
python run_all_experiments.py --mode advanced

# 自定义输出目录
python run_all_experiments.py --mode all --output my_results
```

**包含的 13 个测试方法**：

**基础验证（5个）**：
1. 重构误差
2. 预测准确度
3. SNR 鲁棒性
4. 压缩率
5. 推理速度

**高级实验（8个）**：
6. 掩码比率敏感性
7. 场景性能分析
8. 子载波性能
9. 多普勒移位鲁棒性
10. 跨场景泛化
11. 基线模型对比
12. 注意力机制可视化
13. 错误分布分析

**输出结果**：
- 13+ 性能对比图表
- 详细报告（JSON + Markdown）
- 所有结果保存在 `validation_results/` 目录

---

### 6. WebUI 训练界面 - `webui/app.py`

**用途**：可视化训练管理界面

**启动**：
```bash
python -m webui.app
```

**三种训练方案**：

**方案1：一键训练**
- 自动生成数据 → 自动训练
- 适合快速开始

**方案2：导入数据训练**（推荐）
- 使用已有数据
- 快速训练
- 更灵活

**方案3：仅生成数据**
- 单独运行 MATLAB 数据生成
- 生成后可用于后续训练

**功能**：
- 参数配置
- 实时训练进度
- 日志输出
- 结果保存

---

## 💻 使用流程示例

### 场景 1：快速体验（15分钟）
```bash
1. bash RUN.sh           # 一键启动
2. 打开 http://127.0.0.1:7861
3. 选择"方案2：导入数据训练"
4. 点击训练
```

### 场景 2：完整训练+验证（60分钟）
```bash
1. python train.py                    # 训练模型
2. python run_all_experiments.py --mode all  # 完整验证
3. 查看 validation_results/ 的结果
```

### 场景 3：生成新数据
```bash
1. matlab -batch "run('data_generator.m')"
2. python train.py                    # 用新数据训练
```

### 场景 4：开发新模型
```python
from model import CSIBERT

# 定义新模型
model = CSIBERT(feature_dim=1024, hidden_size=512)

# 使用现有训练脚本
# 或参考 train.py 实现自己的训练循环
```

---

## 📖 命令速查表

| 任务 | 命令 |
|------|------|
| 一键启动 | `bash RUN.sh` 或 `RUN.bat` |
| WebUI 训练 | `python -m webui.app` |
| 命令行训练 | `python train.py` |
| 快速验证 | `python model_validation.py` |
| 完整实验 | `python run_all_experiments.py --mode all` |
| 生成数据 | `matlab -batch "run('data_generator.m')"` |

---

## 🔧 环境配置

### 自动配置（推荐）
```bash
python setup_environment.py
```

### 手动配置
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## ⚠️ 常见问题

**Q：模型训练很慢？**
A：使用 GPU：`python train.py --device cuda`

**Q：如何修改模型配置？**
A：编辑 `train.py` 中的参数，或使用命令行参数

**Q：没有 MATLAB，如何生成数据？**
A：使用预先生成的数据，或参考 `data_generator.m` 用 Python 实现

**Q：WebUI 无法启动？**
A：检查端口 7861 是否被占用，或改为：`python -m webui.app --server_port 8000`

---

## 📚 相关文档

- **README.md** - 项目介绍
- **FILES.md** - 文件详细说明
- **TESTS.md** - 测试项目说明
