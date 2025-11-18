# 🚀 BERT4MIMO 快速入门指南

## 30秒启动

### Windows 用户

1. **双击运行** `START.bat`
2. **浏览器自动打开** http://127.0.0.1:7861
3. **开始使用** WebUI 图形界面

### Linux/Mac 用户

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 WebUI
python webui/app.py

# 3. 浏览器访问
# http://127.0.0.1:7861
```

---

## 第一次使用？

### Step 1: 一键训练

1. 打开 WebUI
2. 点击 **" 一键训练"** 标签页
3. 选择 **"轻量化配置⚡"**（5分钟完成）
4. 点击 **" 开始一键训练"**
5. 等待训练完成

### Step 2: 查看结果

训练完成后，自动生成：
- ✅ 模型文件：`checkpoints/best_model.pt`
- ✅ 损失曲线：`training_validation_loss.png`
- ✅ 测试数据：`validation_data/test_data.npy`

### Step 3: 运行实验

1. 点击 **" 进行实验"** 标签页
2. 选择 **"基础实验"**
3. 选择 **"All Basic Tests - 运行所有基础实验"**
4. 点击 **" 运行基础实验"**
5. 查看实验结果和生成的图表

---

## 配置建议

### 电脑配置参考

| 你的显卡 | 推荐方案 | 预期效果 |
|---------|---------|---------|
| 无独显 / 2GB显存 | 轻量化 | 85%精度，5分钟 |
| 4-8GB 显存 | 标准 | 92%精度，15分钟 |
| 16GB+ 显存 | 高性能 | 95%精度，30分钟 |

### 训练参数说明

| 参数 | 轻量级 | 标准 | 高性能 | 作用 |
|------|--------|------|--------|------|
| Hidden Size | 256 | 256 | 512 | 模型容量 |
| Layers | 4 | 6 | 8 | 网络深度 |
| Heads | 4 | 8 | 8 | 注意力头数 |
| Epochs | 10 | 30 | 50 | 训练轮数 |
| Batch Size | 16 | 32 | 32 | 批次大小 |

---

## 功能导航

###  一键训练
- **最快上手**：选择预设方案，一键开始
- **适合人群**：新手用户、快速验证

### 📥 导入数据训练
- **自定义参数**：精细调整模型配置
- **适合人群**：有经验的用户

###  实验管理

#### 基础实验（5项）
1. **重构误差** - 评估模型基本性能
2. **预测准确度** - 测试时序预测能力
3. **SNR鲁棒性** - 噪声环境测试
4. **压缩质量** - 数据压缩效果
5. **推理速度** - 计算性能测试

#### 高级实验（5项）
1. **掩码比率敏感性** - 测试15种掩码比率
2. **误差分布分析** - 统计分析和可视化
3. **预测步长分析** - 1-20步预测能力
4. **基线方法对比** - 与传统方法比较
5. **注意力可视化** - 模型注意力热力图

---

## 结果查看

### 训练结果
- **位置**：`checkpoints/best_model.pt`
- **查看方式**：WebUI 自动加载，或通过 "模型管理" 标签页

### 验证结果
- **基础测试**：`validation_results/`
  - `validation_report.json` - 详细指标
  - `VALIDATION_REPORT.md` - 可读报告
  - `*.png` - 性能图表

- **高级实验**：`advanced_experiments/`
  - `exp1_masking_ratio.*` - 掩码敏感性
  - `exp2_error_stats.*` - 误差分布
  - `exp3_prediction_horizon.*` - 预测步长
  - `exp4_baseline_comparison.*` - 基线对比
  - `exp5_attention_sample_*.png` - 注意力图

---

## 常见问题

### Q1: 显存不足怎么办？
**A**: 降低参数
- Batch Size: 32 → 16 → 8
- Hidden Size: 512 → 256 → 128
- 使用 CPU 模式（自动检测）

### Q2: 训练很慢？
**A**: 检查设备
- 确认 GPU 可用：WebUI 启动时会显示设备信息
- 如果是 CPU 训练：选择轻量化配置

### Q3: 实验失败？
**A**: 确认前提条件
- 模型已训练：需要先完成训练
- 测试数据存在：`validation_data/test_data.npy`

### Q4: 如何提升精度？
**A**: 优化方案
1. 增加训练轮数（Epochs: 30 → 50 → 100）
2. 使用更大模型（Hidden: 256 → 512）
3. 增加网络深度（Layers: 4 → 6 → 8）
4. 降低学习率（Learning Rate: 1e-4 → 5e-5）

### Q5: 端口被占用？
**A**: 修改端口
编辑 `webui/app.py` 最后几行：
```python
app.launch(
    server_port=7861  # 改为其他端口，如 7862
)
```

---

## 命令行模式

如果你更喜欢命令行：

```bash
# 训练模型
python train.py --num_epochs 30 --batch_size 32

# 基础验证
python model_validation.py

# 高级实验
python experiments_extended.py --experiment 1

# 查看帮助
python train.py --help
python experiments_extended.py --help
```

---

## 更多资源

-  **WebUI 使用指南**：`docs/WEBUI_GUIDE.md`
-  **项目完整说明**：`README.md`
-  **源代码**：查看 `model.py`、`train.py` 等文件

---

## 技术支持

遇到问题？

1. 查看控制台错误信息
2. 阅读 `README.md` 和 `docs/WEBUI_GUIDE.md`
3. 确认依赖已安装：`pip install -r requirements.txt`
4. 检查 Python 版本：`python --version`（需要 3.8+）

---

