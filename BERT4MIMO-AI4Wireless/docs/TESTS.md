# 测试说明 - BERT4MIMO 验证和实验方法

##  所有测试概览

**总计**：13 个测试方法  
**基础验证**：5 个（2-5分钟）  
**高级实验**：8 个（8-25分钟）  

---

##  快速运行

### 仅运行基础验证
```bash
python model_validation.py
```
**耗时**：2-5分钟  
**输出**：5 个性能指标

### 仅运行高级实验
```bash
python run_all_experiments.py --mode advanced
```
**耗时**：8-25分钟  
**输出**：8 个实验结果

### 运行所有测试（推荐）
```bash
python run_all_experiments.py --mode all
```
**耗时**：10-30分钟  
**输出**：13 个完整测试结果

---

##  基础验证测试（5个）

### Test 1：重构误差测试
**文件**：`model_validation.py::test_reconstruction_error()`  
**目的**：衡量模型的基础重构能力  
**持续时间**：~1分钟

**测试配置**：
- 掩码比率：15%
- 批大小：32
- 样本数：100

**输出指标**：
- MSE（均方误差）
- NMSE（归一化均方误差，dB）
- MAE（平均绝对误差）

**成功标准**：
- NMSE > -10 dB

**图表**：
- `error_distribution.png` - 误差分布直方图

---

### Test 2：预测准确度测试
**文件**：`model_validation.py::test_prediction_accuracy()`  
**目的**：评估时序预测能力  
**持续时间**：~1分钟

**测试配置**：
- 预测步长：1, 3, 5, 10
- 历史长度：10 帧
- 样本数：50

**输出指标**：
- 各步长的 NMSE
- 预测准确度曲线
- 平均误差

**成功标准**：
- 短期预测（1-3步）NMSE > -8 dB

**图表**：
- `prediction_vs_steps.png` - 步长 vs 误差曲线

---

### Test 3：SNR 鲁棒性测试
**文件**：`model_validation.py::test_snr_robustness()`  
**目的**：测试在不同信噪比下的性能  
**持续时间**：~1分钟

**测试配置**：
- SNR 范围：-10 ~ 30 dB（11 个采样点）
- 掩码比率：15%
- 重复数：20

**输出指标**：
- 各 SNR 下的 NMSE
- 性能曲线
- 鲁棒性评分

**成功标准**：
- 低 SNR 下 NMSE 不降超过 3dB

**图表**：
- `snr_robustness.png` - SNR vs 性能曲线

---

### Test 4：压缩率测试
**文件**：`model_validation.py::test_compression_quality()`  
**目的**：衡量压缩与质量权衡  
**持续时间**：~1分钟

**测试配置**：
- 掩码比率：10%, 20%, ..., 90%
- 样本数：100
- 每个比率 10 次重复

**输出指标**：
- 各压缩率下的 NMSE
- 压缩比
- 质量评分

**成功标准**：
- 50% 压缩率下 NMSE > -8 dB

**图表**：
- `compression_quality.png` - 压缩率 vs 质量曲线

---

### Test 5：推理速度测试
**文件**：`model_validation.py::test_inference_speed()`  
**目的**：评估计算效率  
**持续时间**：~1分钟

**测试配置**：
- 批大小：1, 8, 16, 32, 64
- 重复数：100
- 记录时间：ms/frame

**输出指标**：
- 各批大小的推理时间
- 吞吐量（frames/sec）
- 平均延迟

**成功标准**：
- 批大小 32 时 < 50ms/frame

**图表**：
- `inference_speed.png` - 批大小 vs 速度曲线

---

##  高级实验（8个）

### Exp 1：掩码比率敏感性分析
**文件**：`experiments_extended.py::experiment_masking_ratio_sensitivity()`  
**来源**：Jupyter Notebook 实验 3  
**目的**：分析模型对掩码比率的敏感性  
**持续时间**：~3分钟

**测试配置**：
- 掩码比率范围：0% ~ 50%（30个间隔）
- 每个比率重复：20 次
- 样本数：200

**输出指标**：
- 敏感性曲线
- 临界点分析
- 性能下降率

**关键发现**：
- 识别最优掩码比率
- 性能下降的加速点

**图表**：
- `masking_ratio_vs_mse.png` - 掩码比率 vs MSE

---

### Exp 2：场景性能分析
**文件**：`experiments_extended.py::experiment_scenario_wise_performance()`  
**来源**：Jupyter Notebook 实验 2  
**目的**：评估不同场景下的性能  
**持续时间**：~2分钟

**测试场景**：
1. Stationary（静止）- 低多普勒
2. High-Speed（高速）- 高多普勒
3. Urban Macro（城市宏站）- 混合

**输出指标**：
- 各场景的 NMSE
- 场景间的性能差异
- 泛化能力评估

**关键发现**：
- 最具挑战的场景
- 模型适应性

**图表**：
- `scenario_performance.png` - 场景性能对比条形图

---

### Exp 3：子载波性能测试
**文件**：`experiments_extended.py::experiment_subcarrier_performance()`  
**来源**：Jupyter Notebook 实验 5  
**目的**：分析频域性能分布  
**持续时间**：~2分钟

**测试配置**：
- 子载波分组：8 组（每组 8 个子载波）
- 样本数：100
- 指标：MSE, 标准差, 最大误差

**输出指标**：
- 各组的 MSE
- 标准差分布
- 最大误差

**关键发现**：
- 频域性能不均匀性
- 高风险子载波识别

**图表**：
- `subcarrier_performance.png` - 3 张子图（MSE/Std/Max）

---

### Exp 4：多普勒移位鲁棒性
**文件**：`experiments_extended.py::experiment_doppler_shift_robustness()`  
**来源**：Jupyter Notebook 实验 9  
**目的**：测试在不同多普勒移位下的性能  
**持续时间**：~3分钟

**测试配置**：
- 多普勒范围：50 ~ 400 Hz（20个间隔）
- 每个多普勒重复：20 次
- 样本数：100

**输出指标**：
- 多普勒 vs NMSE 曲线
- 鲁棒性评分
- 性能下降率

**关键发现**：
- 最大可承受多普勒
- 性能衰减特性

**图表**：
- `doppler_robustness.png` - 多普勒 vs 性能曲线

---

### Exp 5：跨场景泛化测试
**文件**：`experiments_extended.py::experiment_cross_scenario_generalization()`  
**来源**：Jupyter Notebook 实验 10  
**目的**：评估跨场景泛化能力  
**持续时间**：~3分钟

**测试配置**：
- 场景对：3×3（9种组合）
  - 训练场景 vs 测试场景
- 重复数：20

**输出指标**：
- 泛化热图 (3×3 矩阵)
- 对角线值：同场景性能
- 非对角线值：跨场景性能
- 泛化指数

**关键发现**：
- 场景之间的可转移性
- 最具挑战的跨场景转移

**图表**：
- `generalization_heatmap.png` - 3×3 热力图

---

### Exp 6：基线模型对比
**文件**：`experiments_extended.py::experiment_baseline_comparison()`  
**来源**：Jupyter Notebook 实验 8  
**目的**：与传统方法对比  
**持续时间**：~2分钟

**对比模型**：
1. CSIBERT（本项目）
2. 线性回归（Linear Regression）
3. MLP（多层感知机）

**输出指标**：
- 各模型的 NMSE
- 性能提升百分比
- 复杂度对比

**关键发现**：
- CSIBERT 相对提升
- 性能与复杂度权衡

**图表**：
- `baseline_comparison.png` - 模型性能对比条形图

---

### Exp 7：注意力机制可视化
**文件**：`experiments_extended.py::experiment_attention_visualization()`  
**来源**：Jupyter Notebook 实验 4  
**目的**：理解模型学到的注意力模式  
**持续时间**：~2分钟

**可视化内容**：
- 注意力权重热图
- 多头注意力分布
- 层间注意力演化

**输出指标**：
- 注意力权重分布
- 注意力集中度
- 关键位置识别

**关键发现**：
- 模型重点关注的频域/时域特征
- 多头的分工方式

**图表**：
- `attention_weights.png` - 多张热力图

---

### Exp 8：错误分布分析
**文件**：`experiments_extended.py::experiment_error_distribution()`  
**来源**：Jupyter Notebook 实验 6  
**目的**：分析错误的统计特性  
**持续时间**：~2分钟

**分析内容**：
- 多子载波组的错误分布
- 错误的统计特性
- 异常值识别

**输出指标**：
- 错误分布直方图
- 统计参数（均值、方差、偏度）
- 异常值比例

**关键发现**：
- 错误分布类型
- 高风险子载波识别

**图表**：
- `error_distribution.png` - 多张直方图

---

##  测试结果解释

### 性能指标说明

| 指标 | 单位 | 越小越好 | 说明 |
|------|------|---------|------|
| MSE | - | ✓ | 均方误差，衡量重构精度 |
| NMSE | dB | ✓ | > -10 dB 为优秀，-5～-10 为良好 |
| MAE | - | ✓ | 平均绝对误差 |
| 推理时间 | ms | ✓ | 低延迟是实时应用的关键 |
| 吞吐量 | frames/sec | ✗ | 越大越好，吞吐能力 |
| 鲁棒性 | - | ✗ | 越高越好，对干扰的抗性 |

---

##  测试选择指南

| 需求 | 推荐测试 | 耗时 |
|------|---------|------|
| 快速验证 | Test 1-5 | 2-5分钟 |
| 性能评估 | Exp 1-3, 6 | 8分钟 |
| 可靠性评估 | Exp 4-5 | 6分钟 |
| 完整评估 | 全部 | 10-30分钟 |
| 模型分析 | Exp 7-8 | 4分钟 |

---

##  输出结果

### 生成的文件
```
validation_results/
├── VALIDATION_REPORT.md           # 总结报告
├── validation_report.json         # 基础验证详细数据
├── advanced_experiments_summary.json  # 高级实验汇总
├── error_distribution.png
├── prediction_vs_steps.png
├── snr_robustness.png
├── compression_quality.png
├── inference_speed.png
├── masking_ratio_vs_mse.png
├── scenario_performance.png
├── subcarrier_performance.png
├── doppler_robustness.png
├── generalization_heatmap.png
├── baseline_comparison.png
├── attention_weights.png
└── error_distribution_analysis.png
```

### 结果文件格式

**JSON 格式**（detailed data）：
```json
{
  "test_name": "Test 1: Reconstruction Error",
  "timestamp": "2025-11-14T10:30:00",
  "metrics": {
    "mse": 0.0234,
    "nmse_db": -16.3,
    "mae": 0.0125
  },
  "samples": 100,
  "status": "PASS"
}
```

**Markdown 格式**（summary report）：
```markdown
## Test Results Summary

| Test Name | NMSE (dB) | Status |
|-----------|-----------|--------|
| Test 1 | -16.3 | ✓ PASS |
| Test 2 | -14.1 | ✓ PASS |
...
```

---

##  如何自定义测试

### 修改测试参数
编辑 `model_validation.py` 或 `experiments_extended.py` 中的参数：

```python
# 例如：修改掩码比率范围
mask_ratios = np.linspace(0.0, 0.8, 100)  # 更多间隔
num_trials = 50  # 更多重复
```

### 添加新测试
继承 `CSIBERTValidator` 类，添加新的测试方法：

```python
def test_custom_metric(self):
    """自定义测试"""
    # 实现测试逻辑
    pass
```

---

##  相关文档

- **README.md** - 项目介绍
- **USAGE.md** - 使用指南
- **FILES.md** - 文件说明
