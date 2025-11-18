# CSIBERT 数据处理流程详解

> **最后更新**: 2025年11月18日  
> **目标**: 解释原始 `.mat` CSI 数据如何一步步转换为 CSIBERT 模型可以处理的 PyTorch 张量。

---

## 流程图

```mermaid
graph TD
    A[1. 原始数据 <br> csi_data_massive_mimo.mat] --> B{2. 加载与提取 <br> scipy.io.loadmat};
    B --> C[3. 复数矩阵 <br> (num_subcarriers, num_antennas) <br> dtype=complex];
    C --> D{4. 通道分离 <br> Real/Imaginary Parts};
    D --> E[5. 实数张量 <br> (2, num_subcarriers, num_antennas) <br> dtype=float];
    E --> F{6. 序列化 <br> Flatten subcarriers & antennas};
    F --> G[7. 序列向量 <br> (2, seq_len) <br> seq_len = 4096];
    G --> H{8. 归一化 & 维度重排};
    H --> I[9. 模型输入张量 <br> (seq_len, batch_size, 2)];
    I --> J{10. 添加位置编码 <br> nn.PositionalEncoding};
    J --> K[11. 输入 Transformer <br> CSIBERT.forward()];

    subgraph "预处理 (Python)"
        B; C; D; E; F; G; H;
    end

    subgraph "模型内部 (PyTorch)"
        I; J; K;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 详细步骤

### 1. 原始数据格式

- **文件**: `foundation_model_data/csi_data_massive_mimo.mat`
- **生成工具**: `data_generator.m` (MATLAB)
- **结构**: MATLAB `cell` 数组，大小为 `10x1`。每个 cell 代表一个用户，包含 `(64, 64, 200)` 的复数矩阵。
  - **维度 1**: 64 个子载波 (Subcarriers)
  - **维度 2**: 64 个基站天线 (BS Antennas)
  - **维度 3**: 200 个时间步 (Time Steps)
- **数据类型**: 复数 (Complex Float)

### 2. 加载与提取 (`_load_and_preprocess_data` in `train.py`)

```python
import scipy.io as sio
import numpy as np

# 加载 .mat 文件
mat_data = sio.loadmat('foundation_model_data/csi_data_massive_mimo.mat')
csi_data_cells = mat_data['csi_data']

# 提取并拼接所有样本
all_samples = []
# csi_data_cells.shape -> (10, 1)
for user_cell in csi_data_cells:
    # user_csi.shape -> (64, 64, 200)
    user_csi = user_cell[0] 
    # 沿时间步维度拆分，得到 200 个样本
    for i in range(user_csi.shape[2]):
        sample = user_csi[:, :, i] # shape (64, 64)
        all_samples.append(sample)

# all_samples 是一个包含 2000 个 (64, 64) 复数矩阵的列表
csi_data_complex = np.array(all_samples) 
# csi_data_complex.shape -> (2000, 64, 64)
```

### 3. 通道分离 (Complex to Real)

神经网络不直接处理复数。标准做法是将实部和虚部作为两个独立的通道。

```python
# 分离实部和虚部
csi_real = np.real(csi_data_complex)
csi_imag = np.imag(csi_data_complex)

# 将实部和虚部堆叠为两个通道
# shape: (2000, 2, 64, 64)
csi_data_stacked = np.stack([csi_real, csi_imag], axis=1)
```
- `axis=1` 表示在样本数之后、子载波之前插入一个新的维度作为通道维。
- 此时的数据格式非常类似于计算机视觉中的图像数据 `(N, C, H, W)`，其中 `C=2`。

### 4. 序列化 (Flattening)

Transformer 的输入是序列。我们将 `(64, 64)` 的空间/频率平面展平，形成一个长序列。

```python
# 获取维度信息
num_samples, num_channels, num_subcarriers, num_antennas = csi_data_stacked.shape

# 展平后两个维度
# shape: (2000, 2, 4096)
# 4096 = 64 * 64
csi_data_reshaped = csi_data_stacked.reshape(num_samples, num_channels, -1)
```
- `-1` 告诉 NumPy 自动计算该维度的大小。
- 现在，每个样本都是一个 `(2, 4096)` 的矩阵，代表一个长度为 4096、每个 token 有 2 个特征（实部、虚部）的序列。

### 5. 归一化与维度重排

为了符合 PyTorch `nn.Transformer` 的 `(S, N, E)` 输入格式，我们需要进行维度重排。

- **S**: Sequence Length (序列长度)
- **N**: Batch Size (批大小)
- **E**: Embedding Dimension (特征维度)

```python
# 维度重排
# from (N, E, S) to (S, N, E)
# N=2000, E=2, S=4096
# target shape: (4096, 2000, 2)
csi_data_transposed = np.transpose(csi_data_reshaped, (2, 0, 1))

# 数据归一化 (示例: Min-Max Scaling)
# 实际代码中可能更复杂
min_val = np.min(csi_data_transposed)
max_val = np.max(csi_data_transposed)
normalized_data = (csi_data_transposed - min_val) / (max_val - min_val)
```

### 6. 转换为 PyTorch 张量

最后一步是转换为 PyTorch 张量，并划分数据集。

```python
import torch
from sklearn.model_selection import train_test_split

# 转换为张量
data_tensor = torch.from_numpy(normalized_data).float()

# 划分数据集 (在 train.py 中实现)
# train_data, val_data, test_data = ...
```

### 7. 模型输入与位置编码

当一个批次的数据（例如 `batch_size=64`）送入模型时：

- **输入形状**: `(4096, 64, 2)`
- **模型内部**:
  1.  数据首先通过一个线性层（`nn.Linear`），将特征维度从 2 映射到模型的隐藏维度 `hidden_size` (例如 256)。
      - `(4096, 64, 2)` -> `(4096, 64, 256)`
  2.  然后，与一个预先计算好的**位置编码 (Positional Encoding)** 张量相加。这个位置编码张量形状也是 `(4096, 1, 256)`，它为序列中的每个位置提供独一无二的“位置信号”。
  3.  添加了位置信息的张量 `(4096, 64, 256)` 才被送入 Transformer 的多头自注意力层进行计算。

---

## 总结

整个流程将一个结构化的多维复数矩阵，通过**通道分离**、**序列化**和**维度重排**，转换成了一个适合 Transformer 处理的、带有批次维度的长序列张量。**位置编码**则在模型内部弥补了序列化过程中损失的空间/频率结构信息。