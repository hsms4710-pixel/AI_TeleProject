#!/usr/bin/env python3
"""
CSIBERT 模型训练脚本

主要功能:
- 加载 CSI 数据
- 数据预处理（归一化、填充、掩码）
- 模型训练
- 检查点保存
"""

import scipy.io
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW  # Use PyTorch's AdamW

from transformers import get_scheduler
from torch import nn

import matplotlib.pyplot as plt

from tqdm import tqdm

import os

from model import CSIBERT

# Check MPS availability
if torch.backends.mps.is_available():
    print("MPS backend is available!")
else:
    print("MPS backend is not available.")

# Check for available device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load CSI Data
# csi_macro = heterogeneous_data['csiMacro']
# csi_small_cell = heterogeneous_data['csiSmallCell']

cell_data = scipy.io.loadmat('foundation_model_data/csi_data_massive_mimo.mat')['multi_cell_csi']

# Function to preprocess individual CSI matrices
def preprocess_csi_matrix(csi_matrix):
    """
    Preprocess CSI data to handle variable-length sequences and retain time, frequency, and spatial dimensions.
    """
    csi_real = np.real(csi_matrix)
    csi_imag = np.imag(csi_matrix)
    
    # Normalize real and imaginary parts
    csi_real_normalized = (csi_real - np.mean(csi_real)) / np.std(csi_real)
    csi_imag_normalized = (csi_imag - np.mean(csi_imag)) / np.std(csi_imag)

    # Combine real and imaginary components
    csi_combined = np.stack([csi_real_normalized, csi_imag_normalized], axis=-1)  # Shape: (subcarriers, Tx, Rx, 2)
    
    # Flatten freq, spatial, and real/imaginary into a single feature dimension
    time_dim = csi_combined.shape[0]
    feature_dim = np.prod(csi_combined.shape[1:])  # Tx × Rx × 2
    csi_combined = csi_combined.reshape(time_dim, feature_dim)  # Shape: (time, feature_dim)
    
    return csi_combined

# Traverse the nested cell structure
preprocessed_data = []
sequence_lengths = []

for cell_idx in range(cell_data.shape[0]):  # Iterate over cells
    for ue_idx in range(cell_data.shape[1]):  # Iterate over UEs
        ue_data = cell_data[cell_idx, ue_idx]
        for scenario in ue_data[0]:  # Each UE has multiple scenarios
            processed_csi = preprocess_csi_matrix(scenario)
            preprocessed_data.append(processed_csi)
            sequence_lengths.append(processed_csi.shape[0])  # Track sequence lengths

# Convert to padded 3D array (batch_size, sequence_length, feature_dim)
max_sequence_length = max(sequence_lengths)
feature_dim = preprocessed_data[0].shape[-1]

# Pad sequences dynamically
padded_data = np.zeros((len(preprocessed_data), max_sequence_length, feature_dim), dtype=np.float32)
attention_masks = np.zeros((len(preprocessed_data), max_sequence_length), dtype=np.float32)
for i, sequence in enumerate(preprocessed_data):
    seq_len = sequence.shape[0]
    padded_data[i, :seq_len, :] = sequence
    attention_masks[i, :seq_len] = 1  # Mask for unpadded tokens

# Mask data for masked signal prediction task
def mask_data(data, mask_ratio=0.15):
    mask = np.random.rand(*data.shape[:-1]) < mask_ratio  # Exclude the last dimension (real/imaginary parts)
    masked_data = np.copy(data)
    masked_data[mask, :] = 0  # Replace masked elements with 0
    return masked_data, mask

masked_data, mask = mask_data(padded_data)

# Ensure consistent data types
masked_data = masked_data.astype(np.float32)
attention_masks = torch.tensor(attention_masks, dtype=torch.float32).to(device)

print("Cell data shape:", cell_data.shape)

batch_size = 32  # Larger batch size for efficiency

train_dataset = TensorDataset(
    torch.tensor(masked_data).float(),  # Masked inputs
    torch.tensor(padded_data).float(),  # Labels (original data)
    torch.tensor(attention_masks).float()  # Attention masks
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define a directory to save the best model
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize Model and Optimizer
print("=" * 50)
print("轻量级训练配置:")
print(f"  hidden_size: 256")
print(f"  num_hidden_layers: 4")
print(f"  num_attention_heads: 4")
print(f"  batch_size: {batch_size}")
print(f"  learning_rate: 2e-4")
print("=" * 50)

model = CSIBERT(
    feature_dim=feature_dim,
    hidden_size=256,           # 轻量级: 256 (原为 768)
    num_hidden_layers=4,       # 轻量级: 4 层 (原为 6)
    num_attention_heads=4      # 轻量级: 4 头 (原为 6)
)
optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)  # 轻量级: 更高的学习率

lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=300, num_training_steps=5000  # 轻量级: 减少步数
)

loss_fn = nn.MSELoss()

model = model.to(device)

# Variables to track the best model
best_loss = float('inf')
patience = 10  # Early stopping patience
patience_counter = 0

loss_values = []  # Track loss over iterations
max_epochs = 200  # Maximum training epochs

print(f"\n开始训练... (最大轮数: {max_epochs}, patience: {patience})")
print(f"训练数据: {len(train_loader)} batches")

# In[10]:
for epoch in tqdm(range(max_epochs)):  # 轻量级: 从 1000000 改为 200
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        inputs, labels, attention_mask = batch  # Add attention mask
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, attention_mask=attention_mask)
        
        # Compute loss
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        lr_scheduler.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
    
    # Average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_loss)
    
    # Check if this is the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0  # Reset patience counter
        # Save the best model with all configuration
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "feature_dim": feature_dim,  # Save feature_dim
            "hidden_size": 256,  # Save model config
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"New best model saved with loss {avg_loss:.10f} at epoch {epoch + 1}")
    else:
        patience_counter += 1
    
    # Check if early stopping criteria are met
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.10f}")
        break

# Visualize training loss
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve')
plt.grid()
plt.show()

import pandas as pd

# Create a pandas DataFrame
df_loss_vals = pd.DataFrame({
    'Epoch': range(1, len(loss_values) + 1),
    'Loss': loss_values
})

# Save to CSV
df_loss_vals.to_csv('loss_values-12-layers.csv', index=False)

print(df_loss_vals)
