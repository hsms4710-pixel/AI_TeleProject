#!/usr/bin/env python3
"""
CSIBERT 模型训练脚本 / CSIBERT Model Training Script

主要功能 / Main Features:
- 加载和预处理 CSI 数据 / Load and preprocess CSI data
- 数据拆分（训练/验证/测试集）/ Data split (train/validation/test)
- 模型训练与验证 / Model training and validation
- 保存最佳模型 / Save best model
- 生成训练曲线 / Generate training curves

使用方法 / Usage:
    python train.py --hidden_size 256 --num_epochs 50 --batch_size 16
"""

import os
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split

from model import CSIBERT


# 检测可用设备 / Detect available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 Apple MPS")
else:
    device = torch.device("cpu")
    print("使用 CPU")


def load_csi_data(data_path='foundation_model_data/csi_data_massive_mimo.mat'):
    """
    加载 CSI 数据 / Load CSI data
    
    Args:
        data_path: .mat 文件路径 / Path to .mat file
        
    Returns:
        cell_data: 原始 CSI 数据 / Raw CSI data
    """
    print(f"📂 加载数据: {data_path}")
    cell_data = scipy.io.loadmat(data_path)['multi_cell_csi']
    print(f"   数据形状: {cell_data.shape}")
    return cell_data


def preprocess_csi_matrix(csi_matrix):
    """
    预处理单个 CSI 矩阵 / Preprocess a single CSI matrix
    
    处理步骤 / Processing steps:
    1. 分离实部和虚部 / Separate real and imaginary parts
    2. 标准化 / Normalize
    3. 展平为特征向量 / Flatten to feature vector
    
    Args:
        csi_matrix: 复数 CSI 矩阵 / Complex CSI matrix
        
    Returns:
        csi_combined: 预处理后的 CSI (time, feature_dim)
    """
    # 分离实部和虚部 / Separate real and imaginary parts
    csi_real = np.real(csi_matrix)
    csi_imag = np.imag(csi_matrix)
    
    # 标准化 / Normalize
    csi_real_normalized = (csi_real - np.mean(csi_real)) / (np.std(csi_real) + 1e-8)
    csi_imag_normalized = (csi_imag - np.mean(csi_imag)) / (np.std(csi_imag) + 1e-8)

    # 组合并展平 / Combine and flatten
    csi_combined = np.stack([csi_real_normalized, csi_imag_normalized], axis=-1)
    time_dim = csi_combined.shape[0]
    feature_dim = np.prod(csi_combined.shape[1:])
    csi_combined = csi_combined.reshape(time_dim, feature_dim)
    
    return csi_combined


def mask_data(data, mask_ratio=0.15):
    """
    对数据应用掩码（用于自监督学习）/ Apply masking to data (for self-supervised learning)
    
    Args:
        data: 输入数据 / Input data
        mask_ratio: 掩码比例 / Mask ratio
        
    Returns:
        masked_data: 掩码后的数据 / Masked data
        mask: 掩码位置 / Mask positions
    """
    mask = np.random.rand(*data.shape) < mask_ratio
    masked_data = np.copy(data)
    masked_data[mask] = 0
    return masked_data, mask


def create_dataloader(data, batch_size, shuffle=True):
    """
    创建 DataLoader / Create DataLoader
    
    对每个样本进行掩码处理，并创建 DataLoader
    
    Args:
        data: 数据列表 / List of data samples
        batch_size: 批次大小 / Batch size
        shuffle: 是否打乱 / Whether to shuffle
        
    Returns:
        DataLoader 对象
    """
    # 对数据进行掩码 / Apply masking
    masked_data, masks = zip(*[mask_data(d) for d in data])
    
    # 填充序列到相同长度 / Pad sequences to same length
    max_len = max(len(d) for d in data)
    feature_dim = data[0].shape[1]
    
    padded_inputs = np.zeros((len(data), max_len, feature_dim), dtype=np.float32)
    padded_labels = np.zeros((len(data), max_len, feature_dim), dtype=np.float32)
    attention_masks = np.zeros((len(data), max_len), dtype=np.float32)
    
    for i, (masked, original) in enumerate(zip(masked_data, data)):
        seq_len = len(original)
        padded_inputs[i, :seq_len, :] = masked
        padded_labels[i, :seq_len, :] = original
        attention_masks[i, :seq_len] = 1
    
    # 转换为 PyTorch 张量 / Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(padded_inputs).float()
    labels_tensor = torch.from_numpy(padded_labels).float()
    masks_tensor = torch.from_numpy(attention_masks).float()
    
    dataset = TensorDataset(inputs_tensor, labels_tensor, masks_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main(hidden_size, num_hidden_layers, num_attention_heads, learning_rate, batch_size, num_epochs, patience):
    """
    主训练函数 / Main training function
    
    Args:
        patience: 早停耐心值，验证损失连续多少轮不改善则停止训练。设为0则禁用早停。
    """
    print("\n" + "="*70)
    print("🚀 CSIBERT 训练流程开始 / CSIBERT Training Pipeline Started")
    print("="*70)
    
    # 1️⃣ 加载并预处理数据 / Load and preprocess data
    cell_data = load_csi_data()
    
    preprocessed_data = []
    print("🔄 预处理数据中...")
    
    for cell_idx in range(cell_data.shape[0]):
        for ue_idx in range(cell_data.shape[1]):
            ue_data = cell_data[cell_idx, ue_idx]
            for scenario in ue_data[0]:
                processed_csi = preprocess_csi_matrix(scenario)
                preprocessed_data.append(processed_csi)
    
    print(f"   ✓ 预处理完成，总样本数: {len(preprocessed_data)}")
    
    # 2️⃣ 数据拆分 / Data split
    print("\n📊 数据拆分:")
    # 先分出测试集 (20%)
    train_val_data, test_data = train_test_split(
        preprocessed_data, test_size=0.2, random_state=42
    )
    # 再从剩余数据中分出验证集 (10% of total = 12.5% of train_val)
    train_data, val_data = train_test_split(
        train_val_data, test_size=0.125, random_state=42
    )
    
    print(f"   训练集: {len(train_data)} 样本 (70%)")
    print(f"   验证集: {len(val_data)} 样本 (10%)")
    print(f"   测试集: {len(test_data)} 样本 (20%)")
    
    # 保存测试集供后续验证使用 / Save test set for later validation
    os.makedirs('validation_data', exist_ok=True)
    np.save('validation_data/test_data.npy', np.array(test_data, dtype=object))
    print(f"   ✓ 测试集已保存至 validation_data/test_data.npy")
    
    # 3️⃣ 创建 DataLoader / Create DataLoaders
    print("\n🔧 创建数据加载器...")
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader)}")
    
    # 4️⃣ 初始化模型 / Initialize model
    feature_dim = preprocessed_data[0].shape[1]
    print(f"\n🧠 初始化模型:")
    print(f"   特征维度: {feature_dim}")
    print(f"   隐藏层大小: {hidden_size}")
    print(f"   Transformer 层数: {num_hidden_layers}")
    print(f"   注意力头数: {num_attention_heads}")
    
    model = CSIBERT(
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads
    ).to(device)
    
    # 5️⃣ 初始化优化器和学习率调度器 / Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,  # 10% warmup
        num_training_steps=num_training_steps
    )
    
    loss_function = nn.MSELoss()
    
    # 6️⃣ 训练循环 / Training loop
    print(f"\n🎯 开始训练:")
    print(f"   最大轮数: {num_epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {learning_rate}")
    if patience > 0:
        print(f"   早停耐心值: {patience}")
    else:
        print(f"   早停: 禁用")
    print(f"   设备: {device}")
    print("="*70 + "\n")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段 / Training phase
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [训练]")
        for batch in progress_bar:
            inputs, labels, attention_mask = [b.to(device) for b in batch]
            
            # 前向传播 / Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            loss = loss_function(outputs, labels)
            
            # 反向传播 / Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段 / Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, attention_mask = [b.to(device) for b in batch]
                outputs = model(inputs, attention_mask=attention_mask)
                loss = loss_function(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 打印当前轮次结果 / Print current epoch results
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", end="")
        
        # 保存最佳模型 / Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # 重置早停计数器
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'feature_dim': feature_dim,
                'hidden_size': hidden_size,
                'num_hidden_layers': num_hidden_layers,
                'num_attention_heads': num_attention_heads
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, 'checkpoints/best_model.pt')
            print(" ✓ [已保存最佳模型]")
        else:
            patience_counter += 1
            if patience > 0:
                print(f" (未改善: {patience_counter}/{patience})")
            else:
                print()
        
        # 早停检查 / Early stopping check
        if patience > 0 and patience_counter >= patience:
            print(f"\n🛑 早停触发！验证损失连续 {patience} 轮未改善")
            print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
            print(f"   训练在第 {epoch+1} 轮停止")
            break
    
    # 7️⃣ 绘制训练曲线 / Plot training curves
    print("\n📈 生成训练曲线...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ 训练曲线已保存至 training_validation_loss.png")
    
    # 8️⃣ 保存损失历史 / Save loss history
    np.savez('training_history.npz', 
             train_losses=train_losses, 
             val_losses=val_losses,
             best_val_loss=best_val_loss)
    print(f"   ✓ 训练历史已保存至 training_history.npz")
    
    print("\n" + "="*70)
    print(f"✅ 训练完成！")
    print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"   实际训练轮数: {len(train_losses)}/{num_epochs}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 CSIBERT 模型 / Train CSIBERT Model')
    
    # 模型参数 / Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层大小 / Hidden layer size (default: 256)')
    parser.add_argument('--num_hidden_layers', type=int, default=4,
                        help='Transformer 层数 / Number of Transformer layers (default: 4)')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='注意力头数 / Number of attention heads (default: 4)')
    
    # 训练参数 / Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率 / Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 / Batch size (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数 / Number of epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值 / Early stopping patience (default: 15, 0=disable)')
    
    args = parser.parse_args()
    
    # 执行训练 / Execute training
    main(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
