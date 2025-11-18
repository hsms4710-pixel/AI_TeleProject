#!/usr/bin/env python3
"""
CSIBERT 模型定义 / CSIBERT Model Definition

This module defines the CSIBERT neural network architecture based on BERT.
CSIBERT is designed for Channel State Information (CSI) prediction and reconstruction
in MIMO wireless systems.

模型架构 / Model Architecture:
- Time Embedding: 学习时间维度的表征 / Learn temporal patterns
- Feature Embedding: 将CSI特征映射到隐藏空间 / Map CSI features to hidden space
- BERT Encoder: 多层Transformer编码器 / Multi-layer Transformer encoder
- Output Layer: 重构CSI矩阵 / Reconstruct CSI matrix
"""

import torch
from torch import nn
from transformers import BertConfig, BertModel


class CSIBERT(nn.Module):
    """
    CSIBERT: BERT-based model for CSI prediction
    
    基于BERT的信道状态信息预测模型
    使用自注意力机制捕捉CSI数据的时间和空间依赖关系
    
    Args:
        feature_dim (int): CSI特征维度 / Dimension of CSI features
        hidden_size (int): 隐藏层大小 / Hidden layer size (default: 256)
        num_hidden_layers (int): Transformer层数 / Number of Transformer layers (default: 4)
        num_attention_heads (int): 注意力头数量 / Number of attention heads (default: 4)
    """
    def __init__(self, feature_dim, hidden_size=256, num_hidden_layers=4, num_attention_heads=4):
        super(CSIBERT, self).__init__()
        self.hidden_size = hidden_size
        
        # 配置BERT模型参数 / Configure BERT model parameters
        self.config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,  # 标准FFN扩展比例 / Standard FFN expansion ratio
            max_position_embeddings=4096  # 最大序列长度 / Maximum sequence length
        )
        
        # BERT编码器核心 / BERT encoder core
        self.bert = BertModel(self.config)

        # 嵌入层 / Embedding layers
        # 时间嵌入：学习时间步的位置信息 / Time embedding: learn temporal position info
        self.time_embedding = nn.Embedding(1024, hidden_size)
        # 特征嵌入：将CSI特征映射到隐藏空间 / Feature embedding: map CSI to hidden space
        self.feature_embedding = nn.Linear(feature_dim, hidden_size)

        # 输出层：重构原始CSI特征 / Output layer: reconstruct original CSI features
        self.output_layer = nn.Linear(hidden_size, feature_dim)


    def forward(self, inputs, attention_mask=None, output_attentions=False):
        """
        前向传播 / Forward pass
        
        Args:
            inputs: 输入CSI数据 (batch_size, sequence_length, feature_dim)
            attention_mask: 注意力掩码，标记有效token (batch_size, sequence_length)
            output_attentions: 是否返回注意力权重 / Whether to return attention weights
            
        Returns:
            predictions: 预测的CSI数据 (batch_size, sequence_length, feature_dim)
            attentions: (可选) 注意力权重 / (Optional) Attention weights
        """
        batch_size, sequence_length, feature_dim = inputs.shape

        # 1. 生成时间嵌入 / Generate time embeddings
        time_indices = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
        time_embeds = self.time_embedding(time_indices).expand(batch_size, -1, -1)

        # 2. 生成特征嵌入 / Generate feature embeddings
        feature_embeds = self.feature_embedding(inputs)

        # 3. 组合嵌入（相加融合时间和特征信息）/ Combine embeddings (add time and feature info)
        combined_embeds = time_embeds + feature_embeds

        # 4. 通过BERT编码器 / Pass through BERT encoder
        outputs = self.bert(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        # 5. 提取最后一层隐藏状态 / Extract last hidden state
        hidden_states = outputs.last_hidden_state

        # 6. 通过输出层重构CSI / Reconstruct CSI through output layer
        predictions = self.output_layer(hidden_states)

        # 7. 返回预测结果（和可选的注意力权重）/ Return predictions (and optional attention weights)
        if output_attentions:
            return predictions, outputs.attentions
        
        return predictions

