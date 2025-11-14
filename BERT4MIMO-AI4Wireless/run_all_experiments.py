#!/usr/bin/env python3
"""
完整的 CSIBERT 实验运行器

整合了 model_validation.py 的基础验证和 experiments_extended.py 的高级实验
支持运行所有高级实验方法
"""

import torch
import numpy as np
import scipy.io
import argparse
from pathlib import Path

from model_validation import CSIBERTValidator
from experiments_extended import AdvancedCSIBERTExperiments
from model import CSIBERT


def load_and_preprocess_data(data_path):
    """加载和预处理 CSI 数据"""
    print(f"📂 加载数据: {data_path}")
    
    cell_data = scipy.io.loadmat(data_path)['multi_cell_csi']
    
    def preprocess_csi_matrix(csi_matrix):
        csi_real = np.real(csi_matrix)
        csi_imag = np.imag(csi_matrix)
        csi_real_normalized = (csi_real - np.mean(csi_real)) / (np.std(csi_real) + 1e-8)
        csi_imag_normalized = (csi_imag - np.mean(csi_imag)) / (np.std(csi_imag) + 1e-8)
        csi_combined = np.stack([csi_real_normalized, csi_imag_normalized], axis=-1)
        time_dim = csi_combined.shape[0]
        feature_dim = np.prod(csi_combined.shape[1:])
        return csi_combined.reshape(time_dim, feature_dim)
    
    # 预处理数据
    preprocessed_data = []
    sequence_lengths = []
    for cell_idx in range(cell_data.shape[0]):
        for ue_idx in range(cell_data.shape[1]):
            ue_data = cell_data[cell_idx, ue_idx]
            for scenario in ue_data[0]:
                processed_csi = preprocess_csi_matrix(scenario)
                preprocessed_data.append(processed_csi)
                sequence_lengths.append(processed_csi.shape[0])
    
    # 填充数据
    max_sequence_length = max(sequence_lengths)
    feature_dim = preprocessed_data[0].shape[-1]
    
    padded_data = np.zeros((len(preprocessed_data), max_sequence_length, feature_dim), dtype=np.float32)
    attention_masks = np.zeros((len(preprocessed_data), max_sequence_length), dtype=np.float32)
    
    for i, sequence in enumerate(preprocessed_data):
        seq_len = sequence.shape[0]
        padded_data[i, :seq_len, :] = sequence
        attention_masks[i, :seq_len] = 1
    
    print(f"✅ 数据形状: {padded_data.shape}")
    return padded_data, attention_masks, feature_dim


def load_model(model_path, feature_dim, device):
    """加载训练好的模型"""
    print(f"🤖 加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 推断模型配置
    num_hidden_layers = checkpoint.get('num_hidden_layers', 
                                       checkpoint.get('model_config', {}).get('num_hidden_layers', 12))
    
    model = CSIBERT(
        feature_dim=feature_dim,
        num_hidden_layers=num_hidden_layers,
        hidden_size=checkpoint.get('model_config', {}).get('hidden_size', 256),
        num_attention_heads=checkpoint.get('model_config', {}).get('num_attention_heads', 4)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功 (layers={num_hidden_layers})")
    return model


def main():
    parser = argparse.ArgumentParser(description='CSIBERT 完整实验运行器')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='模型检查点路径')
    parser.add_argument('--data', type=str, 
                       default='foundation_model_data/csi_data_massive_mimo.mat',
                       help='CSI 数据文件路径')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='计算设备')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['basic', 'advanced', 'all'],
                       help='运行模式: basic(基础验证), advanced(高级实验), all(全部)')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("🚀 CSIBERT 完整实验运行器")
    print("="*70)
    print(f"设备: {device}")
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print("="*70 + "\n")
    
    # 加载数据和模型
    try:
        padded_data, attention_masks, feature_dim = load_and_preprocess_data(args.data)
        model = load_model(args.model, feature_dim, device)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 创建掩码数据
    masked_data = np.copy(padded_data)
    masked_data[:, ::10, :] = 0  # 掩码每第10个样本
    
    # 运行基础验证 (使用 model_validation.py)
    if args.mode in ['basic', 'all']:
        print("\n" + "="*70)
        print("📋 运行基础验证测试")
        print("="*70 + "\n")
        
        try:
            validator = CSIBERTValidator(args.model, args.data, device=device)
            validator.run_all_tests()
        except Exception as e:
            print(f"❌ 基础验证失败: {e}")
    
    # 运行高级实验 (使用 experiments_extended.py)
    if args.mode in ['advanced', 'all']:
        print("\n" + "="*70)
        print("🔬 运行高级实验")
        print("="*70 + "\n")
        
        try:
            experiments = AdvancedCSIBERTExperiments(
                model=model,
                padded_data=padded_data,
                masked_data=masked_data,
                feature_dim=feature_dim,
                device=device,
                attention_masks=attention_masks,
                output_dir=args.output
            )
            
            results = experiments.run_all_advanced_experiments()
            
            # 保存结果总结
            import json
            with open(f"{args.output}/advanced_experiments_summary.json", 'w') as f:
                # 转换 DataFrame 为可序列化的格式
                summary = {}
                for key, val in results.items():
                    if hasattr(val, 'to_dict'):
                        summary[key] = val.to_dict()
                    else:
                        summary[key] = str(val)
                json.dump(summary, f, indent=2)
                print(f"\n✅ 结果总结已保存: {args.output}/advanced_experiments_summary.json")
        
        except Exception as e:
            print(f"❌ 高级实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✨ 实验运行完成！")
    print(f"📊 结果保存在: {args.output}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
