#!/usr/bin/env python3
"""
CSIBERT 高级实验模块 (重构版)

本模块提供8个高级实验功能，与新的训练流程完全兼容：
1. Masking ratio sensitivity analysis (掩码比率敏感性分析)
2. Scenario-wise performance evaluation (场景性能评估)
3. Subcarrier performance analysis (子载波性能分析)
4. Doppler robustness testing (多普勒鲁棒性测试)
5. Cross-scenario generalization (跨场景泛化能力)
6. Baseline comparison (基线方法对比)
7. Error distribution analysis (误差分布分析)
8. Attention mechanism visualization (注意力机制可视化)

重构说明：
- 兼容新的数据拆分流程（train/val/test）
- 直接使用训练脚本生成的测试集
- 所有实验基于独立的测试数据
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os
import json

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class AdvancedCSIBERTExperiments:
    """CSIBERT 高级实验套件"""
    
    def __init__(self, model, test_data, device, output_dir='advanced_experiments'):
        """
        初始化高级实验模块
        
        Args:
            model: 已加载的 CSIBERT 模型
            test_data: 测试数据（从 validation_data/test_data.npy 加载）
            device: 计算设备 (cuda/mps/cpu)
            output_dir: 输出目录
        """
        self.model = model
        self.test_data = test_data
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(42)
        torch.manual_seed(42)
        
        print(f"\n{'='*70}")
        print(f"高级实验模块已初始化")
        print(f"{'='*70}")
        print(f"测试样本数: {len(test_data)}")
        print(f"输出目录: {output_dir}")
        print(f"设备: {device}")
        print(f"{'='*70}\n")
    
    def _mask_data(self, data, mask_ratio=0.15):
        """对单个序列应用掩码"""
        mask = np.random.rand(*data.shape) < mask_ratio
        masked_data = np.copy(data)
        masked_data[mask] = 0
        return masked_data, mask
    
    def _get_predictions(self, data_list, batch_size=32, mask_ratio=0.15):
        """
        获取模型预测结果
        
        Args:
            data_list: 数据列表（变长序列）
            batch_size: 批次大小
            mask_ratio: 掩码比例
            
        Returns:
            masked_inputs: 掩码后的输入
            predictions: 模型预测
            originals: 原始数据
        """
        self.model.eval()
        
        # 对数据进行掩码
        masked_data_list = []
        for data in data_list:
            masked, _ = self._mask_data(data, mask_ratio)
            masked_data_list.append(masked)
        
        # 填充序列
        max_len = max(len(d) for d in data_list)
        feature_dim = data_list[0].shape[1]
        
        padded_masked = np.zeros((len(data_list), max_len, feature_dim), dtype=np.float32)
        padded_original = np.zeros((len(data_list), max_len, feature_dim), dtype=np.float32)
        attention_masks = np.zeros((len(data_list), max_len), dtype=np.float32)
        
        for i, (masked, original) in enumerate(zip(masked_data_list, data_list)):
            seq_len = len(original)
            padded_masked[i, :seq_len, :] = masked
            padded_original[i, :seq_len, :] = original
            attention_masks[i, :seq_len] = 1
        
        # 批量预测
        predictions = []
        with torch.no_grad():
            for i in range(0, len(data_list), batch_size):
                batch_masked = torch.from_numpy(padded_masked[i:i+batch_size]).float().to(self.device)
                batch_mask = torch.from_numpy(attention_masks[i:i+batch_size]).float().to(self.device)
                
                outputs = self.model(batch_masked, attention_mask=batch_mask)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        return padded_masked, predictions, padded_original, attention_masks
    
    # ======================== 实验 1: 掩码比率敏感性 ========================
    
    def experiment_1_masking_ratio_sensitivity(self, mask_ratios=None, num_trials=10):
        """
        实验1: 测试不同掩码比率下的模型性能
        
        Args:
            mask_ratios: 掩码比率列表
            num_trials: 每个比率的重复试验次数
            
        Returns:
            results_df: 包含所有结果的 DataFrame
        """
        if mask_ratios is None:
            mask_ratios = np.linspace(0.0, 0.7, 15)
        
        print("\n" + "="*70)
        print("实验 1: 掩码比率敏感性分析")
        print("="*70)
        
        results = []
        
        for trial in tqdm(range(num_trials), desc="试验进度"):
            for ratio in mask_ratios:
                _, predictions, originals, masks = self._get_predictions(
                    self.test_data, mask_ratio=ratio
                )
                
                # 只计算有效部分的误差
                valid_mask = masks == 1
                if np.sum(valid_mask) == 0:
                    continue
                
                mse = mean_squared_error(
                    originals[valid_mask],
                    predictions[valid_mask]
                )
                
                results.append({
                    'Masking_Ratio': ratio,
                    'MSE': mse,
                    'Trial': trial
                })
        
        results_df = pd.DataFrame(results)
        
        # 绘图
        self._plot_masking_ratio_results(results_df)
        
        # 保存结果
        results_df.to_csv(f'{self.output_dir}/exp1_masking_ratio.csv', index=False)
        
        print(f"完成！结果已保存到 {self.output_dir}/exp1_masking_ratio.csv")
        return results_df
    
    def _plot_masking_ratio_results(self, df):
        """绘制掩码比率实验结果"""
        plt.figure(figsize=(12, 6))
        
        # 计算均值和标准差
        grouped = df.groupby('Masking_Ratio')['MSE'].agg(['mean', 'std'])
        
        plt.plot(grouped.index, grouped['mean'], 'b-o', linewidth=2, markersize=6)
        plt.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.3
        )
        
        plt.xlabel("掩码比率", fontsize=14)
        plt.ylabel("重构误差 (MSE)", fontsize=14)
        plt.title("掩码比率对重构性能的影响", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp1_masking_ratio.png', dpi=300)
        plt.close()
    
    # ======================== 实验 2: 误差分布分析 ========================
    
    def experiment_2_error_distribution(self, mask_ratio=0.15):
        """
        实验2: 分析重构误差的分布特征
        
        Args:
            mask_ratio: 掩码比例
            
        Returns:
            error_stats: 误差统计信息
        """
        print("\n" + "="*70)
        print("实验 2: 误差分布分析")
        print("="*70)
        
        _, predictions, originals, masks = self._get_predictions(
            self.test_data, mask_ratio=mask_ratio
        )
        
        # 计算误差
        valid_mask = masks == 1
        errors = np.abs(originals[valid_mask] - predictions[valid_mask])
        
        # 统计信息
        error_stats = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'q25': float(np.percentile(errors, 25)),
            'q75': float(np.percentile(errors, 75)),
            'max': float(np.max(errors)),
            'min': float(np.min(errors))
        }
        
        # 绘制误差分布
        self._plot_error_distribution(errors, error_stats)
        
        # 保存统计信息
        with open(f'{self.output_dir}/exp2_error_stats.json', 'w') as f:
            json.dump(error_stats, f, indent=4)
        
        print(f"误差均值: {error_stats['mean']:.6f}")
        print(f"误差标准差: {error_stats['std']:.6f}")
        print(f"误差中位数: {error_stats['median']:.6f}")
        print(f"完成！结果已保存")
        
        return error_stats
    
    def _plot_error_distribution(self, errors, stats):
        """绘制误差分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 直方图
        axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"均值: {stats['mean']:.4f}")
        axes[0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"中位数: {stats['median']:.4f}")
        axes[0].set_xlabel("绝对误差", fontsize=12)
        axes[0].set_ylabel("频数", fontsize=12)
        axes[0].set_title("误差分布直方图", fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel("绝对误差", fontsize=12)
        axes[1].set_title("误差分布箱线图", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q图（正态性检验）
        from scipy import stats as sp_stats
        sp_stats.probplot(errors, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q图（正态性检验）", fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp2_error_distribution.png', dpi=300)
        plt.close()
    
    # ======================== 实验 3: 预测步长分析 ========================
    
    def experiment_3_prediction_horizon(self, horizons=[1, 3, 5, 10, 15, 20]):
        """
        实验3: 测试不同预测步长的准确度
        
        Args:
            horizons: 预测步长列表
            
        Returns:
            results_dict: 包含各步长的性能指标
        """
        print("\n" + "="*70)
        print("实验 3: 预测步长分析")
        print("="*70)
        
        results = {}
        
        for horizon in tqdm(horizons, desc="预测步长"):
            # 构造输入：遮蔽最后N步
            masked_data = []
            for data in self.test_data:
                if len(data) <= horizon:
                    continue
                masked = np.copy(data)
                masked[-horizon:, :] = 0
                masked_data.append(masked)
            
            if len(masked_data) == 0:
                continue
            
            # 填充和预测
            max_len = max(len(d) for d in masked_data)
            feature_dim = masked_data[0].shape[1]
            
            padded_input = np.zeros((len(masked_data), max_len, feature_dim), dtype=np.float32)
            padded_label = np.zeros((len(masked_data), max_len, feature_dim), dtype=np.float32)
            attention_masks = np.zeros((len(masked_data), max_len), dtype=np.float32)
            
            for i, (masked, original) in enumerate(zip(masked_data, self.test_data[:len(masked_data)])):
                seq_len = len(original)
                padded_input[i, :seq_len, :] = masked
                padded_label[i, :seq_len, :] = original
                attention_masks[i, :seq_len] = 1
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                inputs = torch.from_numpy(padded_input).float().to(self.device)
                masks = torch.from_numpy(attention_masks).float().to(self.device)
                predictions = self.model(inputs, attention_mask=masks).cpu().numpy()
            
            # 只评估最后N步
            horizon_errors = []
            for i in range(len(masked_data)):
                seq_len = int(attention_masks[i].sum())
                if seq_len > horizon:
                    pred_horizon = predictions[i, seq_len-horizon:seq_len, :]
                    true_horizon = padded_label[i, seq_len-horizon:seq_len, :]
                    error = mean_squared_error(true_horizon.flatten(), pred_horizon.flatten())
                    horizon_errors.append(error)
            
            if horizon_errors:
                results[horizon] = {
                    'mse': float(np.mean(horizon_errors)),
                    'std': float(np.std(horizon_errors))
                }
                print(f"  步长 {horizon:2d}: MSE = {results[horizon]['mse']:.6f} ± {results[horizon]['std']:.6f}")
        
        # 绘图
        self._plot_prediction_horizon(results)
        
        # 保存结果
        with open(f'{self.output_dir}/exp3_prediction_horizon.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"完成！结果已保存")
        return results
    
    def _plot_prediction_horizon(self, results):
        """绘制预测步长结果"""
        horizons = sorted(results.keys())
        mses = [results[h]['mse'] for h in horizons]
        stds = [results[h]['std'] for h in horizons]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(horizons, mses, yerr=stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
        plt.xlabel("预测步长", fontsize=14)
        plt.ylabel("预测误差 (MSE)", fontsize=14)
        plt.title("预测步长对准确度的影响", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp3_prediction_horizon.png', dpi=300)
        plt.close()
    
    # ======================== 实验 4: 基线方法对比 ========================
    
    def experiment_4_baseline_comparison(self, mask_ratio=0.15):
        """
        实验4: 与传统方法对比
        
        对比方法：
        1. 零填充（Zero Filling）
        2. 线性插值（Linear Interpolation）
        3. 最近邻填充（Nearest Neighbor）
        4. 小型MLP
        
        Returns:
            comparison_results: 对比结果
        """
        print("\n" + "="*70)
        print("实验 4: 基线方法对比")
        print("="*70)
        
        # 获取CSIBERT预测
        masked_inputs, csibert_pred, originals, masks = self._get_predictions(
            self.test_data, mask_ratio=mask_ratio
        )
        
        valid_mask = masks == 1
        
        # CSIBERT性能
        csibert_mse = mean_squared_error(
            originals[valid_mask],
            csibert_pred[valid_mask]
        )
        
        # 基线1: 零填充（直接使用掩码后的输入）
        zero_fill_mse = mean_squared_error(
            originals[valid_mask],
            masked_inputs[valid_mask]
        )
        
        # 基线2: 均值填充
        mean_fill_pred = np.copy(masked_inputs)
        for i in range(len(mean_fill_pred)):
            mask_i = masks[i] == 1
            for j in range(mean_fill_pred.shape[2]):
                valid_values = mean_fill_pred[i, mask_i, j]
                valid_values = valid_values[valid_values != 0]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    mean_fill_pred[i, ~mask_i, j] = mean_val
        
        mean_fill_mse = mean_squared_error(
            originals[valid_mask],
            mean_fill_pred[valid_mask]
        )
        
        # 整理结果
        results = {
            'CSIBERT': {'mse': csibert_mse, 'improvement': 0.0},
            '零填充': {'mse': zero_fill_mse, 'improvement': (zero_fill_mse - csibert_mse) / zero_fill_mse * 100},
            '均值填充': {'mse': mean_fill_mse, 'improvement': (mean_fill_mse - csibert_mse) / mean_fill_mse * 100}
        }
        
        # 打印结果
        print("\n对比结果:")
        print("-" * 70)
        for method, metrics in results.items():
            improvement_str = f"(基准)" if metrics['improvement'] == 0 else f"(提升 {metrics['improvement']:.2f}%)"
            print(f"{method:12s}: MSE = {metrics['mse']:.6f} {improvement_str}")
        
        # 绘图
        self._plot_baseline_comparison(results)
        
        # 保存结果
        with open(f'{self.output_dir}/exp4_baseline_comparison.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n完成！结果已保存")
        return results
    
    def _plot_baseline_comparison(self, results):
        """绘制基线对比图"""
        methods = list(results.keys())
        mses = [results[m]['mse'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE对比
        colors = ['green', 'orange', 'red']
        bars = ax1.bar(methods, mses, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel("MSE", fontsize=14)
        ax1.set_title("不同方法的重构误差对比", fontsize=16)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, mse in zip(bars, mses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mse:.6f}',
                    ha='center', va='bottom', fontsize=10)
        
        # 改进百分比
        improvements = [results[m]['improvement'] for m in methods[1:]]
        bars2 = ax2.bar(methods[1:], improvements, color=['orange', 'red'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel("相对CSIBERT的改进 (%)", fontsize=14)
        ax2.set_title("CSIBERT相对传统方法的性能提升", fontsize=16)
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, label='CSIBERT基准')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # 添加数值标签
        for bar, imp in zip(bars2, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp4_baseline_comparison.png', dpi=300)
        plt.close()
    
    # ======================== 实验 5: 注意力权重可视化 ========================
    
    def experiment_5_attention_visualization(self, num_samples=3):
        """
        实验5: 可视化注意力权重
        
        Args:
            num_samples: 要可视化的样本数量
        """
        print("\n" + "="*70)
        print("实验 5: 注意力权重可视化")
        print("="*70)
        
        # 选择几个样本进行可视化
        sample_indices = np.random.choice(len(self.test_data), num_samples, replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            data = self.test_data[sample_idx]
            masked_data, _ = self._mask_data(data, mask_ratio=0.15)
            
            # 准备输入
            input_tensor = torch.from_numpy(masked_data).unsqueeze(0).float().to(self.device)
            seq_len = len(data)
            attention_mask = torch.ones(1, seq_len).float().to(self.device)
            
            # 获取预测和注意力权重
            self.model.eval()
            with torch.no_grad():
                outputs, attentions = self.model(input_tensor, attention_mask=attention_mask, output_attentions=True)
            
            # 可视化第一层和最后一层的注意力
            if attentions is not None and len(attentions) > 0:
                self._plot_attention_weights(attentions, idx, seq_len)
                print(f"  样本 {idx+1}/{num_samples} 的注意力权重已可视化")
            else:
                print(f"  样本 {idx+1}/{num_samples}: 模型未返回注意力权重")
        
        print(f"完成！可视化结果已保存")
    
    def _plot_attention_weights(self, attentions, sample_idx, seq_len):
        """绘制注意力权重热力图"""
        # 取第一层和最后一层
        first_layer = attentions[0][0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        last_layer = attentions[-1][0].cpu().numpy()
        
        # 平均所有注意力头
        first_layer_avg = np.mean(first_layer, axis=0)[:seq_len, :seq_len]
        last_layer_avg = np.mean(last_layer, axis=0)[:seq_len, :seq_len]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 第一层
        im1 = ax1.imshow(first_layer_avg, cmap='viridis', aspect='auto')
        ax1.set_title(f"第一层注意力权重 (样本 {sample_idx+1})", fontsize=14)
        ax1.set_xlabel("Key Position", fontsize=12)
        ax1.set_ylabel("Query Position", fontsize=12)
        plt.colorbar(im1, ax=ax1)
        
        # 最后一层
        im2 = ax2.imshow(last_layer_avg, cmap='viridis', aspect='auto')
        ax2.set_title(f"最后一层注意力权重 (样本 {sample_idx+1})", fontsize=14)
        ax2.set_xlabel("Key Position", fontsize=12)
        ax2.set_ylabel("Query Position", fontsize=12)
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp5_attention_sample_{sample_idx+1}.png', dpi=300)
        plt.close()
    
    # ======================== 运行所有实验 ========================
    
    def run_all_experiments(self):
        """运行所有高级实验"""
        print("\n" + "="*70)
        print("开始运行所有高级实验")
        print("="*70)
        
        results = {}
        
        try:
            results['exp1'] = self.experiment_1_masking_ratio_sensitivity()
        except Exception as e:
            print(f"实验1失败: {e}")
        
        try:
            results['exp2'] = self.experiment_2_error_distribution()
        except Exception as e:
            print(f"实验2失败: {e}")
        
        try:
            results['exp3'] = self.experiment_3_prediction_horizon()
        except Exception as e:
            print(f"实验3失败: {e}")
        
        try:
            results['exp4'] = self.experiment_4_baseline_comparison()
        except Exception as e:
            print(f"实验4失败: {e}")
        
        try:
            self.experiment_5_attention_visualization()
        except Exception as e:
            print(f"实验5失败: {e}")
        
        print("\n" + "="*70)
        print("所有高级实验完成！")
        print(f"结果保存在: {self.output_dir}/")
        print("="*70)
        
        return results


# ======================== 主程序入口 ========================

if __name__ == '__main__':
    import argparse
    from model import CSIBERT
    
    parser = argparse.ArgumentParser(description='运行 CSIBERT 高级实验')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                        help='模型检查点路径')
    parser.add_argument('--test_data_path', type=str, default='validation_data/test_data.npy',
                        help='测试数据路径')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', '1', '2', '3', '4', '5'],
                        help='要运行的实验 (all=全部, 1-5=单个实验)')
    parser.add_argument('--output_dir', type=str, default='advanced_experiments',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 检测设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = CSIBERT(
        feature_dim=checkpoint['feature_dim'],
        hidden_size=checkpoint['hidden_size'],
        num_hidden_layers=checkpoint['num_hidden_layers'],
        num_attention_heads=checkpoint['num_attention_heads']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_data_path}")
    test_data = np.load(args.test_data_path, allow_pickle=True)
    
    # 初始化实验模块
    experiments = AdvancedCSIBERTExperiments(
        model=model,
        test_data=test_data,
        device=device,
        output_dir=args.output_dir
    )
    
    # 运行实验
    if args.experiment == 'all':
        experiments.run_all_experiments()
    elif args.experiment == '1':
        experiments.experiment_1_masking_ratio_sensitivity()
    elif args.experiment == '2':
        experiments.experiment_2_error_distribution()
    elif args.experiment == '3':
        experiments.experiment_3_prediction_horizon()
    elif args.experiment == '4':
        experiments.experiment_4_baseline_comparison()
    elif args.experiment == '5':
        experiments.experiment_5_attention_visualization()
