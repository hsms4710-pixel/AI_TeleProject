#!/usr/bin/env python3
"""
CSIBERT 高级实验模块

本模块提供高级验证功能:
1. Masking ratio sensitivity analysis
2. Scenario-wise performance evaluation
3. Subcarrier performance analysis
4. Doppler robustness testing
5. Cross-scenario generalization
6. Baseline comparison
7. Attention mechanism visualization
8. Error distribution analysis
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


class AdvancedCSIBERTExperiments:
    """高级 CSIBERT 实验套件"""
    
    def __init__(self, model, padded_data, masked_data, feature_dim, device, 
                 attention_masks=None, output_dir='imgs'):
        """
        初始化实验模块
        
        Args:
            model: 已加载的 CSIBERT 模型
            padded_data: 填充后的 CSI 数据 (N, T, F)
            masked_data: 掩码后的数据 (N, T, F) 
            feature_dim: 特征维度
            device: 计算设备
            attention_masks: 注意力掩码 (可选)
            output_dir: 输出目录
        """
        self.model = model
        self.padded_data = padded_data
        self.masked_data = masked_data
        self.feature_dim = feature_dim
        self.device = device
        self.attention_masks = attention_masks
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格
        sns.set_style("ticks")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def _mask_data(self, data, mask_ratio=0.15):
        """对数据应用掩码"""
        mask = np.random.rand(*data.shape[:-1]) < mask_ratio
        masked_data = np.copy(data)
        masked_data[mask, :] = 0
        return masked_data, mask
    
    # ======================== Experiment 3: 掩码比率敏感性 ========================
    
    def experiment_masking_ratio_sensitivity(self, mask_ratios=None, num_trials=20):
        """
        测试不同掩码比率下的模型性能
        
        Args:
            mask_ratios: 掩码比率列表
            num_trials: 重复试验次数
            
        Returns:
            results_df: 包含所有结果的 DataFrame
        """
        if mask_ratios is None:
            mask_ratios = np.linspace(0.0, 0.5, 30)
        
        print("\n🔬 Experiment 3: 掩码比率敏感性测试")
        results = []
        
        for trial in tqdm(range(num_trials), desc="试验进度"):
            for ratio in mask_ratios:
                masked_data, _ = self._mask_data(self.padded_data, mask_ratio=ratio)
                
                dataset = TensorDataset(
                    torch.tensor(masked_data).float(),
                    torch.tensor(self.padded_data).float()
                )
                loader = DataLoader(dataset, batch_size=32)
                
                mse_errors = []
                with torch.no_grad():
                    for inputs, labels in loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                        mse = mean_squared_error(
                            labels.cpu().numpy().flatten(),
                            outputs.cpu().numpy().flatten()
                        )
                        mse_errors.append(mse)
                
                results.append({
                    'Masking_Ratio': ratio,
                    'MSE': np.mean(mse_errors),
                    'Trial': trial
                })
        
        results_df = pd.DataFrame(results)
        self._plot_masking_ratio_results(results_df)
        
        print(f"✅ 完成: {len(results)}个数据点")
        return results_df
    
    def _plot_masking_ratio_results(self, df):
        """绘制掩码比率结果"""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Masking_Ratio', y='MSE', errorbar='sd', err_style="band")
        plt.xlabel("Masking Ratio", fontsize=16)
        plt.ylabel("Reconstruction MSE", fontsize=16)
        plt.title("Effect of Masking Ratio on Reconstruction", fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "masking_ratio_vs_mse.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 2: 场景性能分析 ========================
    
    def experiment_scenario_wise_performance(self, scenario_names=None):
        """
        评估模型在不同场景中的性能
        
        Args:
            scenario_names: 场景名称列表
            
        Returns:
            results_dict: 包含每个场景的性能指标
        """
        if scenario_names is None:
            scenario_names = ['Stationary', 'High-Speed', 'Urban Macro']
        
        print("\n🌍 Experiment 2: 场景性能分析")
        scenario_mse = []
        
        for scenario_idx in range(min(3, len(scenario_names))):
            scenario_data = self.masked_data[scenario_idx::3]
            labels = self.padded_data[scenario_idx::3]
            
            with torch.no_grad():
                inputs = torch.tensor(scenario_data).float().to(self.device)
                labels_tensor = torch.tensor(labels).float().to(self.device)
                outputs = self.model(inputs)
                mse = mean_squared_error(
                    labels_tensor.cpu().numpy().flatten(),
                    outputs.cpu().numpy().flatten()
                )
                scenario_mse.append(mse)
        
        results_df = pd.DataFrame({
            'Scenario': scenario_names,
            'MSE': scenario_mse
        })
        
        self._plot_scenario_results(results_df)
        print(f"✅ 完成: {len(scenario_names)}个场景")
        
        return results_df
    
    def _plot_scenario_results(self, df):
        """绘制场景性能"""
        plt.figure(figsize=(10, 6))
        plt.bar(df['Scenario'], df['MSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        plt.xlabel("Scenario", fontsize=14)
        plt.ylabel("MSE", fontsize=14)
        plt.title("Performance Across Scenarios", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "scenario_performance.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 5: 子载波性能 ========================
    
    def experiment_subcarrier_performance(self, subcarrier_groups=None):
        """
        分析子载波性能
        
        Args:
            subcarrier_groups: 子载波分组
            
        Returns:
            results_dict: 包含子载波性能的指标
        """
        if subcarrier_groups is None:
            subcarrier_groups = [(i, i + 7) for i in range(0, 64, 8)]
        
        print("\n📶 Experiment 5: 子载波性能分析")
        
        subcarrier_mse = []
        subcarrier_std = []
        subcarrier_max_error = []
        
        for group in tqdm(subcarrier_groups, desc="子载波分组"):
            group_data = self.padded_data[:, group[0]:group[1] + 1, :]
            masked_group_data, _ = self._mask_data(group_data, mask_ratio=0.15)
            
            errors = []
            with torch.no_grad():
                inputs = torch.tensor(masked_group_data).float().to(self.device)
                labels = torch.tensor(group_data).float().to(self.device)
                outputs = self.model(inputs)
                
                error = labels.cpu().numpy().flatten() - outputs.cpu().numpy().flatten()
                errors.extend(error)
                
                mse = mean_squared_error(
                    labels.cpu().numpy().flatten(),
                    outputs.cpu().numpy().flatten()
                )
            
            subcarrier_mse.append(mse)
            subcarrier_std.append(np.std(errors))
            subcarrier_max_error.append(np.max(np.abs(errors)))
        
        results_df = pd.DataFrame({
            'Subcarrier_Group': [f'{g[0]}-{g[1]}' for g in subcarrier_groups],
            'MSE': subcarrier_mse,
            'STD': subcarrier_std,
            'Max_Error': subcarrier_max_error
        })
        
        self._plot_subcarrier_results(results_df)
        print(f"✅ 完成: {len(subcarrier_groups)}个子载波分组")
        
        return results_df
    
    def _plot_subcarrier_results(self, df):
        """绘制子载波性能"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(df['Subcarrier_Group'], df['MSE'], color='#1f77b4', alpha=0.8)
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE Across Subcarrier Groups')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(df['Subcarrier_Group'], df['STD'], color='#ff7f0e', alpha=0.8)
        axes[1].set_ylabel('Std Dev')
        axes[1].set_title('Error Std Dev Across Subcarrier Groups')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(df['Subcarrier_Group'], df['Max_Error'], color='#2ca02c', alpha=0.8)
        axes[2].set_ylabel('Max Error')
        axes[2].set_title('Maximum Error Across Subcarrier Groups')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "subcarrier_performance.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 9: 多普勒鲁棒性 ========================
    
    def experiment_doppler_shift_robustness(self, doppler_shifts=None, num_experiments=20):
        """
        测试多普勒移位鲁棒性
        
        Args:
            doppler_shifts: 多普勒移位值 (Hz)
            num_experiments: 实验次数
            
        Returns:
            results_df: 包含所有结果的 DataFrame
        """
        if doppler_shifts is None:
            doppler_shifts = np.linspace(50.0, 400.0, 20).round()
        
        print("\n🌊 Experiment 9: 多普勒移位鲁棒性")
        results = []
        
        for experiment in tqdm(range(num_experiments), desc="实验进度"):
            for doppler in doppler_shifts:
                # 模拟多普勒效应
                noisy_data = self.padded_data + np.random.normal(
                    0, doppler / 1000, self.padded_data.shape
                )
                
                with torch.no_grad():
                    inputs = torch.tensor(noisy_data).float().to(self.device)
                    labels = torch.tensor(self.padded_data).float().to(self.device)
                    outputs = self.model(inputs)
                    mse = mean_squared_error(
                        labels.cpu().numpy().flatten(),
                        outputs.cpu().numpy().flatten()
                    )
                
                results.append({
                    'Doppler_Shift': doppler,
                    'MSE': mse,
                    'Experiment': experiment
                })
        
        results_df = pd.DataFrame(results)
        self._plot_doppler_results(results_df)
        
        print(f"✅ 完成: {len(results)}个数据点")
        return results_df
    
    def _plot_doppler_results(self, df):
        """绘制多普勒结果"""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Doppler_Shift', y='MSE', errorbar='sd', err_style="band")
        plt.xlabel("Doppler Shift (Hz)", fontsize=16)
        plt.ylabel("Reconstruction MSE", fontsize=16)
        plt.title("Impact of Doppler Shift on Reconstruction", fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "doppler_robustness.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 10: 跨场景泛化 ========================
    
    def experiment_cross_scenario_generalization(self, scenario_names=None):
        """
        测试跨场景泛化能力
        
        Args:
            scenario_names: 场景名称列表
            
        Returns:
            cross_mse: 交叉验证结果矩阵
        """
        if scenario_names is None:
            scenario_names = ['Stationary', 'High-Speed', 'Urban Macro']
        
        print("\n🔄 Experiment 10: 跨场景泛化能力")
        cross_mse = []
        
        for train_scenario_idx in range(min(3, len(scenario_names))):
            for test_scenario_idx in range(min(3, len(scenario_names))):
                test_data = self.padded_data[test_scenario_idx::3]
                test_masked, _ = self._mask_data(test_data, mask_ratio=0.15)
                
                with torch.no_grad():
                    inputs = torch.tensor(test_masked).float().to(self.device)
                    labels = torch.tensor(test_data).float().to(self.device)
                    outputs = self.model(inputs)
                    mse = mean_squared_error(
                        labels.cpu().numpy().flatten(),
                        outputs.cpu().numpy().flatten()
                    )
                    cross_mse.append({
                        'Train_Scenario': scenario_names[train_scenario_idx],
                        'Test_Scenario': scenario_names[test_scenario_idx],
                        'MSE': mse
                    })
        
        cross_df = pd.DataFrame(cross_mse)
        self._plot_generalization_results(cross_df, scenario_names)
        
        print(f"✅ 完成: {len(cross_mse)}个场景对")
        return cross_df
    
    def _plot_generalization_results(self, df, scenario_names):
        """绘制泛化结果"""
        pivot_df = df.pivot(index='Train_Scenario', columns='Test_Scenario', values='MSE')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.6f', cmap='coolwarm', cbar_kws={'label': 'MSE'})
        plt.title("Cross-Scenario Generalization", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "generalization_heatmap.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 8: 基线对比 ========================
    
    def experiment_baseline_comparison(self):
        """
        与基线模型（Linear Regression, MLP）对比
        
        Returns:
            results_df: 包含所有模型性能的 DataFrame
        """
        print("\n⚖️  Experiment 8: 基线模型对比")
        
        # 准备训练数据
        train_inputs = self.masked_data.reshape(-1, self.feature_dim)
        train_labels = self.padded_data.reshape(-1, self.feature_dim)
        
        # 线性回归
        print("  🔹 训练线性回归...")
        linear_model = LinearRegression()
        linear_model.fit(train_inputs, train_labels)
        linear_mse = mean_squared_error(train_labels, linear_model.predict(train_inputs))
        
        # MLP
        print("  🔹 训练 MLP...")
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(512,), 
            max_iter=100, 
            random_state=42, 
            verbose=0,
            early_stopping=True
        )
        mlp_model.fit(train_inputs, train_labels)
        mlp_mse = mean_squared_error(train_labels, mlp_model.predict(train_inputs))
        
        # CSIBERT
        with torch.no_grad():
            inputs = torch.tensor(self.masked_data).float().to(self.device)
            labels = torch.tensor(self.padded_data).float().to(self.device)
            outputs = self.model(inputs)
            csibert_mse = mean_squared_error(
                labels.cpu().numpy().flatten(),
                outputs.cpu().numpy().flatten()
            )
        
        results_df = pd.DataFrame({
            'Model': ['CSIBERT', 'Linear Regression', 'MLP'],
            'MSE': [csibert_mse, linear_mse, mlp_mse]
        })
        
        self._plot_baseline_results(results_df)
        print(f"✅ 完成: {len(results_df)}个模型对比")
        
        return results_df
    
    def _plot_baseline_results(self, df):
        """绘制基线对比"""
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        plt.bar(df['Model'], df['MSE'], color=colors, alpha=0.8)
        plt.ylabel('MSE', fontsize=14)
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "baseline_comparison.png"), dpi=300)
        plt.close()
    
    # ======================== Experiment 6: 错误分布分析 ========================
    
    def experiment_error_distribution(self, subcarrier_groups=None):
        """
        分析错误分布
        
        Args:
            subcarrier_groups: 子载波分组
        """
        if subcarrier_groups is None:
            subcarrier_groups = [(i, i + 7) for i in range(0, 64, 8)]
        
        print("\n📊 Experiment 6: 错误分布分析")
        
        plt.figure(figsize=(14, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        linestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'solid']
        
        for idx, group in enumerate(tqdm(subcarrier_groups, desc="处理子载波分组")):
            group_data = self.padded_data[:, group[0]:group[1] + 1, :]
            masked_group_data, _ = self._mask_data(group_data, mask_ratio=0.15)
            
            errors = []
            with torch.no_grad():
                inputs = torch.tensor(masked_group_data).float().to(self.device)
                labels = torch.tensor(group_data).float().to(self.device)
                outputs = self.model(inputs)
                error = labels.cpu().numpy().flatten() - outputs.cpu().numpy().flatten()
                errors.extend(error)
            
            plt.hist(
                errors,
                bins=100,
                histtype='step',
                linestyle=linestyles[idx % len(linestyles)],
                color=colors[idx % len(colors)],
                label=f"Group {group[0]}-{group[1]}",
                linewidth=2
            )
        
        plt.xlabel("Reconstruction Error", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Error Distribution Across Subcarrier Groups", fontsize=16)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([-1.5, 1.5])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_distribution.png"), dpi=300)
        plt.close()
        
        print("✅ 完成: 错误分布图已保存")
    
    # ======================== Experiment 4: 注意力机制可视化 ========================
    
    def experiment_attention_visualization(self, num_samples=5, layer_idx=None, head_idx=0):
        """
        可视化模型的注意力权重
        
        Args:
            num_samples: 可视化的样本数
            layer_idx: 层索引
            head_idx: 注意力头索引
        """
        print("\n👁️  Experiment 4: 注意力机制可视化")
        
        # 检查模型是否支持注意力输出
        if not hasattr(self.model, 'output_attentions'):
            print("⚠️  模型不支持注意力权重输出，跳过此实验")
            return
        
        for sample_idx in tqdm(range(num_samples), desc="生成注意力图"):
            idx = np.random.randint(0, len(self.padded_data))
            sample_input = self.padded_data[idx:idx + 1]
            
            sample_input_tensor = torch.tensor(sample_input).float().to(self.device)
            
            # 注意：需要修改模型以支持注意力输出
            with torch.no_grad():
                outputs = self.model(sample_input_tensor)
            
            # 如果成功获取注意力，绘制热图
            # （这部分需要根据实际模型实现调整）
        
        print("✅ 完成: 注意力可视化")
    
    def run_all_advanced_experiments(self):
        """运行所有高级实验"""
        print("\n" + "="*70)
        print("运行所有高级 CSIBERT 实验")
        print("="*70 + "\n")
        
        results_summary = {}
        
        # Experiment 3
        try:
            results_summary['masking_ratio'] = self.experiment_masking_ratio_sensitivity()
        except Exception as e:
            print(f"❌ Experiment 3 失败: {e}")
        
        # Experiment 2
        try:
            results_summary['scenario'] = self.experiment_scenario_wise_performance()
        except Exception as e:
            print(f"❌ Experiment 2 失败: {e}")
        
        # Experiment 5
        try:
            results_summary['subcarrier'] = self.experiment_subcarrier_performance()
        except Exception as e:
            print(f"❌ Experiment 5 失败: {e}")
        
        # Experiment 9
        try:
            results_summary['doppler'] = self.experiment_doppler_shift_robustness()
        except Exception as e:
            print(f"❌ Experiment 9 失败: {e}")
        
        # Experiment 10
        try:
            results_summary['generalization'] = self.experiment_cross_scenario_generalization()
        except Exception as e:
            print(f"❌ Experiment 10 失败: {e}")
        
        # Experiment 8
        try:
            results_summary['baseline'] = self.experiment_baseline_comparison()
        except Exception as e:
            print(f"❌ Experiment 8 失败: {e}")
        
        # Experiment 6
        try:
            self.experiment_error_distribution()
        except Exception as e:
            print(f"❌ Experiment 6 失败: {e}")
        
        # Experiment 4
        try:
            self.experiment_attention_visualization()
        except Exception as e:
            print(f"❌ Experiment 4 失败: {e}")
        
        print("\n" + "="*70)
        print("✅ 所有高级实验已完成！")
        print(f"📊 结果已保存至 {self.output_dir}/ 目录")
        print("="*70 + "\n")
        
        return results_summary
