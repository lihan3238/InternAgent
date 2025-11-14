#!/usr/bin/env python3
"""
可视化AutoForecast任务的实验结果
读取experiment.py保存的plot_data.json和final_info.json，生成可视化图表
"""
import json
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def load_plot_data(plot_data_path):
    """加载plot_data.json"""
    with open(plot_data_path, 'r') as f:
        data = json.load(f)
    predictions = np.array(data['predictions'])
    targets = np.array(data['targets'])
    pred_length = data.get('pred_length', 10)
    seq_length = data.get('seq_length', 48)
    return predictions, targets, pred_length, seq_length

def load_metrics(metrics_path):
    """加载final_info.json中的指标"""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    metrics = data.get('AutoForecast', {}).get('means', {})
    return metrics

def plot_predictions_vs_targets(predictions, targets, output_path, pred_length):
    """绘制预测值vs真实值的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Energy Consumption Forecasting: Predictions vs Targets', fontsize=16, fontweight='bold')
    
    # 1. 时间序列对比（选择几个样本）
    ax1 = axes[0, 0]
    n_samples = min(10, len(predictions))
    indices = np.linspace(0, len(predictions) - 1, n_samples, dtype=int)
    
    for idx in indices[:5]:  # 只显示前5个样本
        time_steps = np.arange(pred_length)
        ax1.plot(time_steps, predictions[idx], 'o-', alpha=0.6, label=f'Pred {idx}' if idx < 3 else '')
        ax1.plot(time_steps, targets[idx], 's--', alpha=0.6, label=f'True {idx}' if idx < 3 else '')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Energy Consumption (normalized)')
    ax1.set_title('Sample Predictions vs Targets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图：预测值 vs 真实值
    ax2 = axes[0, 1]
    # 展平所有预测和真实值
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()
    # 采样以避免点太多
    if len(pred_flat) > 5000:
        sample_idx = np.random.choice(len(pred_flat), 5000, replace=False)
        pred_flat = pred_flat[sample_idx]
        tgt_flat = tgt_flat[sample_idx]
    
    ax2.scatter(tgt_flat, pred_flat, alpha=0.3, s=1)
    # 添加对角线
    min_val = min(pred_flat.min(), tgt_flat.min())
    max_val = max(pred_flat.max(), tgt_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Predictions vs Targets (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分布
    ax3 = axes[1, 0]
    errors = pred_flat - tgt_flat
    ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 每个时间步的平均误差
    ax4 = axes[1, 1]
    mean_errors = np.mean(predictions - targets, axis=0)
    std_errors = np.std(predictions - targets, axis=0)
    time_steps = np.arange(pred_length)
    ax4.plot(time_steps, mean_errors, 'o-', label='Mean Error')
    ax4.fill_between(time_steps, mean_errors - std_errors, mean_errors + std_errors, 
                     alpha=0.3, label='±1 Std')
    ax4.axhline(0, color='r', linestyle='--', linewidth=1)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Mean Error')
    ax4.set_title('Error by Time Step')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved predictions vs targets plot to {output_path}")

def plot_metrics_summary(metrics, output_path):
    """绘制指标摘要图"""
    if not metrics:
        print("[PLOT] No metrics available for summary plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # 提取主要指标
    metric_names = []
    metric_values = []
    
    # 常见的指标名称
    common_metrics = ['rmse', 'mae', 'mape', 'r2', 'mse']
    for m in common_metrics:
        if m in metrics:
            metric_names.append(m.upper())
            metric_values.append(metrics[m])
    
    if not metric_names:
        # 如果没有找到常见指标，显示所有指标
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
    
    # 1. 条形图
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Value')
    ax1.set_title('Performance Metrics')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在条形上添加数值标签
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 文本摘要
    ax2 = axes[1]
    ax2.axis('off')
    summary_text = "Model Performance Summary\n" + "=" * 30 + "\n\n"
    for name, val in zip(metric_names, metric_values):
        summary_text += f"{name}: {val:.6f}\n"
    ax2.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved metrics summary to {output_path}")

def plot_time_series_samples(predictions, targets, output_path, pred_length, n_samples=20):
    """绘制更多时间序列样本的详细对比"""
    n_samples = min(n_samples, len(predictions))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle('Detailed Time Series Predictions vs Targets', fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.linspace(0, len(predictions) - 1, n_samples, dtype=int)
    time_steps = np.arange(pred_length)
    
    for idx, ax in zip(indices, axes.flat):
        ax.plot(time_steps, predictions[idx], 'o-', label='Predicted', linewidth=2, markersize=4)
        ax.plot(time_steps, targets[idx], 's--', label='True', linewidth=2, markersize=4)
        ax.set_title(f'Sample {idx}', fontsize=10)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for ax in axes.flat[len(indices):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved detailed time series plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize AutoForecast experiment results")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (same as experiment.py)")
    args = parser.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")
    
    plot_data_path = out_dir / "plot_data.json"
    metrics_path = out_dir / "final_info.json"
    
    if not plot_data_path.exists():
        print(f"[WARNING] plot_data.json not found at {plot_data_path}")
        print("[WARNING] Skipping prediction plots. Make sure experiment.py saved plot_data.json")
        plot_data_available = False
    else:
        plot_data_available = True
        predictions, targets, pred_length, seq_length = load_plot_data(plot_data_path)
        print(f"[INFO] Loaded plot data: {len(predictions)} samples, pred_length={pred_length}")
    
    if not metrics_path.exists():
        print(f"[WARNING] final_info.json not found at {metrics_path}")
        metrics_available = False
    else:
        metrics_available = True
        metrics = load_metrics(metrics_path)
        print(f"[INFO] Loaded metrics: {list(metrics.keys())}")
    
    # 生成可视化图表
    if plot_data_available:
        # 1. 预测vs真实值对比图
        plot_predictions_vs_targets(
            predictions, targets, 
            out_dir / "predictions_vs_targets.png",
            pred_length
        )
        
        # 2. 详细时间序列样本
        plot_time_series_samples(
            predictions, targets,
            out_dir / "time_series_samples.png",
            pred_length,
            n_samples=20
        )
    
    if metrics_available:
        # 3. 指标摘要图
        plot_metrics_summary(
            metrics,
            out_dir / "metrics_summary.png"
        )
    
    print(f"[INFO] All plots saved to {out_dir}")

if __name__ == "__main__":
    main()

