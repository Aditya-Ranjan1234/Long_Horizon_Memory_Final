import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set style for professional look
plt.style.use('dark_background')
accent_color = '#facc15'  # Yellow accent

def generate_plots(log_file, output_dir):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df['step'] = df['reward_call_idx']
    
    # 1. Training Reward Trend
    plt.figure(figsize=(10, 6))
    reward_means = df.groupby('step')['total_reward'].mean()
    plt.plot(reward_means.index, reward_means.values, color=accent_color, linewidth=3, label='Mean Reward')
    plt.fill_between(reward_means.index, 0, reward_means.values, color=accent_color, alpha=0.2)
    plt.title('GRPO Training Reward Trend', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150)
    plt.close()

    # 2. Action Distribution
    plt.figure(figsize=(10, 6))
    ops = df['operation'].value_counts(normalize=True)
    ops.plot(kind='bar', color=[accent_color, '#444444', '#888888'], alpha=0.8)
    plt.title('Action Selection Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Operation', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=150)
    plt.close()

    # 3. Precision / Recall / F1 (Heuristic from rewards)
    # We'll treat task_reward > 0.4 as "Correct Add/Remove" (True Positive)
    # task_reward < 0 as "Error" (False Positive or False Negative)
    # This is a simplification for visualization
    
    tp = (df['task_reward'] > 0.4).sum()
    fp = (df['task_reward'] < 0).sum()
    fn = ((df['task_reward'] >= 0) & (df['task_reward'] <= 0.4) & (df['operation'] == 'noop')).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=accent_color, alpha=0.8)
    plt.ylim(0, 1.1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_results.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    log_path = "d:/6th Sem/scaler/Long_Horizon_Memory_Final/grpo_reward_log.jsonl"
    out_dir = "d:/6th Sem/scaler/Long_Horizon_Memory_Final/images"
    generate_plots(log_path, out_dir)
    print("Professional plots generated in images/")
