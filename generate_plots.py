import json
import matplotlib.pyplot as plt
import os

def generate_plots():
    log_file = "d:/6th Sem/scaler/Long_Horizon_Memory_Final/grpo_reward_log.jsonl"
    save_path = "d:/6th Sem/scaler/Long_Horizon_Memory_Final/images/training_results.png"
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    rewards = []
    task_rewards = []
    
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                rewards.append(data.get("total_reward", 0))
                task_rewards.append(data.get("task_reward", 0))

    if not rewards:
        print("No data found in log file.")
        return

    plt.figure(figsize=(10, 6), facecolor='#000000')
    ax = plt.gca()
    ax.set_facecolor('#000000')
    
    plt.plot(rewards, color='#facc15', linewidth=2, label='Total Reward')
    plt.plot(task_rewards, color='#646669', linewidth=1, linestyle='--', label='Task Reward')
    
    plt.title('GRPO Training Progress', color='#d1d0c5', fontsize=16, pad=20)
    plt.xlabel('Step', color='#646669')
    plt.ylabel('Reward', color='#646669')
    
    plt.legend(facecolor='#2c2e31', edgecolor='#3a3c40', labelcolor='#d1d0c5')
    
    # Grid and spines
    plt.grid(True, color='#2c2e31', linestyle=':', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#3a3c40')
    
    ax.tick_params(colors='#646669')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    generate_plots()
