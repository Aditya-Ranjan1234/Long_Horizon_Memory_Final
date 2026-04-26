# 🧠 Long Horizon Memory Final

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-Hosted%20Env-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/spaces/aditya-ranjan1234/long-horizon-memory-env-final)
[![Training Notebook](https://img.shields.io/badge/Notebook-Training%20Pipeline-blue?logo=jupyter&style=flat-square)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/complete_SFT_GRPO.ipynb)

A professional-grade Reinforcement Learning environment and training pipeline for evaluating long-horizon memory management in LLM agents. This project leverages **Supervised Fine-Tuning (SFT)** and **Group Relative Policy Optimization (GRPO)** to teach agents how to efficiently manage a constrained memory buffer over thousands of tokens.

---

## 🏗️ Architecture & Environment

The environment simulates a high-pressure memory management task where an agent must decide which information to keep, add, or discard from a fixed-capacity buffer.

### 🎮 The Environment Mechanics
- **Capacity**: Configurable (default: 8 or 16 slots).
- **Operations**:
  - `add`: Add the current message to memory.
  - `remove`: Remove an entry at a specific index to free space.
  - `noop`: Do nothing (ignore irrelevant information).
- **Challenge**: The agent must identify "Relevant" messages across different domains (Cybersecurity, ML Ops, etc.) and maintain them in memory while evicting "Irrelevant" noise.

---

## 🚀 Training Pipeline

We follow a two-stage training process to transform a base LLM (**Qwen2.5-1.5B-Instruct**) into a specialized memory manager.

### 1️⃣ Phase 1: Supervised Fine-Tuning (SFT)
The model is trained on a "Seed Dataset" of perfect memory operations. This teaches the model the basic JSON schema and the semantic difference between relevant and irrelevant messages.
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Method**: LoRA (4-bit quantization)
- **Data**: ~175 samples of perfect memory management.

![SFT Loss Curve](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/sft_loss_curve.png)
*SFT Training Log Reference:*
![SFT Training Log](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/sft_training_log.jpg)

### 2️⃣ Phase 2: Group Relative Policy Optimization (GRPO)
Using the SFT model as a starting point, we apply GRPO to optimize for long-horizon performance. Unlike traditional PPO, GRPO samples a group of completions for the same prompt and calculates the advantage relative to the group mean, significantly reducing memory overhead.

#### 🧠 Training Methodology (Log Analysis)
Analysis of the [📜 GRPO Training Log](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/GRPO%20Training%20Log.txt) reveals key architectural strategies:
- **Active Exploration via Logit-Biasing**: The training engine applies a **+3.5 bias** to tokens associated with the `remove` operation. This forces the model to explore buffer cleaning strategies rather than defaulting to a safe "noop" state.
- **Style-Curated Curriculum**: Training data is mixed with specific scenarios:
  - `remove_friendly`: Scenarios where the buffer is nearly full and low-value items must be evicted.
  - `fill_first`: Scenarios focused on early-episode identification of relevant data.
- **Adapter-Based Training**: Uses a LoRA adapter (Rank 8-16) to specialize the **Qwen2.5-1.5B** backbone for JSON-structured environment interaction.

#### ⚖️ Reward Signal: Accuracy & Precision Deep Dive
The environment uses a **Shaped Reward Signal** designed to break "noop-collapse" (where models learn to do nothing to avoid penalties).

| Action | State | Reward | Logic |
| :--- | :--- | :--- | :--- |
| **Add** | Relevant Message | `+0.60` | High incentive for capturing facts. |
| **Add** | Irrelevant Noise | `-0.60` | Heavy penalty for memory pollution. |
| **Remove** | Irrelevant Item | `+0.40` | Reward for active buffer maintenance (Precision). |
| **Remove** | Relevant Item | `-0.50` | Penalty for losing historical context (Recall). |
| **Noop** | Relevant Message | `-0.30` | Penalty for missing information (Omission). |
| **Noop** | Irrelevant Noise | `+0.05` | Minor reward for filtering out noise. |

**Terminal Bonus (The Judge's Metric):**
At the end of each episode, a bonus of **0.5 × F1-Score** is applied. 
- **Precision-Based Reward**: `Correct_Kept / Total_Kept`. Penalizes agents that fill memory with junk.
- **Recall-Based Reward**: `Correct_Kept / Total_Relevant_Seen`. Penalizes agents that ignore key facts.
- **F1-Score**: The harmonic mean ensures the agent must balance both accuracy in selection and diligence in retention.

---

## 📊 Results & Benchmarks

The following summary compares the base model, SFT model, and the final GRPO-tuned model across 20 episodes and 1768 steps.

### Performance Comparison
![Benchmark Metrics](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/benchmark_metrics.png)

### Action Selection Evolution
![Action Distribution Comparison](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/action_distribution_comparison.png)

### Benchmark Summary Data
![Benchmark Summary](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/benchmark_summary.png)

---

## 📂 Logs & Traceability

Detailed execution logs for both training phases are available for audit:
- [📜 GRPO Training Full Log (Text)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/GRPO%20Training%20Log.txt)
- [📜 SFT Training Full Log (Text)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/SFT%20Training%20For%20JSON%20Format.txt)

---

## 📂 Project Structure

```bash
.
├── server/                 # FastAPI Environment & Logic
├── ui/                     # React Dashboard (Pure Black Theme)
├── verification/           # Scripts for testing & path validation
├── images/                 # Benchmark visualizations
├── complete_SFT_GRPO.ipynb # Full Training Pipeline
├── data.py                 # System prompts & Seed data
└── train_grpo_memory.py    # GRPO Training Script
```

---

## 🛠️ Verification
To verify the environment paths and WebSocket connectivity:
```powershell
python verification/verify_env_paths.py
```
