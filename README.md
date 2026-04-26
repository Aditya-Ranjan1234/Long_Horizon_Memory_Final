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

### 2️⃣ Phase 2: Group Relative Policy Optimization (GRPO)
Using the SFT model as a starting point, we apply GRPO to optimize for long-horizon performance. GRPO compares a group of completions against each other to calculate advantages without a separate value function.

#### ⚖️ Reward Function Logic
The policy is optimized using a multi-objective reward function:
- **Task Reward**: Based on the environment state (`task_score`).
  - Correct Add: `+0.6`
  - Correct Remove: `+0.4`
  - Correct Noop: `+0.05`
  - Errors: `-0.5` to `-1.0`
- **Format Reward**: `+0.05` for valid, parseable JSON output.
- **Diversity Bonus**: Encourages the model to explore different operations within a group.
- **Correct Remove Bonus**: An additional `+0.20` to encourage active buffer cleaning.

---

## 📊 Results & Benchmarks

The following results demonstrate the agent's ability to maintain high precision even as the memory buffer fills up.

### Precision & Recall
![Precision Recall](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/precision_recall.png)

### F1-Score Evolution
![F1 Scores](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/f1_scores.png)

### Action Frequencies
The agent learns a balanced distribution of `add` and `remove` operations, moving away from a "noop-only" collapse.
![Action Distribution](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/action_distribution.png)

### Training Reward Evolution
![Training Results](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/training_results.png)

### Benchmark Summary
![Benchmark Table](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/benchmark_table.png)

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
