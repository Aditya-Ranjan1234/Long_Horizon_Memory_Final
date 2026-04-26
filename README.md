---
title: Long Horizon Memory Final
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
---

# Long Horizon Memory Final

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-Hosted%20Env-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/spaces/aditya-ranjan1234/long-horizon-memory-env-final)
[![Training Notebook](https://img.shields.io/badge/Notebook-Training%20Pipeline-blue?logo=jupyter&style=flat-square)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/complete_SFT_GRPO.ipynb)
[![Full Training Script](https://img.shields.io/badge/Colab-Full%20Training%20Script-orange?logo=googlecolab&style=flat-square)](https://colab.research.google.com/drive/1IWAyAtn565JfX7vCLqyCaiW8a2dt-0cv?usp=sharing)

A professional-grade Reinforcement Learning environment and training pipeline for evaluating long-horizon memory management in LLM agents. This project leverages **Supervised Fine-Tuning (SFT)** and **Group Relative Policy Optimization (GRPO)** to teach agents how to efficiently manage a constrained memory buffer over thousands of tokens.

---

## Links

- **GitHub Repository**: [Aditya-Ranjan1234/Long_Horizon_Memory_Final](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final)
- **Hugging Face Space**: [Live Dashboard Environment](https://huggingface.co/spaces/aditya-ranjan1234/long-horizon-memory-env-final)
- **Full Training Script**: [Google Colab Notebook](https://colab.research.google.com/drive/1IWAyAtn565JfX7vCLqyCaiW8a2dt-0cv?usp=sharing)

<div align="center">

![Live Dashboard Environment](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/hosted_hugging_face_space.png)
<br>
*Live dashboard on Hugging Face*

</div>

---

## Architecture & Environment

The environment simulates a high-pressure memory management task where an agent must decide which information to keep, add, or discard from a fixed-capacity buffer.

### The Environment Mechanics
- **Capacity**: Configurable (default: 8 or 16 slots).
- **Operations**:
  - `add`: Add the current message to memory.
  - `remove`: Remove an entry at a specific index to free space.
  - `noop`: Do nothing (ignore irrelevant information).
- **Challenge**: The agent must identify "Relevant" messages across different domains (Cybersecurity, ML Ops, etc.) and maintain them in memory while evicting "Irrelevant" noise.

---

## Training Pipeline

We follow a two-stage training process to transform a base LLM (**Qwen2.5-1.5B-Instruct**) into a specialized memory manager.

### Phase 1: Supervised Fine-Tuning (SFT)
The model is trained on a "Seed Dataset" of perfect memory operations. This teaches the model the basic JSON schema and the semantic difference between relevant and irrelevant messages.
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Method**: LoRA (4-bit quantization)
- **Data**: ~175 samples of perfect memory management.

<div align="center">

![SFT Loss Curve](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/sft_loss_curve.png)
<br>
*SFT Training Log Reference:*
<br>
![SFT Training Log](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/sft_training_log.jpg)

</div>

### Phase 2: Group Relative Policy Optimization (GRPO)
Using the SFT model as a starting point, we apply GRPO to optimize for long-horizon performance. Unlike traditional PPO, GRPO samples a group of completions for the same prompt and calculates the advantage relative to the group mean, significantly reducing memory overhead.

<div align="center">

![GRPO Training Step Example](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/grpo_step_example.png)

</div>

#### Training Methodology (Log Analysis)
Analysis of the [GRPO Training Log](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/GRPO%20Training%20Log.txt) reveals key architectural strategies:
- **Active Exploration via Logit-Biasing**: The training engine applies a **+3.5 bias** to tokens associated with the `remove` operation. This forces the model to explore buffer cleaning strategies rather than defaulting to a safe "noop" state.
- **Style-Curated Curriculum**: Training data is mixed with specific scenarios:
  - `remove_friendly`: Scenarios where the buffer is nearly full and low-value items must be evicted.
  - `fill_first`: Scenarios focused on early-episode identification of relevant data.
- **Adapter-Based Training**: Uses a LoRA adapter (Rank 8-16) to specialize the **Qwen2.5-1.5B** backbone for JSON-structured environment interaction.

#### Reward Signal: Accuracy & Precision Deep Dive
The environment uses a **Shaped Reward Signal** designed to break "noop-collapse" (where models learn to do nothing to avoid penalties).

**The Core Formula (Terminal Evaluation):**
At the end of each episode, the model is judged by a composite success metric:
$$task\_score = 0.6 \times recall + 0.4 \times precision - 0.25 \times incorrect\_rate - 0.15 \times overflow\_rate$$

| Component | Weight/Penalty | Purpose |
| :--- | :--- | :--- |
| **Recall** | `0.60` (Highest) | Ensures relevant items are actually captured. |
| **Precision** | `0.40` | Ensures stored items are actually relevant. |
| **Incorrect Rate** | `-0.25` | Penalizes memory slots wasted on noise. |
| **Overflow Rate** | `-0.15` | Penalizes exceeding the 8-slot capacity. |

**Per-Step Shaped Reward (Training Signal):**
During training, immediate feedback is provided to guide the policy:

| Action | State | Reward | Logic |
| :--- | :--- | :--- | :--- |
| **Add** | Relevant Message | `+0.60` | High incentive for capturing facts. |
| **Add** | Irrelevant Noise | `-0.60` | Heavy penalty for memory pollution. |
| **Remove** | Irrelevant Item | `+0.40` | Reward for active buffer maintenance (Precision). |
| **Remove** | Relevant Item | `-0.50` | Penalty for losing historical context (Recall). |
| **Noop** | Relevant Message | `-0.30` | Penalty for missing information (Omission). |
| **Noop** | Irrelevant Noise | `+0.05` | Minor reward for filtering out noise. |

---

## Results & Benchmarks

The following summary compares the base model, SFT model, and the final GRPO-tuned model across 20 episodes and 1768 steps.

### Benchmark Summary
<div align="center">

![Benchmark Summary](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/benchmark_summary.png)

</div>

### Benchmark Metrics
<div align="center">

![Benchmark Metrics](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/benchmark_metrics.png)

</div>

### Action Distribution Comparison
<div align="center">

![Action Distribution Comparison](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/action_distribution_comparison.png)

</div>

### Mean Reward Per Step
<div align="center">

![Mean Reward Per Step Comparison](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/reward_per_step.jpg)

</div>

### Cumulative Reward Over Episode
<div align="center">

![Cumulative Reward Over Episode Steps](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/reward_accumulation.jpg)

</div>

### Precision Versus Recall Tradeoff
<div align="center">

![Model Precision Versus Recall Tradeoff](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/precision_vs_recall.png)

</div>

### F1 Heatmap Across Episodes
<div align="center">

![F1 Heatmap by Model Episode](https://raw.githubusercontent.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/main/images/f1_heatmap.jpg)

</div>

---

## Logs & Traceability

Detailed execution logs for both training phases are available for audit:
- [GRPO Training Full Log (Text)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/GRPO%20Training%20Log.txt)
- [SFT Training Full Log (Text)](https://github.com/Aditya-Ranjan1234/Long_Horizon_Memory_Final/blob/main/SFT%20Training%20For%20JSON%20Format.txt)

---

## Project Structure

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

## Verification
To verify the environment paths and WebSocket connectivity:
```powershell
python verification/verify_env_paths.py
```

---

## Real-World Applications

As the **AI Agents Boom** accelerates, managing extended context windows becomes a critical bottleneck. Long Horizon Memory solves real-world challenges where **token scarcity** and **inference costs** prohibit feeding massive, unbounded logs into LLMs.

1. **Cybersecurity & Threat Hunting (SOC Agents)**
   Security agents process millions of telemetry logs. A tuned memory model acts as a highly effective filter, discarding routine noise and buffering only anomalous indicators of compromise (IoCs) across hours of activity.
2. **Autonomous Software Engineers**
   When debugging enterprise codebases, agents easily exhaust context windows tracing stack traces. Selective memory allows the agent to retain only the relevant function definitions and error logs without rereading unchanged boilerplate.
3. **Continuous Healthcare Monitoring**
   AI assistants tracking patient vitals over days cannot process 72-hour heart-rate logs at once. Memory-tuned agents preserve only critical spikes or medication changes to synthesize concise handover reports.
4. **Algorithmic Trading & Finance**
   Processing continuous news streams and ticker data is expensive. Memory management enables agents to selectively buffer high-impact earnings statements while ignoring market chatter, keeping context windows lean for low-latency decision making.
5. **Cost-Optimized Customer Support**
   Enterprise AI chatbots handling month-long customer disputes suffer from escalating API costs when reloading entire conversation histories. Optimized memory buffers distill 50-message chains into 8 core facts, drastically reducing token costs while maintaining perfect recall.
