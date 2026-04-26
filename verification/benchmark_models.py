"""
Benchmark four memory-action policies on the Long Horizon Memory env.

Models compared (configurable via env vars / MODEL_SPECS below):

    base_1.5b   - Qwen/Qwen2.5-1.5B-Instruct, no LoRA
    sft_1.5b    - Qwen/Qwen2.5-1.5B-Instruct + memory_action_sft_qwen15b
    grpo_1.5b   - Qwen/Qwen2.5-1.5B-Instruct + memory_action_grpo_qwen15b
    base_7b     - Qwen/Qwen2.5-7B-Instruct, no LoRA (4-bit)

Why this script exists
----------------------
After SFT and GRPO finish we want to *measure* whether RL actually helped, on
the same eval set, with the same prompt format, and the same decoding rules.
Comparing four models in one harness lets us isolate:
  - the value of SFT alone over a base instruct model
  - the marginal value of GRPO over SFT
  - whether a 5x bigger general-purpose model beats a 1.5B model that has been
    trained specifically for the task.

Outputs (under ./benchmark_results/<run_id>/):
  - <model_id>_steps.jsonl       per-step records (one JSON per generation)
  - <model_id>_episodes.jsonl    per-episode aggregates
  - <model_id>_summary.json      aggregated metrics across all episodes
  - comparison.json              all summaries side by side
  - comparison.md                human-readable report incl. failure modes
                                 and per-episode head-to-head highlights
  - comparison_table.txt         ASCII table printed at the end

Run from the project root:

    cd Long_Horizon_Memory
    python benchmark_models.py

Useful overrides:

    N_EPISODES=15                       (default 20)
    BATCH_SIZE=8                        (1.5B; tuning helps on small GPUs)
    BATCH_SIZE_7B=2                     (7B)
    MAX_NEW_TOKENS=32
    SKIP_MODELS=base_7b                 (comma-separated)
    ONLY_MODELS=grpo_1.5b,sft_1.5b      (overrides SKIP_MODELS)
    EPISODES_FILE=episodes_grpo_long.json
    LONG_HORIZON_MEMORY_CAPACITY=16
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Capacity must be set before the env class is imported because the env reads
# LONG_HORIZON_MEMORY_CAPACITY at class-definition time.
os.environ.setdefault("LONG_HORIZON_MEMORY_CAPACITY", "16")

import torch  # noqa: E402


# ── path resolution (matches train_grpo_memory.py) ──────────────────────────


def _resolve_here() -> Path:
    if "__file__" in globals():
        p = Path(__file__).resolve().parent
        if (p / "server").exists() and (p / "data.py").exists():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "server").exists() and (cwd / "data.py").exists():
        return cwd
    if (cwd / "Long_Horizon_Memory" / "server").exists():
        return cwd / "Long_Horizon_Memory"
    raise RuntimeError("Could not locate Long_Horizon_Memory root.")


HERE = _resolve_here()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "server"))


from data import SYSTEM_PROMPT, format_observation  # noqa: E402
from models import LongHorizonMemoryAction  # noqa: E402
from server.long_horizon_memory_environment import (  # noqa: E402
    LongHorizonMemoryEnvironment,
)


# ── benchmark configuration ─────────────────────────────────────────────────


@dataclass
class BenchConfig:
    n_episodes: int = int(os.getenv("N_EPISODES", "20"))
    seed: int = int(os.getenv("BENCH_SEED", "20260427"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "32"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))
    batch_size_7b: int = int(os.getenv("BATCH_SIZE_7B", "2"))
    episodes_file: str = os.getenv("EPISODES_FILE", "episodes_grpo_long.json")
    capacity: int = int(os.getenv("LONG_HORIZON_MEMORY_CAPACITY", "16"))

    skip_models: List[str] = field(
        default_factory=lambda: [
            s.strip() for s in os.getenv("SKIP_MODELS", "").split(",") if s.strip()
        ]
    )
    only_models: List[str] = field(
        default_factory=lambda: [
            s.strip() for s in os.getenv("ONLY_MODELS", "").split(",") if s.strip()
        ]
    )

    output_root: str = os.getenv("OUTPUT_ROOT", str(HERE / "benchmark_results"))
    run_id: str = os.getenv("BENCH_RUN_ID", datetime.now().strftime("run_%Y%m%d_%H%M%S"))


CFG = BenchConfig()


@dataclass
class ModelSpec:
    model_id: str
    hf_name: str
    adapter_path: Optional[str]
    quant_4bit: bool = True
    description: str = ""


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        model_id="base_1.5b",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=None,
        description="Qwen2.5-1.5B-Instruct, no fine-tuning. Floor for the action format.",
    ),
    ModelSpec(
        model_id="sft_1.5b",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=str(HERE / "memory_action_sft_qwen15b"),
        description="SFT only. Same base + memory_action_sft_qwen15b LoRA.",
    ),
    ModelSpec(
        model_id="grpo_1.5b",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=str(HERE / "memory_action_grpo_qwen15b"),
        description="GRPO continued from SFT. memory_action_grpo_qwen15b LoRA.",
    ),
    ModelSpec(
        model_id="base_7b",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=None,
        description="Larger general-purpose model, 4-bit. No task-specific training.",
    ),
]


def selected_models() -> List[ModelSpec]:
    if CFG.only_models:
        wanted = set(CFG.only_models)
        return [m for m in MODEL_SPECS if m.model_id in wanted]
    skip = set(CFG.skip_models)
    return [m for m in MODEL_SPECS if m.model_id not in skip]


# ── prompt + completion helpers ─────────────────────────────────────────────


def to_obs_dict(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    return {
        "domain": env.current_domain,
        "task_name": env.current_difficulty,
        "memory": [{"index": i, "text": m.get("text", "")} for i, m in enumerate(env.memory)],
        "new_message": (env._current_message() or {}).get("text", ""),
    }


def build_chat_prompt(obs: Dict[str, Any], tokenizer) -> str:
    body = format_observation(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{body}\n<|assistant|>\n"


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_first_json(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def parse_completion(text: str) -> Tuple[Optional[LongHorizonMemoryAction], str]:
    obj_str = _extract_first_json(text)
    if obj_str is None:
        return None, "no_json"
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError:
        return None, "json_invalid"
    try:
        return LongHorizonMemoryAction.model_validate(obj), "ok"
    except Exception:
        return None, "pydantic_invalid"


def fallback_action() -> LongHorizonMemoryAction:
    """When parsing fails the env still needs *something* to step. We use noop
    so we don't punish the env state further; the reward is already low for
    the unparseable completion via ``score_action``."""
    return LongHorizonMemoryAction(operation="noop")


# ── episode-store swap (so the env reads the long-horizon dataset) ──────────


def use_episodes(env_dir: Path, episodes_filename: str) -> Optional[Path]:
    src = env_dir / episodes_filename
    if not src.exists():
        print(f"[bench] episodes file {src} missing; using server/episodes.json as-is")
        return None
    dst = env_dir / "episodes.json"
    backup = env_dir / ".episodes_backup_for_bench.json"
    if not backup.exists():
        backup.write_bytes(dst.read_bytes())
    dst.write_bytes(src.read_bytes())
    return backup


def restore_episodes(backup: Optional[Path], env_dir: Path) -> None:
    if backup is None or not backup.exists():
        return
    (env_dir / "episodes.json").write_bytes(backup.read_bytes())
    backup.unlink()


def select_eval_episode_ids(n: int, seed: int) -> List[int]:
    """Pick n distinct episode_ids deterministically from the loaded eps file.
    The same seed yields the same set across runs and across models."""
    env = LongHorizonMemoryEnvironment()
    all_ids = [int(ep.get("episode_id", i + 1)) for i, ep in enumerate(env.episodes)]
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    return all_ids[: min(n, len(all_ids))]


# ── model loading ───────────────────────────────────────────────────────────


def load_model_and_tokenizer(spec: ModelSpec):
    """Load a Qwen base + optional LoRA adapter in 4-bit on a single GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the benchmark in 4-bit.")

    print(f"[bench] [{spec.model_id}] loading tokenizer: {spec.hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[bench] [{spec.model_id}] loading 4-bit base: {spec.hf_name}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": 0},
    )
    model.eval()
    model.config.use_cache = True

    if spec.adapter_path is not None:
        from peft import PeftModel

        adapter = Path(spec.adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
        print(f"[bench] [{spec.model_id}] attaching LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()

    return model, tokenizer


def free_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── batched generation ──────────────────────────────────────────────────────


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    """Greedy decode prompts in chunks. Returns the *new* text per prompt."""
    out: List[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        gen = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lens = encoded["attention_mask"].sum(dim=1)
        for i in range(gen.shape[0]):
            full_ids = gen[i].tolist()
            input_len = int(prompt_lens[i].item())
            new_ids = full_ids[input_len:] if input_len > 0 else full_ids
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            out.append(text)
    return out


# ── core evaluation loop ────────────────────────────────────────────────────


def _state_metrics(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    cap = env.MEMORY_CAPACITY
    n = len(env.memory)
    n_irrel = sum(1 for m in env.memory if not m.get("isRelevant", False))
    n_rel = n - n_irrel
    msg = env._current_message() or {}
    return {
        "memory_fill": n,
        "memory_capacity": cap,
        "memory_full": n >= cap,
        "memory_irrelevant_slots": n_irrel,
        "memory_relevant_slots": n_rel,
        "msg_relevant": bool(msg.get("isRelevant", False)),
    }


def evaluate_model_on_episodes(
    spec: ModelSpec,
    model,
    tokenizer,
    episode_ids: List[int],
    cfg: BenchConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a single model. Episodes are stepped forward in lockstep so we
    can batch generation across them. Returns (step_records, episode_summaries).
    """
    bs = cfg.batch_size_7b if "7b" in spec.model_id else cfg.batch_size

    envs: List[LongHorizonMemoryEnvironment] = []
    for eid in episode_ids:
        env = LongHorizonMemoryEnvironment()
        env._episode_id_override = eid
        env._set_random_episode()
        envs.append(env)

    n_envs = len(envs)
    step_records: List[Dict[str, Any]] = []
    ep_step_count = [0] * n_envs
    ep_total_reward = [0.0] * n_envs
    ep_max_fill = [0] * n_envs
    ep_ops = [Counter() for _ in range(n_envs)]
    ep_errors = [Counter() for _ in range(n_envs)]
    ep_parse_status = [Counter() for _ in range(n_envs)]
    ep_correct_remove = [0] * n_envs
    ep_wrong_remove = [0] * n_envs

    round_idx = 0
    t0 = time.time()
    while any(not e._done for e in envs):
        active = [(i, e) for i, e in enumerate(envs) if not e._done]
        prompts = [build_chat_prompt(to_obs_dict(e), tokenizer) for _, e in active]
        completions = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=bs,
        )

        for (i, env), completion_text in zip(active, completions):
            pre_state = _state_metrics(env)
            action, status = parse_completion(completion_text)
            applied = action if action is not None else fallback_action()

            obs = env.step(applied)
            reward = float(obs.reward)
            error = env.last_action_error  # set inside step before reward calc

            op = applied.operation
            ep_ops[i][op] += 1
            ep_parse_status[i][status] += 1
            if error:
                ep_errors[i][error] += 1
            if op == "remove":
                if reward > 0:
                    ep_correct_remove[i] += 1
                else:
                    ep_wrong_remove[i] += 1
            ep_total_reward[i] += reward
            ep_step_count[i] += 1
            ep_max_fill[i] = max(ep_max_fill[i], pre_state["memory_fill"])

            step_records.append(
                {
                    "model": spec.model_id,
                    "episode_id": episode_ids[i],
                    "step": ep_step_count[i],
                    "round": round_idx,
                    "pre_state": pre_state,
                    "completion_text": completion_text,
                    "parse_status": status,
                    "operation_requested": (action.operation if action else None),
                    "operation_applied": op,
                    "remove_index": (action.remove_index if action else None),
                    "task_reward": reward,
                    "env_error": error,
                    "memory_after": [
                        {"isRelevant": bool(m.get("isRelevant", False))}
                        for m in env.memory
                    ],
                }
            )

        round_idx += 1
        active_remaining = sum(1 for e in envs if not e._done)
        if round_idx % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[bench] [{spec.model_id}] round={round_idx} "
                f"active={active_remaining}/{n_envs} elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"[bench] [{spec.model_id}] finished {n_envs} eps in {elapsed:.1f}s")

    episode_summaries: List[Dict[str, Any]] = []
    for i, eid in enumerate(episode_ids):
        env = envs[i]
        n = ep_step_count[i] or 1
        # Final precision / recall / F1 from env state (running metrics).
        m = env._memory_stats()
        kept = len(env.memory)
        total_seen = max(1, env.total_relevant_seen)
        precision = m["correct"] / kept if kept > 0 else 1.0
        recall = m["correct"] / total_seen
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        episode_summaries.append(
            {
                "model": spec.model_id,
                "episode_id": eid,
                "domain": env.current_domain,
                "difficulty": env.current_difficulty,
                "n_messages": len(env.messages),
                "total_relevant_in_episode": env.total_relevant_in_episode,
                "n_steps": ep_step_count[i],
                "max_memory_fill": ep_max_fill[i],
                "final_memory_fill": kept,
                "final_memory_correct": m["correct"],
                "final_memory_incorrect": m["incorrect"],
                "final_precision": precision,
                "final_recall": recall,
                "final_f1": f1,
                "mean_step_reward": ep_total_reward[i] / n,
                "total_reward": ep_total_reward[i],
                "ops": dict(ep_ops[i]),
                "errors": dict(ep_errors[i]),
                "parse_status": dict(ep_parse_status[i]),
                "correct_remove": ep_correct_remove[i],
                "wrong_remove": ep_wrong_remove[i],
            }
        )

    return step_records, episode_summaries


# ── aggregation + comparison ────────────────────────────────────────────────


def summarize_model(model_id: str, step_records: List[Dict[str, Any]], ep_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_records:
        return {"model": model_id, "n_steps": 0}

    n_steps = len(step_records)
    parse_counter: Counter = Counter()
    op_counter: Counter = Counter()
    err_counter: Counter = Counter()
    correct_remove = 0
    wrong_remove = 0
    rewards: List[float] = []
    full_mem_steps = 0
    long_horizon_steps = 0
    for s in step_records:
        parse_counter[s["parse_status"]] += 1
        op_counter[s["operation_applied"]] += 1
        if s["env_error"]:
            err_counter[s["env_error"]] += 1
        if s["operation_applied"] == "remove":
            if s["task_reward"] > 0:
                correct_remove += 1
            else:
                wrong_remove += 1
        rewards.append(s["task_reward"])
        fill = s["pre_state"]["memory_fill"]
        if s["pre_state"]["memory_full"]:
            full_mem_steps += 1
        if fill >= 8:
            long_horizon_steps += 1

    n_ep = len(ep_summaries)
    f1s = [e["final_f1"] for e in ep_summaries]
    precs = [e["final_precision"] for e in ep_summaries]
    recs = [e["final_recall"] for e in ep_summaries]
    max_fills = [e["max_memory_fill"] for e in ep_summaries]

    summary = {
        "model": model_id,
        "n_episodes": n_ep,
        "n_steps": n_steps,
        "parse_rate": parse_counter["ok"] / n_steps,
        "parse_status": dict(parse_counter),
        "action_dist": {k: op_counter[k] / n_steps for k in ("add", "remove", "noop")},
        "remove_correctness": (
            correct_remove / max(1, correct_remove + wrong_remove)
        ),
        "remove_correct_count": correct_remove,
        "remove_wrong_count": wrong_remove,
        "env_errors": dict(err_counter),
        "mean_step_reward": sum(rewards) / n_steps,
        "median_step_reward": _median(rewards),
        "mean_final_f1": _mean(f1s),
        "median_final_f1": _median(f1s),
        "mean_final_precision": _mean(precs),
        "mean_final_recall": _mean(recs),
        "mean_max_memory_fill": _mean(max_fills),
        "long_horizon_step_pct": long_horizon_steps / n_steps,
        "memory_full_step_pct": full_mem_steps / n_steps,
    }
    return summary


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def head_to_head(summaries: List[Dict[str, Any]], all_episodes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Return per-episode wins (highest F1) per model and top deltas."""
    if "grpo_1.5b" not in all_episodes:
        return {}
    episode_ids = sorted({e["episode_id"] for e in all_episodes["grpo_1.5b"]})
    by_model_ep = {
        m: {e["episode_id"]: e for e in all_episodes[m]}
        for m in all_episodes
    }
    wins = Counter()
    grpo_vs_sft: List[Tuple[int, float, float, float]] = []
    grpo_vs_base: List[Tuple[int, float, float, float]] = []
    grpo_vs_7b: List[Tuple[int, float, float, float]] = []

    for eid in episode_ids:
        scores = {m: by_model_ep[m].get(eid, {}).get("final_f1", float("nan")) for m in all_episodes}
        best_model = max(scores, key=lambda m: (scores[m] if not math.isnan(scores[m]) else -1))
        wins[best_model] += 1
        if "sft_1.5b" in scores:
            grpo_vs_sft.append((eid, scores["grpo_1.5b"], scores["sft_1.5b"], scores["grpo_1.5b"] - scores["sft_1.5b"]))
        if "base_1.5b" in scores:
            grpo_vs_base.append((eid, scores["grpo_1.5b"], scores["base_1.5b"], scores["grpo_1.5b"] - scores["base_1.5b"]))
        if "base_7b" in scores:
            grpo_vs_7b.append((eid, scores["grpo_1.5b"], scores["base_7b"], scores["grpo_1.5b"] - scores["base_7b"]))

    grpo_vs_sft.sort(key=lambda r: r[3], reverse=True)
    grpo_vs_base.sort(key=lambda r: r[3], reverse=True)
    grpo_vs_7b.sort(key=lambda r: r[3], reverse=True)

    return {
        "wins": dict(wins),
        "n_episodes": len(episode_ids),
        "top_grpo_gains_vs_sft": grpo_vs_sft[:5],
        "top_sft_gains_over_grpo": list(reversed(grpo_vs_sft))[:5],
        "top_grpo_gains_vs_base": grpo_vs_base[:5],
        "top_grpo_gains_vs_7b": grpo_vs_7b[:5],
    }


def collect_failure_modes(step_records: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """Return small samples of the most common failure modes for the report."""
    parse_failures = [s for s in step_records if s["parse_status"] != "ok"]
    env_errors = [s for s in step_records if s["env_error"]]

    parse_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in parse_failures:
        if len(parse_examples[s["parse_status"]]) < top_k:
            parse_examples[s["parse_status"]].append(
                {
                    "episode_id": s["episode_id"],
                    "step": s["step"],
                    "completion_text": s["completion_text"][:200],
                }
            )
    error_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in env_errors:
        if len(error_examples[s["env_error"]]) < top_k:
            error_examples[s["env_error"]].append(
                {
                    "episode_id": s["episode_id"],
                    "step": s["step"],
                    "operation_applied": s["operation_applied"],
                    "remove_index": s["remove_index"],
                    "memory_fill": s["pre_state"]["memory_fill"],
                    "completion_text": s["completion_text"][:200],
                }
            )
    return {
        "parse_failure_examples": dict(parse_examples),
        "env_error_examples": dict(error_examples),
    }


# ── reporting ───────────────────────────────────────────────────────────────


def render_table(summaries: List[Dict[str, Any]]) -> str:
    rows = [
        ("model", lambda s: s["model"]),
        ("n_eps", lambda s: f"{s['n_episodes']}"),
        ("n_steps", lambda s: f"{s['n_steps']}"),
        ("parse_ok %", lambda s: f"{s['parse_rate']*100:5.1f}"),
        ("add %", lambda s: f"{s['action_dist']['add']*100:5.1f}"),
        ("remove %", lambda s: f"{s['action_dist']['remove']*100:5.1f}"),
        ("noop %", lambda s: f"{s['action_dist']['noop']*100:5.1f}"),
        ("rem-correct %", lambda s: f"{s['remove_correctness']*100:5.1f}"),
        ("mean step rwd", lambda s: f"{s['mean_step_reward']:+.3f}"),
        ("mean final F1", lambda s: f"{s['mean_final_f1']:.3f}"),
        ("mean precision", lambda s: f"{s['mean_final_precision']:.3f}"),
        ("mean recall", lambda s: f"{s['mean_final_recall']:.3f}"),
        ("mean max fill", lambda s: f"{s['mean_max_memory_fill']:.1f}"),
        ("long-horiz %", lambda s: f"{s['long_horizon_step_pct']*100:5.1f}"),
        ("mem-full %", lambda s: f"{s['memory_full_step_pct']*100:5.1f}"),
    ]
    name_w = max(len(r[0]) for r in rows) + 1
    col_w = max(15, max((len(r[1](s)) for s in summaries for r in rows), default=15))
    lines = []
    header = " " * name_w + "".join(f"{s['model']:>{col_w}}" for s in summaries)
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for label, fn in rows:
        line = f"{label:<{name_w}}" + "".join(f"{fn(s):>{col_w}}" for s in summaries)
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def render_markdown_report(
    summaries: List[Dict[str, Any]],
    h2h: Dict[str, Any],
    failures: Dict[str, Dict[str, Any]],
    cfg: BenchConfig,
) -> str:
    lines: List[str] = []
    lines.append(f"# Long-Horizon Memory: model comparison ({cfg.run_id})")
    lines.append("")
    lines.append(f"- Episodes: **{cfg.n_episodes}** from `{cfg.episodes_file}`")
    lines.append(f"- Memory capacity: **{cfg.capacity}**")
    lines.append(f"- Decoding: greedy, max_new_tokens={cfg.max_new_tokens}")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("```")
    lines.append(render_table(summaries))
    lines.append("```")
    lines.append("")

    if h2h:
        lines.append("## Head-to-head F1")
        lines.append("")
        lines.append(f"Per-episode wins (highest final_f1): `{h2h.get('wins', {})}` over {h2h['n_episodes']} episodes.")
        lines.append("")
        if h2h.get("top_grpo_gains_vs_sft"):
            lines.append("### Top 5 episodes where GRPO beat SFT (by F1 delta)")
            lines.append("")
            lines.append("| episode | grpo F1 | sft F1 | delta |")
            lines.append("|---------|---------|--------|-------|")
            for eid, g, s, d in h2h["top_grpo_gains_vs_sft"]:
                lines.append(f"| {eid} | {g:.3f} | {s:.3f} | {d:+.3f} |")
            lines.append("")
        if h2h.get("top_sft_gains_over_grpo"):
            lines.append("### Top 5 episodes where SFT beat GRPO (regressions to investigate)")
            lines.append("")
            lines.append("| episode | grpo F1 | sft F1 | delta |")
            lines.append("|---------|---------|--------|-------|")
            for eid, g, s, d in h2h["top_sft_gains_over_grpo"]:
                lines.append(f"| {eid} | {g:.3f} | {s:.3f} | {d:+.3f} |")
            lines.append("")
        if h2h.get("top_grpo_gains_vs_7b"):
            lines.append("### Top 5 episodes where GRPO 1.5B beat base 7B")
            lines.append("")
            lines.append("| episode | grpo F1 | 7b F1 | delta |")
            lines.append("|---------|---------|-------|-------|")
            for eid, g, s, d in h2h["top_grpo_gains_vs_7b"]:
                lines.append(f"| {eid} | {g:.3f} | {s:.3f} | {d:+.3f} |")
            lines.append("")

    lines.append("## Failure modes per model")
    lines.append("")
    for model_id, fails in failures.items():
        lines.append(f"### {model_id}")
        lines.append("")
        if fails.get("parse_failure_examples"):
            lines.append("**Parse failures** (sample):")
            lines.append("")
            for status, exs in fails["parse_failure_examples"].items():
                lines.append(f"- `{status}` (showing {len(exs)}):")
                for ex in exs:
                    lines.append(f"  - ep={ex['episode_id']} step={ex['step']} text={ex['completion_text']!r}")
            lines.append("")
        if fails.get("env_error_examples"):
            lines.append("**Env errors** (after parse, action was illegal):")
            lines.append("")
            for err, exs in fails["env_error_examples"].items():
                lines.append(f"- `{err}` (showing {len(exs)}):")
                for ex in exs:
                    lines.append(
                        f"  - ep={ex['episode_id']} step={ex['step']} "
                        f"op={ex['operation_applied']} remove_index={ex['remove_index']} "
                        f"mem_fill={ex['memory_fill']}"
                    )
            lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── orchestration ───────────────────────────────────────────────────────────


def main() -> None:
    out_dir = Path(CFG.output_root) / CFG.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] writing results to {out_dir}")

    env_dir = HERE / "server"
    backup = use_episodes(env_dir, CFG.episodes_file)
    try:
        episode_ids = select_eval_episode_ids(CFG.n_episodes, CFG.seed)
        print(f"[bench] selected {len(episode_ids)} episodes: {episode_ids[:10]}{'...' if len(episode_ids) > 10 else ''}")
        write_json(out_dir / "config.json", {**asdict(CFG), "episode_ids": episode_ids})

        all_summaries: List[Dict[str, Any]] = []
        all_episode_summaries_by_model: Dict[str, List[Dict[str, Any]]] = {}
        all_failures: Dict[str, Dict[str, Any]] = {}

        for spec in selected_models():
            print(f"\n[bench] === {spec.model_id} ===  ({spec.description})")
            try:
                model, tokenizer = load_model_and_tokenizer(spec)
            except Exception as e:
                print(f"[bench] [{spec.model_id}] failed to load: {e!r}; skipping")
                continue
            try:
                steps, eps = evaluate_model_on_episodes(spec, model, tokenizer, episode_ids, CFG)
            finally:
                free_model(model)

            summary = summarize_model(spec.model_id, steps, eps)
            failures = collect_failure_modes(steps)

            write_jsonl(out_dir / f"{spec.model_id}_steps.jsonl", steps)
            write_jsonl(out_dir / f"{spec.model_id}_episodes.jsonl", eps)
            write_json(out_dir / f"{spec.model_id}_summary.json", summary)
            write_json(out_dir / f"{spec.model_id}_failures.json", failures)

            all_summaries.append(summary)
            all_episode_summaries_by_model[spec.model_id] = eps
            all_failures[spec.model_id] = failures

            print("\n[bench] partial summary so far:")
            print(render_table(all_summaries))

        h2h = head_to_head(all_summaries, all_episode_summaries_by_model)
        report_md = render_markdown_report(all_summaries, h2h, all_failures, CFG)
        table = render_table(all_summaries)

        write_json(out_dir / "comparison.json", {
            "config": asdict(CFG),
            "summaries": all_summaries,
            "head_to_head": h2h,
        })
        (out_dir / "comparison.md").write_text(report_md, encoding="utf-8")
        (out_dir / "comparison_table.txt").write_text(table, encoding="utf-8")

        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)
        print(table)
        print(f"\n[bench] Wrote: {out_dir / 'comparison.md'}")
    finally:
        restore_episodes(backup, env_dir)


if __name__ == "__main__":
    main()
