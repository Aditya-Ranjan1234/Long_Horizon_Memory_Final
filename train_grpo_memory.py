# Jupyter-safe GRPO trainer for Long Horizon Memory
# Includes:
# - notebook-safe path resolution
# - robust TRL config compatibility
# - generation_batch_size fix
# - reward/completion JSONL logging + console summaries
# - safer single-GPU loading for notebook stability

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Long-horizon training: lift the env's memory capacity from 8 -> 16 so we
# actually exercise eviction. The env reads this env var at class-definition
# time, so it must be set before the env module is imported below.
os.environ.setdefault("LONG_HORIZON_MEMORY_CAPACITY", "16")

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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

    raise RuntimeError(
        "Could not locate Long_Horizon_Memory root. "
        "Run this notebook from repo root or Long_Horizon_Memory directory."
    )


HERE = _resolve_here()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "server"))

from data import SYSTEM_PROMPT, format_observation  # noqa: E402
from models import LongHorizonMemoryAction  # noqa: E402
from server.long_horizon_memory_environment import (  # noqa: E402
    LongHorizonMemoryEnvironment,
    score_action,
)


@dataclass
class TrainConfig:
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path: str = os.getenv("ADAPTER_PATH", str(HERE / "memory_action_sft_qwen15b"))
    output_dir: str = os.getenv("OUTPUT_DIR", str(HERE / "memory_action_grpo_qwen15b"))
    # Long-horizon dataset: episodes 60-110 msgs, 19-20 relevant per ep,
    # capacity is set to 16 below so memory actually fills and we get real
    # eviction pressure (remove must be used).
    episodes_file: str = os.getenv("EPISODES_FILE", "episodes_grpo_long.json")

    num_train_epochs: int = int(os.getenv("NUM_TRAIN_EPOCHS", "2"))
    per_device_train_batch_size: int = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "5e-6"))
    # Bigger group => non-zero advantages even when one op dominates locally
    num_generations: int = int(os.getenv("NUM_GENERATIONS", "4"))
    beta: float = float(os.getenv("BETA", "0.04"))
    # Capacity 16 means memory list can grow long; bump prompt budget.
    max_prompt_length: int = int(os.getenv("MAX_PROMPT_LENGTH", "768"))
    max_completion_length: int = int(os.getenv("MAX_COMPLETION_LENGTH", "32"))
    # Sampling diversity for exploration of `remove` and `remove_index`.
    temperature: float = float(os.getenv("GRPO_TEMPERATURE", "1.3"))
    top_p: float = float(os.getenv("GRPO_TOP_P", "0.95"))
    top_k: int = int(os.getenv("GRPO_TOP_K", "50"))

    # Memory capacity used for long-horizon training. Pushed into env via
    # LONG_HORIZON_MEMORY_CAPACITY before the env class is imported (the env
    # reads this at class-definition time, so we set it as an env var here
    # if it has not already been set explicitly).
    memory_capacity: int = int(os.getenv("LONG_HORIZON_MEMORY_CAPACITY", "16"))

    dataset_size: int = int(os.getenv("DATASET_SIZE", "512"))
    rollouts_per_episode: int = int(os.getenv("ROLLOUTS_PER_EPISODE", "4"))
    seed: int = int(os.getenv("GRPO_SEED", "20260426"))

    # Reward shaping knobs for diversity / exploration during GRPO.
    diversity_bonus: float = float(os.getenv("GRPO_DIVERSITY_BONUS", "0.05"))
    diversity_penalty: float = float(os.getenv("GRPO_DIVERSITY_PENALTY", "0.05"))

    # Logging
    reward_log_file: str = os.getenv("REWARD_LOG_FILE", str(HERE / "grpo_reward_log.jsonl"))
    print_each_generation: bool = os.getenv("PRINT_EACH_GENERATION", "1") == "1"
    print_summary_every: int = int(os.getenv("PRINT_SUMMARY_EVERY", "20"))
    max_logged_text_chars: int = int(os.getenv("MAX_LOGGED_TEXT_CHARS", "300"))


CFG = TrainConfig()


def use_episodes(env_dir: Path, episodes_filename: str) -> Optional[Path]:
    env_dir = Path(env_dir)
    src = env_dir / episodes_filename
    if not src.exists():
        print(f"[grpo] episodes file {src} missing; using server/episodes.json as-is")
        return None

    dst = env_dir / "episodes.json"
    backup = env_dir / ".episodes_backup_for_grpo.json"
    if not backup.exists():
        backup.write_bytes(dst.read_bytes())
    dst.write_bytes(src.read_bytes())
    return backup


def restore_episodes(backup: Optional[Path], env_dir: Path) -> None:
    env_dir = Path(env_dir)
    if backup is None or not backup.exists():
        return
    dst = env_dir / "episodes.json"
    dst.write_bytes(backup.read_bytes())
    backup.unlink()


def to_obs_dict(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    return {
        "domain": env.current_domain,
        "task_name": env.current_difficulty,
        "memory": [{"index": i, "text": m.get("text", "")} for i, m in enumerate(env.memory)],
        "new_message": (env._current_message() or {}).get("text", ""),
    }


def to_score_state(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    msg = env._current_message()
    is_last = env.total_message_number == len(env.messages) - 1
    return {
        "memory": [
            {"text": m.get("text", ""), "isRelevant": bool(m.get("isRelevant", False))}
            for m in env.memory
        ],
        "message": None if msg is None else {
            "text": msg.get("text", ""),
            "isRelevant": bool(msg.get("isRelevant", False)),
        },
        "total_relevant_seen": env.total_relevant_seen,
        "is_last_step": is_last,
    }


def build_chat_prompt(observation_dict: Dict[str, Any], tokenizer) -> str:
    body = format_observation(observation_dict)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{body}\n<|assistant|>\n"


def _record_state(rows, env, fill_hist, op_hist, tokenizer) -> None:
    obs = to_obs_dict(env)
    score_state = to_score_state(env)
    prompt = build_chat_prompt(obs, tokenizer)
    rows.append({
        "prompt": prompt,
        "state_json": json.dumps(score_state, ensure_ascii=False),
    })
    fill_hist[len(env.memory)] += 1
    msg = env._current_message() or {}
    op_hist[("rel" if msg.get("isRelevant") else "irrel")] += 1


def _force_fill_to(env, target_fill: int, accept_irrelevant_prob: float, rng) -> None:
    """
    Drive the env forward by force-adding messages until memory has at least
    ``target_fill`` items or the episode ends. We always add relevant messages,
    and add irrelevant ones with probability ``accept_irrelevant_prob`` so the
    final memory has a controllable mix of relevant / irrelevant slots.
    """
    while not env._done and len(env.memory) < target_fill:
        msg = env._current_message() or {}
        is_rel = bool(msg.get("isRelevant", False))
        if is_rel or rng.random() < accept_irrelevant_prob:
            env.step(LongHorizonMemoryAction(operation="add"))
        else:
            env.step(LongHorizonMemoryAction(operation="noop"))


def build_dataset(tokenizer) -> Dataset:
    """
    Build a curated training set that mixes three kinds of (state, prompt)
    pairs so GRPO sees:

      1. ``oracle_natural`` (35%) - states reached by following the oracle.
         Distributionally close to what a competent policy will see.
      2. ``fill_first``    (35%) - states where memory is at or near capacity,
         forcing the policy to choose between ``add`` (rejected when full),
         ``remove`` (correct when an irrelevant slot exists), or ``noop``.
      3. ``remove_friendly`` (30%) - states where memory holds several
         irrelevant items, so a literal ``remove`` is the highest-reward op.

    A memory-fullness histogram is printed so you can see that the trainer
    is actually exercising long-horizon eviction (slots in the 8-16 range).
    """
    rng = random.Random(CFG.seed)
    capacity = LongHorizonMemoryEnvironment.MEMORY_CAPACITY
    env = LongHorizonMemoryEnvironment()

    rows: List[Dict[str, Any]] = []
    fill_hist: Counter = Counter()
    op_hist: Counter = Counter()
    style_hist: Counter = Counter()

    near_full = max(2, capacity - 2)

    while len(rows) < CFG.dataset_size:
        env.reset()
        style = rng.choices(
            ["oracle_natural", "fill_first", "remove_friendly"],
            weights=[0.35, 0.35, 0.30],
            k=1,
        )[0]
        style_hist[style] += 1

        if style == "fill_first":
            _force_fill_to(env, capacity, accept_irrelevant_prob=0.4, rng=rng)
        elif style == "remove_friendly":
            _force_fill_to(env, near_full, accept_irrelevant_prob=0.85, rng=rng)

        steps_in_ep = 0
        max_steps = max(8, CFG.rollouts_per_episode * 8)
        while not env._done and len(rows) < CFG.dataset_size and steps_in_ep < max_steps:
            _record_state(rows, env, fill_hist, op_hist, tokenizer)

            roll = rng.random()
            if roll < 0.4:
                action = env._oracle_action()
            elif roll < 0.65:
                action = LongHorizonMemoryAction(operation="add")
            elif roll < 0.85 and env.memory:
                action = LongHorizonMemoryAction(
                    operation="remove",
                    remove_index=rng.randrange(len(env.memory)),
                )
            else:
                action = LongHorizonMemoryAction(operation="noop")
            env.step(action)
            steps_in_ep += 1

    rng.shuffle(rows)
    rows = rows[: CFG.dataset_size]

    n = sum(fill_hist.values())
    if n > 0:
        print("[grpo] dataset memory utilization (slots filled when state was recorded):")
        for lo, hi, label in [
            (0, 0, "empty"),
            (1, 3, "1-3"),
            (4, 7, "4-7"),
            (8, 15, "8-15  (long-horizon)"),
            (16, 999, "16+   (full / capacity)"),
        ]:
            count = sum(c for k, c in fill_hist.items() if lo <= k <= hi)
            print(f"   {label:24s}: {count:5d} ({count / n:.1%})")
        print(f"[grpo] curated style mix: {dict(style_hist)}")
        print(f"[grpo] current-msg label mix in states: {dict(op_hist)}")
        print(f"[grpo] capacity={capacity}  episodes_file={CFG.episodes_file}")

    return Dataset.from_list(rows)


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_first_json(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _completion_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        last = c[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(c)


def _parse_action(text: str) -> Tuple[Optional[LongHorizonMemoryAction], str]:
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


class RewardMonitor:
    def __init__(self, log_file: str, print_each_generation: bool, print_summary_every: int, max_chars: int):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.print_each_generation = print_each_generation
        self.print_summary_every = max(1, print_summary_every)
        self.max_chars = max_chars

        self.call_idx = 0
        self.total_samples = 0
        self.total_reward_sum = 0.0
        self.task_reward_sum = 0.0
        self.format_reward_sum = 0.0

        self.parse_status = Counter()
        self.action_ops = Counter()

    def _clip(self, x: str) -> str:
        x = str(x) if x is not None else ""
        return x if len(x) <= self.max_chars else (x[: self.max_chars] + "...<truncated>")

    def log_one(
        self,
        sample_idx: int,
        completion_text: str,
        parse_status: str,
        action_obj: Optional[LongHorizonMemoryAction],
        task_reward: float,
        format_reward: float,
        total_reward: float,
        prompt_text: Optional[str] = None,
    ):
        self.total_samples += 1
        self.total_reward_sum += total_reward
        self.task_reward_sum += task_reward
        self.format_reward_sum += format_reward
        self.parse_status[parse_status] += 1

        op = None
        remove_idx = None
        if action_obj is not None:
            op = action_obj.operation
            remove_idx = action_obj.remove_index
            self.action_ops[op] += 1

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "reward_call_idx": self.call_idx,
            "sample_idx": sample_idx,
            "parse_status": parse_status,
            "operation": op,
            "remove_index": remove_idx,
            "task_reward": float(task_reward),
            "format_reward": float(format_reward),
            "total_reward": float(total_reward),
            "completion_text": self._clip(completion_text),
            "prompt_text": self._clip(prompt_text or ""),
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.print_each_generation:
            print(
                f"[reward] call={self.call_idx} sample={sample_idx} "
                f"status={parse_status} op={op} "
                f"task={task_reward:+.3f} fmt={format_reward:+.3f} total={total_reward:+.3f}"
            )

    def maybe_print_summary(self):
        if self.call_idx % self.print_summary_every != 0:
            return

        n = max(1, self.total_samples)
        parse_ok = self.parse_status["ok"] / n
        mean_total = self.total_reward_sum / n
        mean_task = self.task_reward_sum / n
        mean_fmt = self.format_reward_sum / n

        action_total = sum(self.action_ops.values())
        add_rate = self.action_ops["add"] / action_total if action_total else 0.0
        rem_rate = self.action_ops["remove"] / action_total if action_total else 0.0
        noop_rate = self.action_ops["noop"] / action_total if action_total else 0.0

        print(
            "[reward-summary] "
            f"calls={self.call_idx} samples={n} "
            f"parse_ok={parse_ok:.2%} "
            f"mean_total={mean_total:+.4f} mean_task={mean_task:+.4f} mean_fmt={mean_fmt:+.4f} "
            f"ops(add/remove/noop)=({add_rate:.1%}/{rem_rate:.1%}/{noop_rate:.1%}) "
            f"fail(no_json/json_invalid/pydantic_invalid)="
            f"({self.parse_status['no_json']}/{self.parse_status['json_invalid']}/{self.parse_status['pydantic_invalid']})"
        )


REWARD_MONITOR = RewardMonitor(
    log_file=CFG.reward_log_file,
    print_each_generation=CFG.print_each_generation,
    print_summary_every=CFG.print_summary_every,
    max_chars=CFG.max_logged_text_chars,
)


def combined_reward_fn(completions, **kwargs) -> List[float]:
    """
    Compute task + format + diversity rewards.

    The diversity term is what fixes the noop-collapse / advantage-zero issue:
    if every completion in a GRPO group picks the same operation, we apply a
    small penalty (so the group is not stuck at zero advantage); if the group
    samples >=3 distinct ops we apply a small bonus to encourage exploration.
    The diversity term is small relative to ``score_action`` so it cannot
    drown out the actual task signal.
    """
    states = kwargs.get("state_json")
    prompts = kwargs.get("prompts", None) or kwargs.get("prompt", None)

    if states is None:
        return [0.0] * len(completions)

    capacity = LongHorizonMemoryEnvironment.MEMORY_CAPACITY
    REWARD_MONITOR.call_idx += 1

    parsed: List[Tuple[str, Optional[LongHorizonMemoryAction], str]] = []
    for comp in completions:
        text = _completion_text(comp)
        action, status = _parse_action(text)
        parsed.append((text, action, status))

    group_members: Dict[str, List[int]] = defaultdict(list)
    for i, raw_state in enumerate(states):
        state_key = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, sort_keys=True)
        group_members[state_key].append(i)

    diversity_reward = [0.0] * len(completions)
    for _, members in group_members.items():
        if len(members) < 2:
            continue
        ops = [parsed[i][1].operation if parsed[i][1] is not None else None for i in members]
        unique = {o for o in ops if o is not None}
        if len(unique) <= 1:
            for i in members:
                diversity_reward[i] -= CFG.diversity_penalty
        elif len(unique) >= 3:
            for i in members:
                diversity_reward[i] += CFG.diversity_bonus

    rewards: List[float] = []
    for i, (raw_state, (completion_text, action, status)) in enumerate(zip(states, parsed)):
        state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state

        if action is None:
            task_r = -1.0
        else:
            try:
                task_r, _ = score_action(
                    state,
                    action.model_dump(exclude_none=True),
                    capacity=capacity,
                )
                task_r = float(task_r)
            except Exception:
                task_r = -1.0

        fmt_r = 0.05 if action is not None else -0.1
        div_r = diversity_reward[i]
        total_r = float(task_r + fmt_r + div_r)
        rewards.append(total_r)

        prompt_text = None
        if isinstance(prompts, list) and i < len(prompts):
            prompt_text = prompts[i]

        REWARD_MONITOR.log_one(
            sample_idx=i,
            completion_text=completion_text,
            parse_status=status,
            action_obj=action,
            task_reward=task_r,
            format_reward=fmt_r + div_r,
            total_reward=total_r,
            prompt_text=prompt_text,
        )

    REWARD_MONITOR.maybe_print_summary()
    return rewards


def load_model_and_tokenizer():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to fine-tune Qwen 2.5-1.5B with 4-bit quantization.")

    print(f"[grpo] Loading tokenizer: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[grpo] Loading 4-bit base model: {CFG.model_name}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": 0},  # single GPU for notebook stability
    )
    base.config.use_cache = False

    print(f"[grpo] Attaching SFT LoRA adapter (trainable=True): {CFG.adapter_path}")
    model = PeftModel.from_pretrained(base, CFG.adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    env_dir = HERE / "server"
    backup = use_episodes(env_dir, CFG.episodes_file)
    try:
        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as e:
            raise ImportError(
                "[grpo] Missing dependency 'trl'. Run: %pip install -U trl accelerate peft"
            ) from e

        random.seed(CFG.seed)
        torch.manual_seed(CFG.seed)

        print(
            "[grpo] CONFIG  "
            f"capacity={LongHorizonMemoryEnvironment.MEMORY_CAPACITY} "
            f"episodes={CFG.episodes_file}  "
            f"num_generations={CFG.num_generations} "
            f"temperature={CFG.temperature} top_p={CFG.top_p} top_k={CFG.top_k}  "
            f"max_prompt={CFG.max_prompt_length} max_compl={CFG.max_completion_length}  "
            f"dataset_size={CFG.dataset_size}"
        )

        model, tokenizer = load_model_and_tokenizer()

        print(f"[grpo] Building dataset of {CFG.dataset_size} states from {CFG.episodes_file}")
        dataset = build_dataset(tokenizer)
        print(f"[grpo] Dataset built. Columns: {list(dataset.column_names)}")
        print(f"[grpo] Reward log file: {CFG.reward_log_file}")

        grpo_config_kwargs: Dict[str, Any] = dict(
            output_dir=CFG.output_dir,
            num_train_epochs=CFG.num_train_epochs,
            per_device_train_batch_size=CFG.per_device_train_batch_size,
            gradient_accumulation_steps=CFG.gradient_accumulation_steps,
            learning_rate=CFG.learning_rate,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            num_generations=CFG.num_generations,
            generation_batch_size=CFG.num_generations,  # ensures divisibility
            beta=CFG.beta,
            max_prompt_length=CFG.max_prompt_length,
            max_completion_length=CFG.max_completion_length,
            temperature=CFG.temperature,
            top_p=CFG.top_p,
            top_k=CFG.top_k,
            seed=CFG.seed,
            bf16=False,
            fp16=False,
            optim="paged_adamw_8bit",
            max_grad_norm=1.0,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            log_completions=True,
            num_completions_to_print=2,
        )

        valid = set(GRPOConfig.__init__.__code__.co_varnames)
        grpo_config_kwargs = {k: v for k, v in grpo_config_kwargs.items() if k in valid}
        args = GRPOConfig(**grpo_config_kwargs)

        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            reward_funcs=[combined_reward_fn],
            processing_class=tokenizer,
        )

        print("[grpo] Starting training...")
        trainer.train()
        trainer.save_model(CFG.output_dir)
        tokenizer.save_pretrained(CFG.output_dir)
        print(f"[grpo] Saved adapter and tokenizer to: {CFG.output_dir}")
    finally:
        restore_episodes(backup, env_dir)


# Notebook usage:
#   import train_grpo_memory
#   train_grpo_memory.main()
#
# Auto-run only when the file is the entry point. This guard keeps the
# module importable for tests and inspection without launching training.
if __name__ == "__main__":
    main()