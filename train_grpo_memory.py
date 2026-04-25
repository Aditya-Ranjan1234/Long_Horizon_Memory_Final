"""
GRPO trainer for the Long Horizon Memory action policy.

Loads a Qwen 2.5-1.5B-Instruct base in 4-bit, attaches the SFT LoRA adapter
from ``./memory_action_sft_qwen15b`` (produced by ``train_sft_qwen.py``), and
fine-tunes those LoRA weights with GRPO using rewards computed by
``score_action`` from the re-engineered Long Horizon Memory environment.

Why this script converges where the original env did not:
- Per-step shaped reward gives every (state, action) pair an immediately
  attributable score in ``[-1, +1]`` (see ``score_action``).
- Reward is positive only for the right decision (``add`` for relevant,
  ``noop`` / ``remove`` of irrelevant for the rest), so degenerate policies
  like always-noop or always-add are strictly punished.
- A small ``format_reward_fn`` rewards parseable JSON so the model is never
  pushed away from the SFT-learned schema during exploration.
- We continue from the SFT adapter, so the policy starts with a high
  parseability rate (verify with ``verify_sft_model.py``).

Usage:
    cd Long_Horizon_Memory
    python train_grpo_memory.py

Environment overrides:
    MODEL_NAME, ADAPTER_PATH, EPISODES_FILE, OUTPUT_DIR
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
    LEARNING_RATE, NUM_GENERATIONS, BETA, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH
    DATASET_SIZE, ROLLOUTS_PER_EPISODE, EVAL_EVERY
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "server"))

from data import SYSTEM_PROMPT, format_observation  # noqa: E402
from models import LongHorizonMemoryAction  # noqa: E402
from long_horizon_memory_environment import (  # noqa: E402
    LongHorizonMemoryEnvironment,
    score_action,
)


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path: str = os.getenv("ADAPTER_PATH", str(HERE / "memory_action_sft_qwen15b"))
    output_dir: str = os.getenv("OUTPUT_DIR", str(HERE / "memory_action_grpo_qwen15b"))
    episodes_file: str = os.getenv("EPISODES_FILE", "episodes_grpo.json")

    num_train_epochs: int = int(os.getenv("NUM_TRAIN_EPOCHS", "2"))
    per_device_train_batch_size: int = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "5e-6"))
    num_generations: int = int(os.getenv("NUM_GENERATIONS", "6"))
    beta: float = float(os.getenv("BETA", "0.04"))
    max_prompt_length: int = int(os.getenv("MAX_PROMPT_LENGTH", "640"))
    max_completion_length: int = int(os.getenv("MAX_COMPLETION_LENGTH", "32"))

    # Dataset construction.
    dataset_size: int = int(os.getenv("DATASET_SIZE", "768"))
    rollouts_per_episode: int = int(os.getenv("ROLLOUTS_PER_EPISODE", "3"))
    seed: int = int(os.getenv("GRPO_SEED", "20260426"))

    eval_every: int = int(os.getenv("EVAL_EVERY", "0"))


CFG = TrainConfig()


# ─── Episodes file routing ───────────────────────────────────────────────────
def use_episodes(env_dir: Path, episodes_filename: str) -> Optional[Path]:
    """Swap server/episodes.json with the requested file, returning the path
    to the backup so we can restore on exit."""
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
    if backup is None or not backup.exists():
        return
    dst = env_dir / "episodes.json"
    dst.write_bytes(backup.read_bytes())
    backup.unlink()


# ─── Prompt construction ─────────────────────────────────────────────────────
def to_obs_dict(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    return {
        "domain": env.current_domain,
        "task_name": env.current_difficulty,
        "memory": [
            {"index": i, "text": m.get("text", "")} for i, m in enumerate(env.memory)
        ],
        "new_message": (env._current_message() or {}).get("text", ""),
    }


def to_score_state(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    msg = env._current_message()
    is_last = (env.total_message_number == len(env.messages) - 1)
    return {
        "memory": [
            {"text": m.get("text", ""), "isRelevant": bool(m.get("isRelevant", False))}
            for m in env.memory
        ],
        "message": (
            None
            if msg is None
            else {"text": msg.get("text", ""), "isRelevant": bool(msg.get("isRelevant", False))}
        ),
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
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{body}\n"
            "<|assistant|>\n"
        )


# ─── Dataset construction ────────────────────────────────────────────────────
def build_dataset(tokenizer) -> Dataset:
    """Sample diverse (state, prompt) pairs by running a noisy mixed policy.

    For each step state we keep the user prompt body (used by GRPO as the
    completion target's input) plus a JSON-serialized state used by the
    reward function.
    """
    rng = random.Random(CFG.seed)
    env = LongHorizonMemoryEnvironment()
    rows: List[Dict[str, Any]] = []

    while len(rows) < CFG.dataset_size:
        env.reset()
        steps_in_ep = 0
        while not env._done and len(rows) < CFG.dataset_size:
            obs = to_obs_dict(env)
            score_state = to_score_state(env)
            prompt = build_chat_prompt(obs, tokenizer)
            rows.append(
                {
                    "prompt": prompt,
                    "state_json": json.dumps(score_state, ensure_ascii=False),
                }
            )

            # Mix policies during data collection so GRPO sees both
            # easy-and-hard memory configurations.
            roll = rng.random()
            if roll < 0.5:
                action = env._oracle_action()
            elif roll < 0.8:
                action = LongHorizonMemoryAction(
                    operation=rng.choice(["add", "noop", "noop"])
                )
            else:
                action = (
                    LongHorizonMemoryAction(
                        operation="remove",
                        remove_index=rng.randrange(len(env.memory)),
                    )
                    if env.memory
                    else LongHorizonMemoryAction(operation="noop")
                )
            env.step(action)
            steps_in_ep += 1
            if steps_in_ep >= CFG.rollouts_per_episode * 8:
                break

    rng.shuffle(rows)
    return Dataset.from_list(rows[: CFG.dataset_size])


# ─── Reward functions ────────────────────────────────────────────────────────
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_first_json(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _completion_text(c: Any) -> str:
    """GRPOTrainer hands us either strings or chat-style lists."""
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


def task_reward_fn(completions, **kwargs) -> List[float]:
    """Score each completion via the env's stateless reward.

    GRPOTrainer passes the dataset's ``state_json`` column through ``kwargs``.
    """
    states = kwargs.get("state_json")
    if states is None:
        return [0.0] * len(completions)
    rewards: List[float] = []
    capacity = LongHorizonMemoryEnvironment.MEMORY_CAPACITY
    for comp, raw_state in zip(completions, states):
        state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        action, status = _parse_action(_completion_text(comp))
        if action is None:
            rewards.append(-1.0)
            continue
        try:
            r, _ = score_action(state, action.model_dump(exclude_none=True), capacity=capacity)
        except Exception:
            r = -1.0
        rewards.append(float(r))
    return rewards


def format_reward_fn(completions, **kwargs) -> List[float]:
    """Small bonus for parseable JSON so the model never drifts off-schema."""
    out: List[float] = []
    for comp in completions:
        action, status = _parse_action(_completion_text(comp))
        if action is None:
            out.append(-0.1)
        else:
            out.append(0.05)
    return out


# ─── Model loading ───────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to fine-tune Qwen 2.5-1.5B with 4-bit quantization.")
    print(f"[grpo] Loading tokenizer: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required by GRPO generation

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
        device_map="auto",
    )
    base.config.use_cache = False

    print(f"[grpo] Attaching SFT LoRA adapter (trainable=True): {CFG.adapter_path}")
    model = PeftModel.from_pretrained(base, CFG.adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


# ─── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    env_dir = HERE / "server"
    backup = use_episodes(env_dir, CFG.episodes_file)
    try:
        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as e:
            raise SystemExit(
                "[grpo] trl is required. Install with `pip install -U trl accelerate peft`"
            ) from e

        random.seed(CFG.seed)
        torch.manual_seed(CFG.seed)

        model, tokenizer = load_model_and_tokenizer()

        print(f"[grpo] Building dataset of {CFG.dataset_size} states from {CFG.episodes_file}")
        dataset = build_dataset(tokenizer)
        print(f"[grpo] Dataset built. Columns: {list(dataset.column_names)}")

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
            beta=CFG.beta,
            max_prompt_length=CFG.max_prompt_length,
            max_completion_length=CFG.max_completion_length,
            temperature=1.0,
            seed=CFG.seed,
            bf16=False,
            fp16=False,
            optim="paged_adamw_8bit",
            max_grad_norm=1.0,
            remove_unused_columns=False,
        )
        # Older versions of trl don't expose every kwarg above; drop unknown
        # ones gracefully so this script keeps running across trl versions.
        valid = set(GRPOConfig.__init__.__code__.co_varnames)
        grpo_config_kwargs = {k: v for k, v in grpo_config_kwargs.items() if k in valid}
        args = GRPOConfig(**grpo_config_kwargs)

        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            reward_funcs=[task_reward_fn, format_reward_fn],
            processing_class=tokenizer,
        )

        print("[grpo] Starting training...")
        trainer.train()
        trainer.save_model(CFG.output_dir)
        tokenizer.save_pretrained(CFG.output_dir)
        print(f"[grpo] Saved adapter and tokenizer to: {CFG.output_dir}")
    finally:
        restore_episodes(backup, env_dir)


if __name__ == "__main__":
    main()
