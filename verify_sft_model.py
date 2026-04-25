"""
Smoke test for the SFT-trained Long Horizon Memory action model.

Loads the LoRA adapter from ./memory_action_sft_qwen15b on top of
Qwen/Qwen2.5-1.5B-Instruct, samples N states from the GRPO-friendly
environment dataset, generates one completion per state, and reports:

- JSON parse rate
- Pydantic validation rate (LongHorizonMemoryAction)
- per-action distribution
- mean per-step reward computed by score_action

Run:
    cd Long_Horizon_Memory
    python verify_sft_model.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
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

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", str(HERE / "memory_action_sft_qwen15b"))
EPISODES_FILE = os.getenv("EPISODES_FILE", "episodes_grpo.json")
N_SAMPLES = int(os.getenv("N_SAMPLES", "32"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "32"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[str]:
    """Return the first JSON object substring inside text, or None."""
    text = text.strip()
    if text.startswith("```"):
        # strip a leading code fence like ```json ... ```
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def parse_completion_to_action(text: str) -> Tuple[Optional[LongHorizonMemoryAction], str]:
    """Try to parse a model completion into a LongHorizonMemoryAction.

    Returns (action, status) where status is one of:
        'ok' | 'no_json' | 'json_invalid' | 'pydantic_invalid'
    """
    obj_str = extract_json_object(text)
    if obj_str is None:
        return None, "no_json"
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError:
        return None, "json_invalid"
    try:
        action = LongHorizonMemoryAction.model_validate(obj)
    except Exception:
        return None, "pydantic_invalid"
    return action, "ok"


def use_grpo_episodes(env_dir: Path, episodes_filename: str) -> Optional[Path]:
    """Temporarily point the env at episodes_grpo.json by symlink/copy.

    The env class always reads from ``server/episodes.json``. We swap that
    file with the requested episodes for the duration of the script and
    restore it on exit.
    """
    src = env_dir / episodes_filename
    if not src.exists():
        return None
    dst = env_dir / "episodes.json"
    backup = env_dir / ".episodes_backup_for_verify.json"
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


def build_state_for_score(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    """Return a state dict consumable by score_action() for the env's
    *current* step (before applying any action)."""
    msg = env._current_message()
    is_last = (env.total_message_number == len(env.messages) - 1)
    return {
        "memory": [{"text": m.get("text", ""), "isRelevant": bool(m.get("isRelevant", False))} for m in env.memory],
        "message": (
            None
            if msg is None
            else {"text": msg.get("text", ""), "isRelevant": bool(msg.get("isRelevant", False))}
        ),
        "total_relevant_seen": env.total_relevant_seen,
        "is_last_step": is_last,
    }


def build_prompt(observation_dict: Dict[str, Any], tokenizer) -> str:
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


def to_obs_dict(env: LongHorizonMemoryEnvironment) -> Dict[str, Any]:
    """Build the observation dict that format_observation expects."""
    return {
        "domain": env.current_domain,
        "task_name": env.current_difficulty,
        "memory": [
            {"index": i, "text": m.get("text", "")} for i, m in enumerate(env.memory)
        ],
        "new_message": (env._current_message() or {}).get("text", ""),
    }


def collect_states(n: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Sample n distinct (observation_dict, score_state_dict) pairs by
    rolling out a noisy oracle policy through random episodes."""
    import random

    env = LongHorizonMemoryEnvironment()
    rng = random.Random(2026)

    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    while len(out) < n:
        env.reset()
        while not env._done and len(out) < n:
            out.append((to_obs_dict(env), build_state_for_score(env)))
            # Step using a noisy-oracle to keep states diverse.
            if rng.random() < 0.7:
                env.step(env._oracle_action())
            else:
                env.step(LongHorizonMemoryAction(operation=rng.choice(["add", "noop"])))
    return out


def main() -> None:
    env_dir = HERE / "server"
    backup = use_grpo_episodes(env_dir, EPISODES_FILE)
    try:
        if not torch.cuda.is_available():
            raise SystemExit("[verify] CUDA required to load the 4-bit base model.")

        print(f"[verify] Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[verify] Loading 4-bit base model: {MODEL_NAME}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
        )

        print(f"[verify] Attaching LoRA adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()

        print(f"[verify] Sampling {N_SAMPLES} states from {EPISODES_FILE}")
        rows = collect_states(N_SAMPLES)

        statuses: Counter = Counter()
        actions: Counter = Counter()
        rewards: List[float] = []
        examples: List[Tuple[str, str, float]] = []

        for i, (obs_dict, score_state) in enumerate(rows):
            prompt = build_prompt(obs_dict, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=TEMPERATURE > 0.0,
                    temperature=max(TEMPERATURE, 1e-5),
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(
                out_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            action, status = parse_completion_to_action(gen)
            statuses[status] += 1

            if action is not None:
                actions[action.operation] += 1
                r, _ = score_action(
                    score_state,
                    action.model_dump(exclude_none=True),
                    capacity=LongHorizonMemoryEnvironment.MEMORY_CAPACITY,
                )
                rewards.append(r)
            else:
                rewards.append(-1.0)

            if len(examples) < 5:
                examples.append((obs_dict["new_message"][:80], gen.strip()[:80], rewards[-1]))

        n = len(rows)
        ok = statuses["ok"]
        print()
        print("=" * 72)
        print(f"  total samples       : {n}")
        print(f"  pydantic-valid     : {ok}/{n}  ({100.0 * ok / n:.1f}%)")
        for s in ("no_json", "json_invalid", "pydantic_invalid"):
            if statuses[s]:
                print(f"  failed ({s:>16}): {statuses[s]}")
        print(f"  action distribution : {dict(actions)}")
        if rewards:
            print(
                f"  mean step reward    : {sum(rewards)/len(rewards):+.3f}   "
                f"(min={min(rewards):+.2f}, max={max(rewards):+.2f})"
            )
        print()
        print("Sample completions (first 5):")
        for i, (msg, gen, rew) in enumerate(examples):
            print(f"  [{i}] msg : {msg!r}")
            print(f"      gen : {gen!r}")
            print(f"      r   : {rew:+.3f}")
        print("=" * 72)

        if ok / max(1, n) < 0.7:
            print(
                "[verify] WARNING: parse rate < 70%. The SFT adapter may not have "
                "converged well. Consider re-running SFT or lowering the GRPO "
                "JSON-format reward threshold."
            )
        else:
            print("[verify] OK: SFT model produces parseable LongHorizonMemoryAction.")
    finally:
        restore_episodes(backup, env_dir)


if __name__ == "__main__":
    main()
