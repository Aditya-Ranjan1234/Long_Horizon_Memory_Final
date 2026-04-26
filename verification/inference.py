"""Competition-compliant inference runner for long_horizon_memory."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Ensure we can import from root
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    from server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from .models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from .server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from long_horizon_memory.server.long_horizon_memory_environment import LongHorizonMemoryEnvironment

import requests

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://aditya-ranjan1234-long-horizon-memory-env-final.hf.space")

BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "long_horizon_memory")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))
TASKS = ["easy", "medium", "hard"]
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", "2"))
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "1337"))

SYSTEM_PROMPT = """You are an expert memory management system for long-horizon tasks. Your goal is to maintain a high-quality memory buffer by keeping relevant information and discarding noise.

SCORING FORMULA (aim for ≥0.7 for success):
task_score = 0.6×recall + 0.4×precision - 0.25×incorrect_rate - 0.15×overflow_rate

KEY INSIGHTS:
- Recall (60% weight) is MORE important than precision (40%)
- Memory capacity: 8 slots maximum
- Prioritize capturing ALL relevant info, but avoid keeping irrelevant distractors
- Better to keep relevant items even if some noise, than miss important info

RELEVANCE INDICATORS (keep these):
- Technical problems, bugs, requirements, constraints
- User goals, pain points, specific needs
- System design decisions, architecture details
- Performance issues, metrics, monitoring needs
- Domain-specific technical details

IRRELEVANCE INDICATORS (discard these):
- Personal hobbies (hobbies, sports, cooking, photography, music)
- Shopping and purchases (buying items, products)
- Lifestyle and daily activities (weekend plans, food, exercise)
- Entertainment (movies, shows, games, books unrelated to task)
- Random observations with no technical connection

STRATEGY:
1. ADD: If message contains technical content relevant to the domain
2. REMOVE: If memory contains noise AND you need space for better content
3. NOOP: If current message is noise and memory is already optimal

CRITICAL: You must output ONLY valid JSON with this exact format:
{"operation": "add"}
OR
{"operation": "remove", "remove_index": 0}
OR
{"operation": "noop"}

Think carefully about each decision based on the scoring formula."""


def check_hf_status() -> None:
    print(f"[CHECK] Verifying Hugging Face Space: {HF_SPACE_URL}", flush=True)
    try:
        resp = requests.get(f"{HF_SPACE_URL}/health", timeout=10)
        if resp.status_code == 200:
            print(f"[OK] Hugging Face Space is LIVE and healthy.", flush=True)
        else:
            print(f"[WARNING] Hugging Face Space returned status {resp.status_code}", flush=True)
    except Exception as e:
        print(f"[ERROR] Could not connect to Hugging Face Space: {e}", flush=True)


def log_start(task: str, env: str, model: str, metadata: Optional[dict] = None) -> None:
    header = f"[START] task={task} env={env} model={model}"
    if metadata:
        params = {k: v for k, v in metadata.items() if k in ["memory_capacity", "task", "episode_id"]}
        header += f" | parameters={params}"
    print(header, flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    metadata: Optional[dict] = None
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"

    # Enhanced logging with metrics
    base_log = f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"

    if metadata:
        score = metadata.get("task_score", 0.0)
        correct = metadata.get("correct_in_memory", 0)
        incorrect = metadata.get("incorrect_in_memory", 0)
        memory_ages = metadata.get("memory_ages", [])
        avg_age = metadata.get("avg_memory_age", 0)

        base_log += f" | score={score:.3f} correct={correct} incorrect={incorrect}"

        if memory_ages:
            base_log += f" mem_ages={memory_ages} avg_age={avg_age:.1f}"

    print(base_log, flush=True)


def log_end(success: bool, steps: int, rewards: List[float], final_score: Optional[float] = None) -> None:
    success_val = str(success).lower()
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)

    log_msg = f"[END] success={success_val} steps={steps}"
    if final_score is not None:
        log_msg += f" final_score={final_score:.3f}"
    log_msg += f" avg_reward={avg_reward:.3f} rewards={rewards_text}"

    print(log_msg, flush=True)


def _heuristic_action(observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    """Improved fallback policy with smarter pattern matching."""
    text = observation.new_message.lower()

    # Expanded irrelevance patterns
    irrelevance_keywords = [
        "hobby", "hobbies", "weekend", "bought", "buy", "purchase", "shopping",
        "coffee", "keyboard", "laptop", "gaming", "game", "movie", "show",
        "cooking", "recipe", "food", "restaurant", "pizza", "lunch", "dinner",
        "sport", "football", "basketball", "running", "jogging", "gym",
        "paint", "painting", "music", "guitar", "piano", "art",
        "vacation", "trip", "travel", "hiking", "concert", "theater",
        "photography", "camera", "book club", "reading", "novel",
        "dog", "cat", "pet", "apartment", "house", "renovating",
        "meditation", "yoga", "sleep", "tire", "bicycle", "neighbor"
    ]

    # Technical relevance patterns
    relevance_keywords = [
        "bug", "error", "issue", "problem", "fail", "crash", "slow",
        "performance", "memory", "cpu", "api", "database", "server",
        "code", "script", "function", "class", "implement", "design",
        "architecture", "system", "pipeline", "process", "monitoring",
        "test", "debug", "optimize", "scale", "deploy", "build",
        "requirement", "feature", "user", "client", "data", "model",
        "algorithm", "metric", "score", "training", "prediction"
    ]

    # Check if message is irrelevant noise
    is_irrelevant = any(kw in text for kw in irrelevance_keywords)
    is_relevant = any(kw in text for kw in relevance_keywords)

    # Decision logic
    if is_irrelevant and not is_relevant:
        # Message is noise, don't add it
        return LongHorizonMemoryAction(operation="noop")

    if is_relevant:
        # Message is relevant
        if observation.memory_count < 8:
            return LongHorizonMemoryAction(operation="add")
        else:
            # Memory full, remove oldest to make room
            return LongHorizonMemoryAction(operation="remove", remove_index=0)

    # Neutral message - conservative approach
    if observation.memory_count < 6:  # Still have room
        return LongHorizonMemoryAction(operation="add")

    return LongHorizonMemoryAction(operation="noop")


def _parse_action(content: str, observation: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`")
        normalized = normalized.replace("json", "", 1).strip()

    try:
        payload = json.loads(normalized)
        op = payload.get("operation", "noop")
        if op == "remove":
            idx = payload.get("remove_index")
            if isinstance(idx, int):
                return LongHorizonMemoryAction(operation="remove", remove_index=idx)
            return LongHorizonMemoryAction(operation="noop")
        if op in {"add", "noop"}:
            return LongHorizonMemoryAction(operation=op)
    except Exception:
        pass
    return _heuristic_action(observation)


def choose_action(
    llm: OpenAI,
    observation: LongHorizonMemoryObservation,
    task_name: str,
) -> LongHorizonMemoryAction:
    # Build enhanced context-aware prompt
    memory_summary = (
        f"Currently storing {observation.memory_count}/8 items in memory.\n"
        f"Current memory contents:\n"
        + "\n".join(f"  [{i}] {msg}" for i, msg in enumerate(observation.memory))
        if observation.memory
        else "Memory is empty (0/8 items)."
    )

    metadata = observation.metadata
    current_score = metadata.get("task_score", 0.0)
    correct_count = metadata.get("correct_in_memory", 0)
    incorrect_count = metadata.get("incorrect_in_memory", 0)

    user_prompt = f"""TASK DIFFICULTY: {task_name}
DOMAIN: {observation.domain}

CURRENT STATE:
{memory_summary}

PERFORMANCE METRICS:
- Current task score: {current_score:.2f} (need ≥0.70 for success)
- Correct items in memory: {correct_count}
- Incorrect items in memory: {incorrect_count}
- Recall: {correct_count}/{metadata.get('memory_capacity', 8)} slots used

NEW INCOMING MESSAGE:
"{observation.new_message}"

DECISION REQUIRED:
Analyze if the new message is relevant to the {observation.domain} domain task.
Consider current memory state and whether you need to make room or keep current optimal state.
Output your decision as JSON only."""

    last_error: Optional[Exception] = None
    for attempt in range(MAX_MODEL_RETRIES + 1):
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Slightly higher for better reasoning
                max_tokens=150,   # More tokens for reasoning
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = completion.choices[0].message.content or "{}"
            print(f"[LLM] Using {MODEL_NAME} (attempt {attempt + 1})", flush=True)
            return _parse_action(content.strip(), observation)
        except Exception as exc:
            last_error = exc
            print(f"[LLM] API call failed (attempt {attempt + 1}): {str(exc)[:100]}", flush=True)

    print(f"[LLM] All retries exhausted, using heuristic fallback", flush=True)
    return _heuristic_action(observation)


def action_to_text(action: LongHorizonMemoryAction) -> str:
    if action.operation == "remove":
        return f"remove:{action.remove_index}"
    return action.operation


def run_task(task_name: str, llm: OpenAI) -> Tuple[bool, List[float]]:
    task_seed = BASELINE_SEED + TASKS.index(task_name)
    os.environ["LONG_HORIZON_MEMORY_SEED"] = str(task_seed)
    os.environ["LONG_HORIZON_MEMORY_TASK"] = task_name
    env = LongHorizonMemoryEnvironment()

    observation = env.reset()
    log_start(task_name, BENCHMARK, MODEL_NAME, observation.metadata)

    rewards: List[float] = []
    success = False
    step_count = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            step_count = step
            action = choose_action(llm, observation, task_name)
            observation = env.step(action)

            reward = float(observation.reward)
            done = bool(observation.done)
            error = observation.metadata.get("last_action_error")

            rewards.append(reward)
            log_step(step, action_to_text(action), reward, done, error, observation.metadata)

            if done:
                score = float(observation.metadata.get("task_score", 0.0))
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if not bool(observation.done):
            score = float(observation.metadata.get("task_score", 0.0))
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        log_step(step_count + 1, "noop", 0.0, True, str(exc))
        success = False
    finally:
        final_score = float(observation.metadata.get("task_score", 0.0)) if observation else 0.0
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success, len(rewards), rewards, final_score)

    return success, rewards


def main() -> None:
    check_hf_status()

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN must be set for inference.")

    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Check if we should run all episodes
    run_all_episodes = os.getenv("RUN_ALL_EPISODES", "false").lower() == "true"

    if run_all_episodes:
        # Run each specific episode ID (will be 1-24 after we add new episodes)
        # For now start with 1-9, will expand to 24
        print(f"[INFO] Running all episodes individually", flush=True)

        total_success = 0
        all_scores = []

        # Determine max episode ID by checking episodes.json
        try:
            with open("server/episodes.json", "r", encoding="utf-8") as f:
                episodes_data = json.load(f)
                max_episode_id = max(ep["episode_id"] for ep in episodes_data)
        except Exception:
            max_episode_id = 9  # Default to 9 if can't read

        for ep_id in range(1, max_episode_id + 1):
            os.environ["LONG_HORIZON_MEMORY_EPISODE_ID"] = str(ep_id)
            os.environ["LONG_HORIZON_MEMORY_TASK"] = "all"

            # Load episode to get difficulty and domain
            try:
                with open("server/episodes.json", "r", encoding="utf-8") as f:
                    episodes_data = json.load(f)
                    episode = next((ep for ep in episodes_data if ep["episode_id"] == ep_id), None)
                    if episode:
                        difficulty = episode.get("difficulty", "unknown")
                        domain = episode.get("conversation_domain", "unknown")
                        print(f"\n[EPISODE] id={ep_id} difficulty={difficulty} domain={domain}", flush=True)
            except Exception:
                pass

            env = LongHorizonMemoryEnvironment()
            observation = env.reset()
            log_start("all", BENCHMARK, MODEL_NAME, observation.metadata)

            rewards: List[float] = []
            success = False

            try:
                for step in range(1, MAX_STEPS + 1):
                    action = choose_action(llm, observation, "all")
                    observation = env.step(action)

                    reward = float(observation.reward)
                    done = bool(observation.done)
                    error = observation.metadata.get("last_action_error")

                    rewards.append(reward)
                    log_step(step, action_to_text(action), reward, done, error, observation.metadata)

                    if done:
                        score = float(observation.metadata.get("task_score", 0.0))
                        success = score >= SUCCESS_SCORE_THRESHOLD
                        all_scores.append(score)
                        break

                if not bool(observation.done):
                    score = float(observation.metadata.get("task_score", 0.0))
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    all_scores.append(score)
            except Exception as exc:
                log_step(len(rewards) + 1, "noop", 0.0, True, str(exc))
                success = False
                all_scores.append(0.0)
            finally:
                final_score = float(observation.metadata.get("task_score", 0.0)) if observation else 0.0
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
                log_end(success, len(rewards), rewards, final_score)

            if success:
                total_success += 1

        # Print summary
        success_rate = (total_success / max_episode_id) * 100 if max_episode_id > 0 else 0
        avg_final_reward = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\n[SUMMARY] Total: {max_episode_id} episodes | Success: {total_success} ({success_rate:.1f}%) | Avg Final Reward: {avg_final_reward:.3f}", flush=True)
    else:
        # Original behavior: run easy, medium, hard tasks
        for task in TASKS:
            run_task(task, llm)


if __name__ == "__main__":
    main()
