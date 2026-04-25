# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Long Horizon Memory Environment (GRPO-friendly rewrite).

Compared to the original implementation, this version is engineered for
LLM-policy training with GRPO. Major changes:

- Per-step "direct credit" reward shaping. Each (action, message_relevance,
  popped_slot_relevance) combination gets an immediate, well-separated reward
  in roughly [-0.6, +0.6]. This breaks the original noop-collapse failure
  mode: an "always noop" policy is now strictly worse than a smart policy
  on every single step where the message is relevant.
- Running recall: precision/recall are computed against
  total_relevant_seen_so_far rather than the entire episode's relevant
  count. This keeps the signal informative on long-horizon episodes.
- F1 task score (smooth and bounded in [0, 1]).
- Step decay penalty removed (the original penalized correct old memories
  on long episodes and dominated the rest of the signal).
- Terminal bonus proportional to final F1 score.
- Stateless action-evaluation helper so GRPO can score a candidate action
  on a snapshot state without mutating the env.
- Configurable MEMORY_CAPACITY via the LONG_HORIZON_MEMORY_CAPACITY env var.
- Backward-compatible Action / Observation schema (add / remove / noop).
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import (
            LongHorizonMemoryAction,
            LongHorizonMemoryObservation,
        )


# ─── Reward shaping table ────────────────────────────────────────────────────
# Per-step rewards are intentionally well-separated so the policy gets a
# strong, immediate gradient on each decision. Sums per episode stay small
# enough to not dwarf the terminal bonus.
REWARD_ADD_RELEVANT = 0.6
REWARD_ADD_IRRELEVANT = -0.6
REWARD_ADD_FULL_REJECTED = -0.05  # minor: agent should evict first instead
REWARD_REMOVE_RELEVANT = -0.5
REWARD_REMOVE_IRRELEVANT = 0.4
REWARD_REMOVE_INVALID = -0.3
REWARD_NOOP_ON_RELEVANT = -0.3
REWARD_NOOP_ON_IRRELEVANT = 0.05
REWARD_INVALID_OP = -0.5
REWARD_DONE_THEN_STEP = -0.1
TERMINAL_F1_BONUS_WEIGHT = 0.5


class LongHorizonMemoryEnvironment(Environment):
    """
    Memory selection environment with shaped, GRPO-friendly rewards.

    Episode flow:
        - reset() picks a random episode (filtered by LONG_HORIZON_MEMORY_TASK).
        - step(action) consumes the *current* message and applies the action.
            * After the action, total_message_number advances by 1.
        - Episode terminates when all messages have been consumed.

    Action schema (unchanged):
        operation in {"add", "remove", "noop"}
        remove_index: int (required for "remove")
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MEMORY_CAPACITY: int = int(os.getenv("LONG_HORIZON_MEMORY_CAPACITY", "8"))

    def __init__(self) -> None:
        episodes_path = Path(__file__).with_name("episodes.json")
        with episodes_path.open("r", encoding="utf-8") as f:
            self.episodes: List[Dict[str, Any]] = json.load(f)

        self._task_name: str = (
            os.getenv("LONG_HORIZON_MEMORY_TASK", "all").strip().lower() or "all"
        )

        seed_env = os.getenv("LONG_HORIZON_MEMORY_SEED")
        self._seed: Optional[int] = (
            int(seed_env) if seed_env and seed_env.lstrip("-").isdigit() else None
        )
        self._rng = random.Random(self._seed)

        episode_id_env = os.getenv("LONG_HORIZON_MEMORY_EPISODE_ID")
        self._episode_id_override: Optional[int] = (
            int(episode_id_env)
            if episode_id_env and episode_id_env.lstrip("-").isdigit()
            else None
        )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count: int = 0

        self.episode: int = 0
        self.current_domain: str = "unknown"
        self.current_difficulty: str = "easy"
        self.messages: List[Dict[str, Any]] = []
        self.total_relevant_in_episode: int = 0
        self.total_relevant_seen: int = 0
        self.total_message_number: int = 0
        self.memory: List[Dict[str, Any]] = []
        self.last_action_error: Optional[str] = None
        self._done: bool = False
        self._cumulative_reward: float = 0.0

        self._set_random_episode()

    # ── episode selection ────────────────────────────────────────────────────
    def _infer_difficulty(self, episode_data: Dict[str, Any], episode_index: int) -> str:
        explicit = str(episode_data.get("difficulty", "")).strip().lower()
        if explicit in {"easy", "medium", "hard"}:
            return explicit
        if episode_index <= 1:
            return "easy"
        if episode_index <= 3:
            return "medium"
        return "hard"

    def _candidate_indices_for_task(self) -> List[int]:
        if self._task_name not in {"easy", "medium", "hard", "all"}:
            self._task_name = "all"
        if self._task_name == "all":
            return list(range(len(self.episodes)))
        return [
            i
            for i, episode_data in enumerate(self.episodes)
            if self._infer_difficulty(episode_data, i) == self._task_name
        ]

    def _set_random_episode(self) -> None:
        candidates = self._candidate_indices_for_task()
        if not candidates:
            candidates = list(range(len(self.episodes)))

        chosen: Optional[int] = None
        if self._episode_id_override is not None:
            for idx in candidates:
                if int(self.episodes[idx].get("episode_id", idx + 1)) == self._episode_id_override:
                    chosen = idx
                    break

        self.episode = chosen if chosen is not None else self._rng.choice(candidates)
        episode_data = self.episodes[self.episode]
        self.current_domain = episode_data.get("conversation_domain", "unknown")
        self.current_difficulty = self._infer_difficulty(episode_data, self.episode)
        self.messages = episode_data.get("string_relevant_messages", [])
        self.total_relevant_in_episode = sum(
            1 for m in self.messages if m.get("isRelevant", True)
        )

        self.total_message_number = 0
        self.total_relevant_seen = 0
        self.memory = []
        self.last_action_error = None
        self._done = len(self.messages) == 0
        self._cumulative_reward = 0.0

    # ── lightweight helpers for offline dataset construction ─────────────────
    def reset_for_sampling(self) -> None:
        """Reset without producing an Observation. Used by GRPO dataset builders."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._set_random_episode()

    def advance_oracle_for_sampling(self, n_steps: int) -> None:
        """Fast oracle warmup: deterministically apply the optimal action n_steps
        times without producing observations. Optimal action = add when relevant
        and capacity available, else noop."""
        for _ in range(n_steps):
            if self._done:
                return
            msg = self._current_message()
            if msg is None:
                self._done = True
                return
            rel = bool(msg.get("isRelevant", False))
            if rel and len(self.memory) < self.MEMORY_CAPACITY:
                self.memory.append(
                    {
                        "text": msg.get("text", ""),
                        "isRelevant": True,
                        "timestamp": self.total_message_number,
                    }
                )
            if rel:
                self.total_relevant_seen += 1
            self.total_message_number += 1
            if self.total_message_number >= len(self.messages):
                self._done = True

    # ── observation construction ─────────────────────────────────────────────
    def _current_message(self) -> Optional[Dict[str, Any]]:
        if self.total_message_number >= len(self.messages):
            return None
        return self.messages[self.total_message_number]

    def _memory_stats(self) -> Dict[str, int]:
        correct = sum(1 for m in self.memory if m.get("isRelevant", False))
        incorrect = len(self.memory) - correct
        return {"correct": correct, "incorrect": incorrect}

    def _running_metrics(self) -> Dict[str, float]:
        stats = self._memory_stats()
        kept = len(self.memory)
        seen = self.total_relevant_seen
        precision = stats["correct"] / kept if kept > 0 else 1.0
        recall = stats["correct"] / seen if seen > 0 else 1.0
        return {"precision": precision, "recall": recall}

    def _task_score(self) -> float:
        """Smooth F1 over what the agent has seen so far."""
        if self.total_relevant_seen == 0:
            return 0.0
        m = self._running_metrics()
        p, r = m["precision"], m["recall"]
        if p + r <= 0:
            return 0.0
        return max(0.0, min(1.0, 2.0 * p * r / (p + r)))

    def _observation(self, reward: float) -> LongHorizonMemoryObservation:
        current_message = self._current_message()
        new_message = "" if current_message is None else current_message.get("text", "")
        stats = self._memory_stats()
        return LongHorizonMemoryObservation(
            domain=self.current_domain,
            task_name=self.current_difficulty,
            new_message=new_message,
            memory=[m.get("text", "") for m in self.memory],
            memory_count=len(self.memory),
            reward=reward,
            done=self._done,
            metadata={
                "reset_count": self._reset_count,
                "episode_id": self.episodes[self.episode].get("episode_id", self.episode + 1),
                "task": self.current_difficulty,
                "memory_capacity": self.MEMORY_CAPACITY,
                "memory_full": len(self.memory) >= self.MEMORY_CAPACITY,
                "correct_in_memory": stats["correct"],
                "incorrect_in_memory": stats["incorrect"],
                "task_score": self._task_score(),
                "total_relevant_seen": self.total_relevant_seen,
                "total_relevant_in_episode": self.total_relevant_in_episode,
                "last_action_error": self.last_action_error,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    # ── core RL API ──────────────────────────────────────────────────────────
    def reset(self) -> LongHorizonMemoryObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._set_random_episode()
        return self._observation(reward=0.0)

    def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:  # type: ignore[override]
        self._state.step_count += 1
        self.last_action_error = None

        if self._done:
            self.last_action_error = "episode_already_done"
            return self._observation(reward=REWARD_DONE_THEN_STEP)

        # Capture pre-action snapshot we need for credit assignment.
        msg = self._current_message()
        msg_was_relevant = bool(msg.get("isRelevant", False)) if msg is not None else False
        op = action.operation

        # Apply the action and capture the popped slot when applicable.
        popped_was_relevant: Optional[bool] = None

        if op == "add":
            if msg is None:
                self.last_action_error = "no_current_message"
            elif len(self.memory) >= self.MEMORY_CAPACITY:
                self.last_action_error = "memory_capacity_reached"
            else:
                self.memory.append(
                    {
                        "text": msg.get("text", ""),
                        "isRelevant": msg_was_relevant,
                        "timestamp": self.total_message_number,
                    }
                )
        elif op == "remove":
            idx = action.remove_index
            if idx is None:
                self.last_action_error = "remove_index_required"
            elif idx < 0 or idx >= len(self.memory):
                self.last_action_error = "remove_index_out_of_range"
            else:
                popped = self.memory.pop(idx)
                popped_was_relevant = bool(popped.get("isRelevant", False))
        elif op == "noop":
            pass
        else:
            self.last_action_error = "invalid_operation"

        # Advance the cursor and update running counters.
        if msg_was_relevant:
            self.total_relevant_seen += 1
        self.total_message_number += 1
        if self.total_message_number >= len(self.messages):
            self._done = True

        # Per-step shaped reward + terminal bonus.
        reward = self._shaped_step_reward(
            op=op,
            msg_was_relevant=msg_was_relevant,
            popped_was_relevant=popped_was_relevant,
            error=self.last_action_error,
        )
        if self._done:
            reward += TERMINAL_F1_BONUS_WEIGHT * self._task_score()

        # Bound to a comfortable range; the per-step table already keeps it
        # well-behaved, but clipping protects against future tweaks.
        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward
        return self._observation(reward=reward)

    def close(self) -> None:
        return None

    @property
    def state(self) -> State:
        return self._state

    # ── stateless evaluation helpers (for GRPO) ──────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copyable snapshot of all per-episode state."""
        return {
            "episode": self.episode,
            "current_domain": self.current_domain,
            "current_difficulty": self.current_difficulty,
            "messages": self.messages,
            "total_relevant_in_episode": self.total_relevant_in_episode,
            "total_relevant_seen": self.total_relevant_seen,
            "total_message_number": self.total_message_number,
            "memory": copy.deepcopy(self.memory),
            "last_action_error": self.last_action_error,
            "_done": self._done,
            "_cumulative_reward": self._cumulative_reward,
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        """Restore per-episode state from a snapshot taken with snapshot()."""
        self.episode = snap["episode"]
        self.current_domain = snap["current_domain"]
        self.current_difficulty = snap["current_difficulty"]
        self.messages = snap["messages"]
        self.total_relevant_in_episode = snap["total_relevant_in_episode"]
        self.total_relevant_seen = snap["total_relevant_seen"]
        self.total_message_number = snap["total_message_number"]
        self.memory = copy.deepcopy(snap["memory"])
        self.last_action_error = snap["last_action_error"]
        self._done = snap["_done"]
        self._cumulative_reward = snap["_cumulative_reward"]

    def evaluate_action(
        self, action: LongHorizonMemoryAction, *, lookahead: int = 0, gamma: float = 0.95
    ) -> float:
        """Score an action on the current state without permanently mutating it.

        If ``lookahead > 0``, additionally roll out the oracle policy for
        ``lookahead`` steps after the action and accumulate discounted reward.
        This is what GRPO's reward function calls per candidate completion.
        """
        snap = self.snapshot()
        try:
            obs = self.step(action)
            total = float(obs.reward)
            if lookahead > 0:
                discount = gamma
                for _ in range(lookahead):
                    if self._done:
                        break
                    next_obs = self.step(self._oracle_action())
                    total += discount * float(next_obs.reward)
                    discount *= gamma
            return total
        finally:
            self.restore(snap)

    def _oracle_action(self) -> LongHorizonMemoryAction:
        """Return the policy-optimal action for the current state."""
        msg = self._current_message()
        if msg is None:
            return LongHorizonMemoryAction(operation="noop")
        rel = bool(msg.get("isRelevant", False))
        if rel:
            if len(self.memory) < self.MEMORY_CAPACITY:
                return LongHorizonMemoryAction(operation="add")
            for i, slot in enumerate(self.memory):
                if not slot.get("isRelevant", False):
                    return LongHorizonMemoryAction(operation="remove", remove_index=i)
            return LongHorizonMemoryAction(operation="noop")
        for i, slot in enumerate(self.memory):
            if not slot.get("isRelevant", False):
                return LongHorizonMemoryAction(operation="remove", remove_index=i)
        return LongHorizonMemoryAction(operation="noop")

    # ── reward shaping ───────────────────────────────────────────────────────
    @staticmethod
    def _shaped_step_reward(
        *,
        op: str,
        msg_was_relevant: bool,
        popped_was_relevant: Optional[bool],
        error: Optional[str],
    ) -> float:
        if error is not None:
            if error == "memory_capacity_reached":
                return REWARD_ADD_FULL_REJECTED
            if error == "no_current_message":
                return -0.05
            if error == "remove_index_required":
                return REWARD_REMOVE_INVALID
            if error == "remove_index_out_of_range":
                return REWARD_REMOVE_INVALID
            if error == "invalid_operation":
                return REWARD_INVALID_OP
            if error == "episode_already_done":
                return REWARD_DONE_THEN_STEP
            return -0.1

        if op == "add":
            return REWARD_ADD_RELEVANT if msg_was_relevant else REWARD_ADD_IRRELEVANT
        if op == "remove":
            return REWARD_REMOVE_RELEVANT if popped_was_relevant else REWARD_REMOVE_IRRELEVANT
        if op == "noop":
            return REWARD_NOOP_ON_RELEVANT if msg_was_relevant else REWARD_NOOP_ON_IRRELEVANT
        return REWARD_INVALID_OP


# ─── Stateless reward helper (GRPO-friendly) ─────────────────────────────────
def score_action(
    state: Dict[str, Any],
    action: Dict[str, Any],
    *,
    capacity: int = LongHorizonMemoryEnvironment.MEMORY_CAPACITY,
    include_terminal_bonus: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Pure-function step scorer used by GRPO reward functions.

    Parameters
    ----------
    state : dict with the keys
        memory : list of {"text": str, "isRelevant": bool}
        message : {"text": str, "isRelevant": bool} | None
        total_relevant_seen : int
        is_last_step : bool   (whether the terminal F1 bonus should fire)
    action : dict with the keys
        operation : "add" | "remove" | "noop"
        remove_index : int | None
    capacity : maximum memory size
    include_terminal_bonus : whether to add the F1 bonus when is_last_step

    Returns
    -------
    reward : float clipped to [-1, 1]
    new_state : dict with updated memory, total_relevant_seen, error
    """
    op = str(action.get("operation", "noop"))
    msg = state.get("message")
    msg_rel = bool(msg.get("isRelevant", False)) if isinstance(msg, dict) else False
    memory = [dict(m) for m in state.get("memory", [])]
    total_relevant_seen = int(state.get("total_relevant_seen", 0))
    is_last_step = bool(state.get("is_last_step", False))

    error: Optional[str] = None
    popped_rel: Optional[bool] = None

    if op == "add":
        if msg is None:
            error = "no_current_message"
        elif len(memory) >= capacity:
            error = "memory_capacity_reached"
        else:
            memory.append({"text": msg.get("text", ""), "isRelevant": msg_rel})
    elif op == "remove":
        idx = action.get("remove_index")
        if idx is None:
            error = "remove_index_required"
        else:
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                error = "remove_index_required"
                idx = -1
            if error is None and (idx < 0 or idx >= len(memory)):
                error = "remove_index_out_of_range"
            elif error is None:
                popped = memory.pop(idx)
                popped_rel = bool(popped.get("isRelevant", False))
    elif op == "noop":
        pass
    else:
        error = "invalid_operation"

    reward = LongHorizonMemoryEnvironment._shaped_step_reward(
        op=op,
        msg_was_relevant=msg_rel,
        popped_was_relevant=popped_rel,
        error=error,
    )

    new_relevant_seen = total_relevant_seen + (1 if msg_rel else 0)

    if include_terminal_bonus and is_last_step:
        correct = sum(1 for m in memory if m.get("isRelevant", False))
        kept = len(memory)
        precision = correct / kept if kept > 0 else 1.0
        recall = correct / new_relevant_seen if new_relevant_seen > 0 else 1.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        reward += TERMINAL_F1_BONUS_WEIGHT * f1

    reward = max(-1.0, min(1.0, reward))
    return reward, {
        "memory": memory,
        "total_relevant_seen": new_relevant_seen,
        "error": error,
    }


if __name__ == "__main__":
    pass
