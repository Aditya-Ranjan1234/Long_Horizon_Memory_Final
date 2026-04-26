"""Random baseline agent for Long Horizon Memory."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

try:
    from .models import LongHorizonMemoryAction
    from .server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
except ImportError:
    from models import LongHorizonMemoryAction
    from server.long_horizon_memory_environment import LongHorizonMemoryEnvironment


@dataclass
class EpisodeResult:
    """Summary for a single random-baseline episode."""

    episode_index: int
    steps: int
    reward: float


class RandomBaselineAgent:
    """Random policy over add/remove/noop actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, memory_count: int) -> LongHorizonMemoryAction:
        op = self._rng.choices(["add", "remove", "noop"], weights=[0.45, 0.25, 0.30], k=1)[0]
        if op == "remove":
            if memory_count == 0:
                return LongHorizonMemoryAction(operation="noop")
            idx = self._rng.randrange(memory_count)
            return LongHorizonMemoryAction(operation="remove", remove_index=idx)

        return LongHorizonMemoryAction(operation=op)

    def run_episode(self, environment: LongHorizonMemoryEnvironment) -> EpisodeResult:
        observation = environment.reset()

        episode_index = environment.episode
        episode_messages = environment.messages
        steps = len(episode_messages)

        cumulative_reward = 0.0
        for _ in range(steps):
            step_result = environment.step(self.act(memory_count=observation.memory_count))
            observation = step_result
            cumulative_reward += float(step_result.reward or 0.0)

        return EpisodeResult(
            episode_index=episode_index,
            steps=steps,
            reward=cumulative_reward,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random baseline agent on the environment.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty bucket to evaluate.",
    )
    args = parser.parse_args()

    os.environ["LONG_HORIZON_MEMORY_TASK"] = args.task
    if args.seed is not None:
        os.environ["LONG_HORIZON_MEMORY_SEED"] = str(args.seed)
    agent = RandomBaselineAgent(seed=args.seed)
    environment = LongHorizonMemoryEnvironment()

    total_reward = 0.0
    for episode_number in range(1, args.episodes + 1):
        result = agent.run_episode(environment)
        total_reward += result.reward
        print(
            f"episode={episode_number} source_episode={result.episode_index + 1} "
            f"steps={result.steps} reward={result.reward:.3f}"
        )

    average_reward = total_reward / max(args.episodes, 1)
    print(f"average_reward={average_reward:.3f}")


if __name__ == "__main__":
    main()