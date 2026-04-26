# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Long Horizon Memory Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
except ImportError:
    from .models import LongHorizonMemoryAction, LongHorizonMemoryObservation


class LongHorizonMemoryEnv(
    EnvClient[LongHorizonMemoryAction, LongHorizonMemoryObservation, State]
):
    """
    Client for the Long Horizon Memory Environment.

    Uses the same action/observation schema as the server and supports low-latency
    multi-step WebSocket episodes.
    """

    def _step_payload(self, action: LongHorizonMemoryAction) -> Dict:
        """
        Convert LongHorizonMemoryAction to JSON payload for step message.

        Args:
            action: LongHorizonMemoryAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"operation": action.operation}
        if action.remove_index is not None:
            payload["remove_index"] = action.remove_index
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[LongHorizonMemoryObservation]:
        """
        Parse server response into StepResult[LongHorizonMemoryObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LongHorizonMemoryObservation
        """
        obs_data = payload.get("observation", {})
        observation = LongHorizonMemoryObservation(
            domain=obs_data.get("domain", "long_horizon_memory"),
            task_name=obs_data.get("task_name", "easy"),
            new_message=obs_data.get("new_message", ""),
            memory=obs_data.get("memory", []),
            memory_count=obs_data.get("memory_count", 0),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
