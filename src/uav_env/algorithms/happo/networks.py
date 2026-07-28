"""Independent HAPPO actors and scalar centralized critic."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from uav_env.algorithms.mappo.networks import SharedActor, _mlp


class IndependentActorSet(nn.Module):
    """Container of parameter-independent discrete actors.

    The current fixed homogeneous 3v3 environment uses identical observation
    and action dimensions for every red UAV, but this class accepts per-agent
    dimension lists so future heterogeneous experiments do not need a new API.
    """

    def __init__(
        self,
        obs_dims: Sequence[int],
        action_dims: Sequence[int],
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if len(obs_dims) != len(action_dims) or not obs_dims:
            raise ValueError("obs_dims and action_dims must have the same nonzero length")
        actors: list[SharedActor] = []
        for index, (obs_dim, action_dim) in enumerate(zip(obs_dims, action_dims)):
            if seed is not None:
                torch.manual_seed(int(seed) + 9973 * (index + 1))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed) + 9973 * (index + 1))
            actors.append(SharedActor(int(obs_dim), int(action_dim), hidden_sizes, activation))
        self.actors = nn.ModuleList(actors)
        self.obs_dims = tuple(int(v) for v in obs_dims)
        self.action_dims = tuple(int(v) for v in action_dims)

    def __len__(self) -> int:
        return len(self.actors)

    def __getitem__(self, agent_id: int) -> SharedActor:
        return self.actors[int(agent_id)]

    def forward_agent(self, agent_id: int, observation: Tensor, available_actions: Tensor | None = None) -> Tensor:
        """Forward one actor only."""

        return self.actors[int(agent_id)](observation, available_actions)

    def forward(self, observations: Tensor, available_actions: Tensor | None = None) -> list[Tensor]:
        """Return per-agent logits without implying parameter sharing."""

        if observations.shape[-2] != len(self.actors):
            raise ValueError("observations must include one row per actor")
        outputs: list[Tensor] = []
        for agent_id, actor in enumerate(self.actors):
            mask = None if available_actions is None else available_actions[..., agent_id, :]
            outputs.append(actor(observations[..., agent_id, :], mask))
        return outputs


class JointCentralizedCritic(nn.Module):
    """Scalar centralized value function V(global_state)."""

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (256, 256), activation: str = "relu") -> None:
        super().__init__()
        self.net = _mlp(int(state_dim), hidden_sizes, 1, activation, 1.0)

    def forward(self, global_states: Tensor) -> Tensor:
        values = self.net(global_states.reshape(-1, global_states.shape[-1])).squeeze(-1)
        return values.reshape(*global_states.shape[:-1])
