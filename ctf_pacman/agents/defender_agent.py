"""Defender agent — role string 'defender', delegates to BaseAgent."""

from __future__ import annotations

import torch

from ctf_pacman.agents.base_agent import BaseAgent
from ctf_pacman.utils.config import AgentConfig, ModelConfig


class DefenderAgent(BaseAgent):
    """Neural network defender agent.

    Inherits the full network architecture from BaseAgent. The only
    specialisation is the role string ``"defender"``, which influences
    reward routing in the RewardCalculator.

    Args:
        agent_id:          Integer ID (1 or 3).
        team_id:           0 or 1.
        config:            AgentConfig.
        model_config:      ModelConfig.
        observation_radius: From env config.
        num_obs_channels:  From env config.
        device:            Torch device.
    """

    def __init__(
        self,
        agent_id: int,
        team_id: int,
        config: AgentConfig,
        model_config: ModelConfig,
        observation_radius: int,
        num_obs_channels: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            team_id=team_id,
            role="defender",
            config=config,
            model_config=model_config,
            observation_radius=observation_radius,
            num_obs_channels=num_obs_channels,
            device=device,
        )

    def build_network(self) -> None:
        """Build network using the shared BaseAgent implementation."""
        self._build_network_impl()
