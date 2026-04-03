"""Abstract base agent with full network assembly and act/forward API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ctf_pacman.models.actor_head import ActorHead
from ctf_pacman.models.critic_head import CriticHead, GlobalStateEncoder, GLOBAL_STATE_DIM
from ctf_pacman.models.cnn_encoder import CNNEncoder
from ctf_pacman.models.message_head import MessageHead
from ctf_pacman.utils.config import AgentConfig, ModelConfig

_NUM_ACTIONS = 5


class BaseAgent(ABC, nn.Module):
    """Abstract base for all CTF-Pacman agents.

    Subclasses must implement ``build_network()`` to set up model components
    (though in practice the default implementation here is used, and
    subclasses just set the role string).

    Network architecture:
        1. CNNEncoder: processes the local grid observation.
        2. Trunk MLP: fuses [CNN output | flat features | received message].
        3. ActorHead: policy logits over 5 actions.
        4. GlobalStateEncoder + CriticHead: centralised value estimate.
        5. MessageHead: outgoing communication vector.

    Args:
        agent_id:     Integer ID (0–3).
        team_id:      0 or 1.
        role:         "attacker" or "defender".
        config:       AgentConfig.
        model_config: ModelConfig.
        env_config:   Minimal env info needed to size the network.
        device:       Torch device.
    """

    def __init__(
        self,
        agent_id: int,
        team_id: int,
        role: str,
        config: AgentConfig,
        model_config: ModelConfig,
        observation_radius: int,
        num_obs_channels: int,
        device: torch.device,
    ) -> None:
        nn.Module.__init__(self)
        self.agent_id = agent_id
        self.team_id = team_id
        self.role = role
        self.config = config
        self.model_config = model_config
        self.observation_radius = observation_radius
        self.num_obs_channels = num_obs_channels
        self.device = device

        self.build_network()
        self.to(device)

    # ------------------------------------------------------------------
    # Network assembly (can be overridden by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_network(self) -> None:
        """Instantiate all network components and store as attributes."""
        ...

    def _build_network_impl(self) -> None:
        """Shared implementation called by concrete subclasses."""
        cfg = self.model_config
        r = self.observation_radius
        in_channels = self.num_obs_channels
        window = 2 * r + 1
        flat_dim = 8          # fixed flat feature size from ObservationBuilder
        msg_dim = self.config.message_dim if self.config.use_communication else 0

        # CNN encoder
        self.cnn = CNNEncoder(cfg, in_channels, window)
        cnn_out = self.cnn.output_dim

        # Trunk MLP: [cnn_out + flat_dim + msg_dim] → hidden_dim
        trunk_in = cnn_out + flat_dim + msg_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = ActorHead(cfg.hidden_dim, _NUM_ACTIONS, cfg.actor_hidden_dim)

        # Centralised critic
        self.global_encoder = GlobalStateEncoder(GLOBAL_STATE_DIM, cfg.critic_hidden_dim)
        self.critic = CriticHead(cfg.critic_hidden_dim, cfg.critic_hidden_dim)

        # Message head
        self.message_head = MessageHead(cfg.hidden_dim, self.config.message_dim, cfg.message_hidden_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        grid_obs: torch.Tensor,
        flat_obs: torch.Tensor,
        received_message: Optional[torch.Tensor] = None,
        global_state: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward pass through all network heads.

        Args:
            grid_obs:          (batch, C, H, W) float32
            flat_obs:          (batch, flat_dim) float32
            received_message:  (batch, message_dim) float32 or None
            global_state:      (batch, GLOBAL_STATE_DIM) float32 or None
            action_mask:       (batch, num_actions) bool or None

        Returns:
            Dict with keys:
              - "action_logits": (batch, num_actions)
              - "value":         (batch, 1)
              - "message":       (batch, message_dim)
              - "action_dist":   torch.distributions.Categorical
              - "hidden":        (batch, hidden_dim) trunk output
        """
        # CNN
        cnn_out = self.cnn(grid_obs)

        # Trunk input
        parts = [cnn_out, flat_obs]
        if self.config.use_communication:
            if received_message is not None:
                parts.append(received_message)
            else:
                parts.append(torch.zeros(
                    cnn_out.shape[0], self.config.message_dim,
                    device=cnn_out.device, dtype=cnn_out.dtype,
                ))

        hidden = self.trunk(torch.cat(parts, dim=-1))

        # Actor
        logits = self.actor(hidden, action_mask)

        # Critic
        if global_state is not None:
            gs_enc = self.global_encoder(global_state)
        else:
            gs_enc = torch.zeros(
                hidden.shape[0], self.model_config.critic_hidden_dim,
                device=hidden.device, dtype=hidden.dtype,
            )
        value = self.critic(gs_enc)

        # Message
        message = self.message_head(hidden)

        # Distribution
        dist = torch.distributions.Categorical(logits=logits)

        return {
            "action_logits": logits,
            "value": value,
            "message": message,
            "action_dist": dist,
            "hidden": hidden,
        }

    # ------------------------------------------------------------------
    # act — convenience wrapper used during rollout collection
    # ------------------------------------------------------------------

    def act(
        self,
        obs: dict,
        received_message: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        """Sample (or greedily select) an action given an observation.

        Args:
            obs:              Dict with "grid" and "flat" numpy arrays or tensors.
            received_message: Optional teammate message tensor (1, message_dim).
            action_mask:      Optional boolean action mask (1, num_actions).
            deterministic:    If True, take argmax of logits.

        Returns:
            Tuple of (action: int, log_prob: float, entropy: Tensor).
        """
        self.eval()
        with torch.no_grad():
            grid = self._to_tensor(obs["grid"]).unsqueeze(0)   # (1, C, H, W)
            flat = self._to_tensor(obs["flat"]).unsqueeze(0)   # (1, flat_dim)

            if received_message is not None and received_message.dim() == 1:
                received_message = received_message.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            out = self.forward(grid, flat, received_message, global_state=None,
                               action_mask=action_mask)
            dist: torch.distributions.Categorical = out["action_dist"]

            if deterministic:
                action = int(out["action_logits"].argmax(dim=-1).item())
            else:
                action = int(dist.sample().item())

            log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
            entropy = float(dist.entropy().item())

        self.train()
        return action, log_prob, entropy

    # ------------------------------------------------------------------
    # get_value
    # ------------------------------------------------------------------

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Compute the critic value for a given global state.

        Args:
            global_state: (batch, GLOBAL_STATE_DIM) float32.

        Returns:
            (batch, 1) value tensor.
        """
        gs_enc = self.global_encoder(global_state)
        return self.critic(gs_enc)

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True):
        return nn.Module.parameters(self, recurse=recurse)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)
