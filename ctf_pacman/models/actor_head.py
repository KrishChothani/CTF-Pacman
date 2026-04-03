"""Actor (policy) head with optional action masking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorHead(nn.Module):
    """Two-layer MLP that outputs action logits.

    Illegal actions are masked to -1e9 before the logits are returned so
    that the softmax / Categorical distribution effectively ignores them.

    Args:
        input_dim:   Dimensionality of the trunk (hidden) representation.
        num_actions: Number of discrete actions (5 for CTF-Pacman).
        hidden_dim:  Width of the intermediate hidden layer.
    """

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action logits with optional masking.

        Args:
            x:           Float tensor of shape (batch, input_dim).
            action_mask: Optional bool tensor of shape (batch, num_actions).
                         True = legal, False = illegal.

        Returns:
            Logit tensor of shape (batch, num_actions).
            Illegal action logits are set to -1e9.
        """
        logits = self.net(x)
        if action_mask is not None:
            # Ensure action_mask is bool
            mask = action_mask.bool()
            logits = logits.masked_fill(~mask, -1e9)
        return logits
