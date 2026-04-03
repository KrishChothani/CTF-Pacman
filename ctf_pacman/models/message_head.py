"""Message head for intra-team communication."""

from __future__ import annotations

import torch
import torch.nn as nn


class MessageHead(nn.Module):
    """Two-layer MLP that produces a bounded communication vector.

    The output is passed through Tanh so all message components lie in
    [-1, 1], providing a natural normalisation for the receiving agent's
    trunk MLP.

    Args:
        input_dim:   Dimension of the trunk representation.
        message_dim: Length of the output message vector.
        hidden_dim:  Width of the intermediate hidden layer.
    """

    def __init__(self, input_dim: int, message_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh(),
        )
        self.message_dim = message_dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Generate a communication message from the trunk representation.

        Args:
            hidden: Float tensor of shape (batch, input_dim).

        Returns:
            Message tensor of shape (batch, message_dim), values in [-1, 1].
        """
        return self.net(hidden)
