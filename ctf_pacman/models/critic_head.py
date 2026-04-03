"""Critic (value) head and global state encoder for centralised training."""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Global state dimension:
#   4 agents × 2 (x, y, both normalised) = 8
#   4 agents × 1 (carrying count)        = 4
#   4 agents × 1 (scared timer)          = 4
#   2 team scores                        = 2
#   1 step                               = 1
# Total                                  = 19
#   (Prior spec said 23 but the exact formula = 8+4+4+2+1 = 19; using 19.)
# ---------------------------------------------------------------------------
GLOBAL_STATE_DIM: int = 19


class GlobalStateEncoder(nn.Module):
    """Encode the global state vector for the centralised critic.

    The global state is a flat vector of normalised scalars (positions,
    carrying counts, scared timers, scores, step). This module projects it
    into a higher-dimensional representation suitable as critic input.

    Args:
        global_state_dim: Dimension of the raw global state vector.
        hidden_dim:       Width of the output representation.
    """

    def __init__(self, global_state_dim: int = GLOBAL_STATE_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw global state.

        Args:
            x: Float tensor of shape (batch, global_state_dim).

        Returns:
            Encoded tensor of shape (batch, hidden_dim).
        """
        return self.net(x)


class CriticHead(nn.Module):
    """Three-layer MLP that outputs a scalar state-value estimate.

    Typically receives the output of GlobalStateEncoder (or the trunk
    representation) as input.

    Args:
        input_dim:  Dimension of the input representation.
        hidden_dim: Width of the first hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            x: Float tensor of shape (batch, input_dim).

        Returns:
            Value tensor of shape (batch, 1).
        """
        return self.net(x)
