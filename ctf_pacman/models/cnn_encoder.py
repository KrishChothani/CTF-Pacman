"""CNN encoder for processing local grid observations."""

from __future__ import annotations

import torch
import torch.nn as nn

from ctf_pacman.utils.config import ModelConfig


class CNNEncoder(nn.Module):
    """Multi-layer convolutional encoder for local grid observations.

    Automatically computes the flattened output dimension by running a dummy
    forward pass during construction.

    Args:
        config:      ModelConfig specifying conv layer parameters.
        in_channels: Number of input channels (= num_observation_channels).
        input_hw:    Spatial size of the input (height == width = 2r+1).
    """

    def __init__(self, config: ModelConfig, in_channels: int, input_hw: int) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_ch, kernel, stride in zip(
            config.cnn_channels,
            config.cnn_kernel_sizes,
            config.cnn_strides,
        ):
            layers.append(
                nn.Conv2d(
                    current_channels, out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,  # same-ish padding to preserve spatial size
                )
            )
            layers.append(nn.ReLU())
            current_channels = out_ch

        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

        # Compute output dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw, input_hw)
            self.output_dim: int = int(self.net(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grid observations.

        Args:
            x: Float tensor of shape (batch, in_channels, H, W).

        Returns:
            Flattened feature tensor of shape (batch, output_dim).
        """
        return self.net(x)
