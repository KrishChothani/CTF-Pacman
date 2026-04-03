"""Pre-allocated rollout buffer with GAE computation and minibatch iteration."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size circular buffer for storing PPO rollout data.

    All tensors are pre-allocated on the target device to avoid repeated
    allocations during training. After ``rollout_length`` steps are filled,
    call ``compute_returns_and_advantages`` then iterate over minibatches
    with ``get_minibatches``.

    Args:
        rollout_length:  Number of environment steps per rollout.
        num_envs:        Number of parallel environments.
        num_agents:      Number of agents per environment (4).
        obs_shape_grid:  Shape tuple for grid observation (C, H, W).
        obs_shape_flat:  Shape tuple for flat observation (flat_dim,).
        message_dim:     Dimension of agent communication messages.
        device:          Torch device for all tensors.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        obs_shape_grid: tuple,
        obs_shape_flat: tuple,
        message_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.device = device
        self.ptr: int = 0

        T, E, A = rollout_length, num_envs, num_agents

        self.obs_grid = torch.zeros((T, E, A, *obs_shape_grid), dtype=torch.float32, device=device)
        self.obs_flat = torch.zeros((T, E, A, *obs_shape_flat), dtype=torch.float32, device=device)
        self.actions = torch.zeros((T, E, A), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((T, E, A), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((T, E, A), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, E, A), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, E), dtype=torch.float32, device=device)
        self.action_masks = torch.zeros((T, E, A, 5), dtype=torch.bool, device=device)
        self.messages_sent = torch.zeros((T, E, A, message_dim), dtype=torch.float32, device=device)
        self.global_states = torch.zeros((T, E, 19), dtype=torch.float32, device=device)

        # Computed by compute_returns_and_advantages
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Insert one timestep
    # ------------------------------------------------------------------

    def insert(self, step: int, **kwargs) -> None:
        """Write one timestep of data.

        Expected keyword arguments (all torch.Tensors):
            obs_grid:      (num_envs, num_agents, C, H, W)
            obs_flat:      (num_envs, num_agents, flat_dim)
            actions:       (num_envs, num_agents)
            log_probs:     (num_envs, num_agents)
            rewards:       (num_envs, num_agents)
            values:        (num_envs, num_agents)
            dones:         (num_envs,)
            action_masks:  (num_envs, num_agents, 5)
            messages_sent: (num_envs, num_agents, message_dim)
            global_states: (num_envs, 19)

        Args:
            step:   Must equal current self.ptr.
            **kwargs: Data tensors as above.
        """
        assert step == self.ptr, f"Expected ptr={self.ptr}, got step={step}"
        t = self.ptr

        self.obs_grid[t] = kwargs["obs_grid"]
        self.obs_flat[t] = kwargs["obs_flat"]
        self.actions[t] = kwargs["actions"]
        self.log_probs[t] = kwargs["log_probs"]
        self.rewards[t] = kwargs["rewards"]
        self.values[t] = kwargs["values"]
        self.dones[t] = kwargs["dones"]
        self.action_masks[t] = kwargs["action_masks"]
        self.messages_sent[t] = kwargs["messages_sent"]
        self.global_states[t] = kwargs["global_states"]

        self.ptr += 1

    # ------------------------------------------------------------------
    # GAE advantage computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns in-place.

        Operates independently for each agent. Advantages are normalised
        across (rollout_length × num_envs) for each agent.

        Args:
            last_values: (num_envs, num_agents) bootstrap values for step T.
            last_dones:  (num_envs,) done flags at step T.
            gamma:       Discount factor.
            gae_lambda:  GAE lambda parameter.
        """
        T, E, A = self.rollout_length, self.num_envs, self.num_agents

        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(E, A, dtype=torch.float32, device=self.device)

        # Broadcast done flag to all agents
        last_done_expanded = last_dones.unsqueeze(-1).expand(E, A)  # (E, A)

        next_values = last_values  # (E, A)
        next_done = last_done_expanded  # (E, A)

        for t in reversed(range(T)):
            if t == T - 1:
                nv = next_values
                nd = next_done
            else:
                nv = self.values[t + 1]
                nd = self.dones[t + 1].unsqueeze(-1).expand(E, A)

            delta = (
                self.rewards[t]
                + gamma * nv * (1.0 - nd)
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * (1.0 - nd) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

        # Normalize per-agent across (T, E) dimensions
        for a in range(A):
            adv_a = self.advantages[:, :, a]
            self.advantages[:, :, a] = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

    # ------------------------------------------------------------------
    # Minibatch iteration
    # ------------------------------------------------------------------

    def get_minibatches(self, num_minibatches: int) -> Iterator[dict]:
        """Flatten, shuffle, and yield minibatch dicts.

        Args:
            num_minibatches: Number of equal-sized chunks.

        Yields:
            Dict of tensors for one minibatch. Keys match the buffer fields
            plus "advantages" and "returns".
        """
        T, E, A = self.rollout_length, self.num_envs, self.num_agents
        N = T * E  # total samples

        # Flatten (T, E) → N
        def flat(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape[2:]
            return x.reshape(N, *shape)

        flat_grid = flat(self.obs_grid)          # (N, A, C, H, W)
        flat_flat = flat(self.obs_flat)          # (N, A, flat_dim)
        flat_act = flat(self.actions)            # (N, A)
        flat_lp = flat(self.log_probs)           # (N, A)
        flat_rew = flat(self.rewards)            # (N, A)
        flat_val = flat(self.values)             # (N, A)
        flat_done = self.dones.reshape(N)        # (N,)
        flat_mask = flat(self.action_masks)      # (N, A, 5)
        flat_msg = flat(self.messages_sent)      # (N, A, msg_dim)
        flat_gs = self.global_states.reshape(N, -1)  # (N, 19)
        flat_adv = flat(self.advantages)         # (N, A)
        flat_ret = flat(self.returns)            # (N, A)

        perm = torch.randperm(N, device=self.device)
        batch_size = N // num_minibatches

        for i in range(num_minibatches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            yield {
                "obs_grid": flat_grid[idx],
                "obs_flat": flat_flat[idx],
                "actions": flat_act[idx],
                "log_probs_old": flat_lp[idx],
                "rewards": flat_rew[idx],
                "values_old": flat_val[idx],
                "dones": flat_done[idx],
                "action_masks": flat_mask[idx],
                "messages_sent": flat_msg[idx],
                "global_states": flat_gs[idx],
                "advantages": flat_adv[idx],
                "returns": flat_ret[idx],
            }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the write pointer; existing tensor data is overwritten."""
        self.ptr = 0
        self.advantages = None
        self.returns = None
