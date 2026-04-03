"""PPO update step with clipped surrogate objective and value loss."""

from __future__ import annotations

from itertools import chain
from typing import Dict

import torch
import torch.nn as nn

from ctf_pacman.agents.base_agent import BaseAgent
from ctf_pacman.training.rollout_buffer import RolloutBuffer
from ctf_pacman.utils.config import TrainingConfig


class PPOUpdater:
    """Proximal Policy Optimisation update for all agents.

    Uses a single shared Adam optimiser over the parameters of all four
    agents. Applies:
      - Clipped surrogate policy loss
      - Clipped value function loss
      - Entropy bonus (with linear decay schedule)
      - Gradient norm clipping
      - Linear learning-rate schedule

    Args:
        agents: Dict mapping agent_id -> BaseAgent.
        config: TrainingConfig with all PPO hypers.
        device: Torch device.
    """

    def __init__(
        self,
        agents: Dict[int, BaseAgent],
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.agents = agents
        self.config = config
        self.device = device

        self._initial_lr = config.learning_rate
        self._initial_ent = config.entropy_coeff

        # Single optimiser for all agent parameters
        all_params = chain(*[a.parameters() for a in agents.values()])
        self.optimizer = torch.optim.Adam(all_params, lr=config.learning_rate, eps=1e-5)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer, current_timestep: int) -> dict:
        """Run PPO epochs over the stored rollout.

        Args:
            buffer:            Completed RolloutBuffer (advantages computed).
            current_timestep:  Global step count for decay schedules.

        Returns:
            Dict of mean scalar metrics over all epochs/minibatches/agents:
              policy_loss, value_loss, entropy_loss, total_loss,
              approx_kl, clip_fraction.
        """
        cfg = self.config

        # Apply learning rate schedule (linear decay to 0)
        frac = max(0.0, 1.0 - current_timestep / cfg.total_timesteps)
        lr = self._initial_lr * frac
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Apply entropy coefficient schedule (linear decay to 0.001)
        entropy_coeff = self._initial_ent * frac + 0.001 * (1.0 - frac)

        # Accumulate losses across epochs and minibatches
        metrics: Dict[str, list] = {
            "policy_loss": [], "value_loss": [], "entropy_loss": [],
            "total_loss": [], "approx_kl": [], "clip_fraction": [],
        }

        num_agents = buffer.num_agents

        for _epoch in range(cfg.num_ppo_epochs):
            for batch in buffer.get_minibatches(cfg.num_minibatches):
                # batch tensors shape: (batch_size, num_agents, ...)
                b_grid = batch["obs_grid"]          # (B, A, C, H, W)
                b_flat = batch["obs_flat"]          # (B, A, flat)
                b_acts = batch["actions"]           # (B, A)
                b_lp_old = batch["log_probs_old"]   # (B, A)
                b_val_old = batch["values_old"]     # (B, A)
                b_adv = batch["advantages"]         # (B, A)
                b_ret = batch["returns"]            # (B, A)
                b_masks = batch["action_masks"]     # (B, A, 5)
                b_msgs = batch["messages_sent"]     # (B, A, msg_dim)
                b_gs = batch["global_states"]       # (B, 19)

                batch_pol_losses, batch_val_losses, batch_ent_losses = [], [], []
                batch_kls, batch_clips = [], []

                for aid, agent in self.agents.items():
                    # Per-agent slices along agent dimension
                    grid_a = b_grid[:, aid]            # (B, C, H, W)
                    flat_a = b_flat[:, aid]            # (B, flat)
                    acts_a = b_acts[:, aid]            # (B,)
                    lp_old_a = b_lp_old[:, aid]       # (B,)
                    val_old_a = b_val_old[:, aid]      # (B,)
                    adv_a = b_adv[:, aid]              # (B,)
                    ret_a = b_ret[:, aid]              # (B,)
                    mask_a = b_masks[:, aid]           # (B, 5)

                    # teammate message as received (from buffer)
                    teammate_id = self._teammate(aid)
                    msg_received = b_msgs[:, teammate_id]  # (B, msg_dim)

                    # Forward pass
                    out = agent.forward(
                        grid_obs=grid_a,
                        flat_obs=flat_a,
                        received_message=msg_received,
                        global_state=b_gs,
                        action_mask=mask_a,
                    )
                    dist: torch.distributions.Categorical = out["action_dist"]
                    new_lp = dist.log_prob(acts_a)       # (B,)
                    entropy = dist.entropy()              # (B,)
                    new_val = out["value"].squeeze(-1)    # (B,)

                    # Clip log probs to [-20, 0] to prevent exploding ratios (Part 11)
                    new_lp = new_lp.clamp(-20.0, 0.0)
                    lp_old_a_c = lp_old_a.clamp(-20.0, 0.0)

                    # --- Policy loss ---
                    log_ratio = new_lp - lp_old_a_c
                    ratio = log_ratio.exp()
                    surr1 = ratio * adv_a
                    surr2 = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv_a
                    pol_loss = -torch.min(surr1, surr2).mean()

                    # --- Clipped value loss ---
                    vf_loss_unclipped = (new_val - ret_a).pow(2)
                    val_clipped = val_old_a + (new_val - val_old_a).clamp(
                        -cfg.clip_epsilon, cfg.clip_epsilon
                    )
                    vf_loss_clipped = (val_clipped - ret_a).pow(2)
                    val_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                    ent_loss = -entropy.mean()

                    # --- KL & clip fraction diagnostics ---
                    with torch.no_grad():
                        approx_kl = (lp_old_a - new_lp).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item()

                    batch_pol_losses.append(pol_loss)
                    batch_val_losses.append(val_loss)
                    batch_ent_losses.append(ent_loss)
                    batch_kls.append(approx_kl)
                    batch_clips.append(clip_frac)

                # Sum losses across agents
                total_pol = sum(batch_pol_losses)
                total_val = sum(batch_val_losses)
                total_ent = sum(batch_ent_losses)
                total_loss = (
                    total_pol
                    + cfg.value_loss_coeff * total_val
                    + entropy_coeff * total_ent
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    chain(*[a.parameters() for a in self.agents.values()]),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"].append(total_pol.item() / num_agents)
                metrics["value_loss"].append(total_val.item() / num_agents)
                metrics["entropy_loss"].append(total_ent.item() / num_agents)
                metrics["total_loss"].append(total_loss.item() / num_agents)
                metrics["approx_kl"].append(float(sum(batch_kls) / num_agents))
                metrics["clip_fraction"].append(float(sum(batch_clips) / num_agents))

        return {k: float(sum(v) / max(1, len(v))) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _teammate(agent_id: int) -> int:
        """Return the teamate's agent_id."""
        # Team 0: agents 0,1  |  Team 1: agents 2,3
        return {0: 1, 1: 0, 2: 3, 3: 2}[agent_id]
