"""Main training loop: VecEnv, rollout collection, PPO updates, self-play."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ctf_pacman.agents.attacker_agent import AttackerAgent
from ctf_pacman.agents.base_agent import BaseAgent
from ctf_pacman.agents.defender_agent import DefenderAgent
from ctf_pacman.environment.env import CTFPacmanEnv
from ctf_pacman.models.critic_head import GLOBAL_STATE_DIM
from ctf_pacman.training.ppo import PPOUpdater
from ctf_pacman.training.rollout_buffer import RolloutBuffer
from ctf_pacman.training.self_play_manager import SelfPlayManager
from ctf_pacman.utils.config import Config
from ctf_pacman.utils.logger import Logger
from ctf_pacman.utils.metrics import EpisodeMetrics, MetricsAggregator
from ctf_pacman.utils.seed import set_global_seed

_NUM_AGENTS = 4
_AGENT_TEAMS = {0: 0, 1: 0, 2: 1, 3: 1}
_AGENT_ROLES = {0: "attacker", 1: "defender", 2: "attacker", 3: "defender"}


# ---------------------------------------------------------------------------
# Synchronous vectorised environment wrapper
# ---------------------------------------------------------------------------

class VecEnv:
    """Synchronous vectorised wrapper around multiple CTFPacmanEnv instances.

    Args:
        envs: List of pre-constructed CTFPacmanEnv instances.
    """

    def __init__(self, envs: List[CTFPacmanEnv]) -> None:
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self) -> List[dict]:
        """Reset all envs and return per-env observation dicts."""
        return [env.reset()[0] for env in self.envs]

    def step(self, actions_batch: List[Dict[int, int]]) -> List[tuple]:
        """Step all envs with their respective action dicts.

        Args:
            actions_batch: List[Dict[agent_id -> action]] of length num_envs.

        Returns:
            List of (obs, rewards, terminated, truncated, info) tuples.
        """
        results = []
        for env, actions in zip(self.envs, actions_batch):
            results.append(env.step(actions))
        return results

    def get_legal_action_masks(self, env_idx: int) -> Dict[int, np.ndarray]:
        return {aid: self.envs[env_idx].get_legal_action_mask(aid) for aid in range(_NUM_AGENTS)}


# ---------------------------------------------------------------------------
# Global state construction
# ---------------------------------------------------------------------------

def build_global_state(state: dict, config) -> np.ndarray:
    """Construct the normalised global state vector for the centralised critic.

    Vector layout (19 floats total):
      - 4 agents × (x_norm, y_norm)   = 8
      - 4 agents × carry_norm          = 4
      - 4 agents × scared_norm         = 4
      - 2 team scores (norm)           = 2
      - 1 step (norm)                  = 1

    Args:
        state:  Full CTFPacmanEnv state dict.
        config: EnvConfig.

    Returns:
        float32 ndarray of shape (19,).
    """
    W, H = config.map_width, config.map_height
    pos = state["agent_positions"]
    carrying = state["agent_carrying"]
    scared = state["agent_scared"]
    scores = state["scores"]
    step = state["step"]

    feats = []
    for aid in range(_NUM_AGENTS):
        x, y = pos[aid]
        feats.append(x / (W - 1))
        feats.append(y / (H - 1))
    for aid in range(_NUM_AGENTS):
        feats.append(carrying[aid] / max(1, config.num_food_per_team))
    for aid in range(_NUM_AGENTS):
        feats.append(scared[aid] / max(1, config.power_pellet_duration))
    feats.append(scores[0] / max(1, config.num_food_per_team))
    feats.append(scores[1] / max(1, config.num_food_per_team))
    feats.append(step / max(1, config.max_steps))

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates training: environment vectorisation, rollout collection,
    PPO updates, self-play snapshots, checkpointing, and logging.

    Args:
        config: Top-level Config dataclass.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        set_global_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        # All four agents
        self.agents: Dict[int, BaseAgent] = self._build_agents()

        # Vectorised envs
        self.vec_env = VecEnv([
            CTFPacmanEnv(config.env, seed=config.seed + i)
            for i in range(config.training.num_envs)
        ])

        # Rollout buffer
        r = config.env.observation_radius
        window = 2 * r + 1
        C = config.env.num_observation_channels
        self.buffer = RolloutBuffer(
            rollout_length=config.training.rollout_length,
            num_envs=config.training.num_envs,
            num_agents=_NUM_AGENTS,
            obs_shape_grid=(C, window, window),
            obs_shape_flat=(8,),
            message_dim=config.agent.message_dim,
            device=self.device,
        )

        # PPO updater
        self.ppo = PPOUpdater(self.agents, config.training, self.device)

        # Self-play
        self.self_play = SelfPlayManager(config.training, self._build_agent_pair)

        # Logger and metrics
        self.logger = Logger(config.logging)
        self.agg = MetricsAggregator()

        # Checkpoint directory
        self.ckpt_dir = os.path.join(config.logging.log_dir, config.logging.experiment_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Per-env episode tracking
        self._ep_returns: List[Dict[int, float]] = [
            {i: 0.0 for i in range(_NUM_AGENTS)} for _ in range(config.training.num_envs)
        ]
        self._ep_lengths: List[int] = [0] * config.training.num_envs
        self._ep_food_col: List[Dict[int, int]] = [
            {i: 0 for i in range(_NUM_AGENTS)} for _ in range(config.training.num_envs)
        ]
        self._ep_food_ret: List[Dict[int, int]] = [
            {i: 0 for i in range(_NUM_AGENTS)} for _ in range(config.training.num_envs)
        ]
        self._ep_caps_made: List[Dict[int, int]] = [
            {i: 0 for i in range(_NUM_AGENTS)} for _ in range(config.training.num_envs)
        ]
        self._ep_caps_suf: List[Dict[int, int]] = [
            {i: 0 for i in range(_NUM_AGENTS)} for _ in range(config.training.num_envs)
        ]

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agents(self) -> Dict[int, BaseAgent]:
        """Build all four agents."""
        cfg = self.config
        agents: Dict[int, BaseAgent] = {}
        for aid in range(_NUM_AGENTS):
            team = _AGENT_TEAMS[aid]
            role = _AGENT_ROLES[aid]
            if role == "attacker":
                agents[aid] = AttackerAgent(
                    agent_id=aid, team_id=team,
                    config=cfg.agent, model_config=cfg.model,
                    observation_radius=cfg.env.observation_radius,
                    num_obs_channels=cfg.env.num_observation_channels,
                    device=self.device,
                )
            else:
                agents[aid] = DefenderAgent(
                    agent_id=aid, team_id=team,
                    config=cfg.agent, model_config=cfg.model,
                    observation_radius=cfg.env.observation_radius,
                    num_obs_channels=cfg.env.num_observation_channels,
                    device=self.device,
                )
        return agents

    def _build_agent_pair(self, team_id: int) -> Tuple[AttackerAgent, DefenderAgent]:
        """Construct a fresh (attacker, defender) pair for self-play."""
        cfg = self.config
        attacker_id = 0 if team_id == 0 else 2
        defender_id = 1 if team_id == 0 else 3
        attacker = AttackerAgent(
            agent_id=attacker_id, team_id=team_id,
            config=cfg.agent, model_config=cfg.model,
            observation_radius=cfg.env.observation_radius,
            num_obs_channels=cfg.env.num_observation_channels,
            device=self.device,
        )
        defender = DefenderAgent(
            agent_id=defender_id, team_id=team_id,
            config=cfg.agent, model_config=cfg.model,
            observation_radius=cfg.env.observation_radius,
            num_obs_channels=cfg.env.num_observation_channels,
            device=self.device,
        )
        return attacker, defender

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config
        tcfg = cfg.training

        total_steps = 0
        obs_list = self.vec_env.reset()   # List[Dict[agent_id -> {"grid", "flat"}]]

        # Initialize agent messages (will be overwritten each step)
        # Shape: (num_envs, num_agents, message_dim)
        messages = torch.zeros(
            tcfg.num_envs, _NUM_AGENTS, cfg.agent.message_dim,
            device=self.device,
        )

        print(f"[Trainer] Starting training for {tcfg.total_timesteps:,} timesteps")
        t_start = time.time()

        while total_steps < tcfg.total_timesteps:
            # ============================================================
            # Rollout collection (no grad needed during environment stepping)
            # ============================================================
            with torch.no_grad():
                for step in range(tcfg.rollout_length):
                    # Tensors for this step
                    step_obs_grid = torch.zeros(
                        tcfg.num_envs, _NUM_AGENTS,
                        cfg.env.num_observation_channels,
                        2 * cfg.env.observation_radius + 1,
                        2 * cfg.env.observation_radius + 1,
                        device=self.device,
                    )
                    step_obs_flat = torch.zeros(
                        tcfg.num_envs, _NUM_AGENTS, 8,
                        device=self.device,
                    )
                    step_actions = torch.zeros(tcfg.num_envs, _NUM_AGENTS, dtype=torch.long, device=self.device)
                    step_log_probs = torch.zeros(tcfg.num_envs, _NUM_AGENTS, device=self.device)
                    step_values = torch.zeros(tcfg.num_envs, _NUM_AGENTS, device=self.device)
                    step_masks = torch.zeros(tcfg.num_envs, _NUM_AGENTS, 5, dtype=torch.bool, device=self.device)
                    step_msgs_out = torch.zeros(tcfg.num_envs, _NUM_AGENTS, cfg.agent.message_dim, device=self.device)

                    actions_batch: List[Dict[int, int]] = []

                    for e_idx in range(tcfg.num_envs):
                        env_obs = obs_list[e_idx]
                        env_actions: Dict[int, int] = {}

                        for aid, agent in self.agents.items():
                            grid_t = torch.tensor(
                                env_obs[aid]["grid"], dtype=torch.float32, device=self.device
                            ).unsqueeze(0)
                            flat_t = torch.tensor(
                                env_obs[aid]["flat"], dtype=torch.float32, device=self.device
                            ).unsqueeze(0)

                            teammate_id = PPOUpdater._teammate(aid)
                            recv_msg = messages[e_idx, teammate_id].unsqueeze(0)

                            legal_mask = self.vec_env.envs[e_idx].get_legal_action_mask(aid)
                            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

                            if hasattr(agent, "forward"):
                                out = agent.forward(grid_t, flat_t, recv_msg, global_state=None, action_mask=mask_t)
                                dist = out["action_dist"]
                                action = int(dist.sample().item())
                                log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
                                value = out["value"].squeeze().item()
                                msg_out = out["message"].squeeze(0)
                            else:
                                # Rule-based: use inference_mode for zero overhead
                                with torch.inference_mode():
                                    enriched = dict(self.vec_env.envs[e_idx]._state)
                                    enriched["grid"] = self.vec_env.envs[e_idx]._grid
                                    enriched["width"] = cfg.env.map_width
                                    enriched["height"] = cfg.env.map_height
                                    action = agent.act(enriched, aid)
                                log_prob = 0.0
                                value = 0.0
                                msg_out = torch.zeros(cfg.agent.message_dim, device=self.device)

                            step_obs_grid[e_idx, aid] = grid_t.squeeze(0)
                            step_obs_flat[e_idx, aid] = flat_t.squeeze(0)
                            step_actions[e_idx, aid] = action
                            step_log_probs[e_idx, aid] = log_prob
                            step_values[e_idx, aid] = value
                            step_masks[e_idx, aid] = mask_t.squeeze(0)
                            step_msgs_out[e_idx, aid] = msg_out
                            env_actions[aid] = action

                        actions_batch.append(env_actions)

                    # Update messages for next step
                    messages = step_msgs_out.detach()

                    # Step all environments
                    results = self.vec_env.step(actions_batch)

                    step_rewards = torch.zeros(tcfg.num_envs, _NUM_AGENTS, device=self.device)
                    step_dones = torch.zeros(tcfg.num_envs, device=self.device)
                    step_gs = torch.zeros(tcfg.num_envs, GLOBAL_STATE_DIM, device=self.device)

                    for e_idx, (new_obs, rewards, terminated, truncated, info) in enumerate(results):
                        done = any(terminated.values()) or any(truncated.values())
                        step_dones[e_idx] = float(done)

                        for aid in range(_NUM_AGENTS):
                            r_val = rewards.get(aid, 0.0)
                            step_rewards[e_idx, aid] = r_val
                            self._ep_returns[e_idx][aid] += r_val

                        self._ep_lengths[e_idx] += 1

                        from ctf_pacman.environment.events import (
                            FoodCollectedEvent, FoodReturnedEvent,
                            AgentCapturedEvent, EpisodeEndEvent,
                        )
                        for evt in info.get(0, {}).get("events", []):
                            if isinstance(evt, FoodCollectedEvent):
                                self._ep_food_col[e_idx][evt.agent_id] += 1
                            elif isinstance(evt, FoodReturnedEvent):
                                self._ep_food_ret[e_idx][evt.agent_id] += evt.food_count
                            elif isinstance(evt, AgentCapturedEvent):
                                self._ep_caps_made[e_idx][evt.capturing_id] += 1
                                self._ep_caps_suf[e_idx][evt.captured_id] += 1

                        gs = build_global_state(self.vec_env.envs[e_idx]._state, cfg.env)
                        step_gs[e_idx] = torch.tensor(gs, device=self.device)

                        if done:
                            scores = info.get(0, {}).get("scores", {0: 0, 1: 0})
                            winner = 0
                            if scores[0] > scores[1]:
                                winner = 1
                            elif scores[1] > scores[0]:
                                winner = -1
                            ep_m = EpisodeMetrics(
                                episode_return=dict(self._ep_returns[e_idx]),
                                episode_length=self._ep_lengths[e_idx],
                                food_collected=dict(self._ep_food_col[e_idx]),
                                food_returned=dict(self._ep_food_ret[e_idx]),
                                captures_made=dict(self._ep_caps_made[e_idx]),
                                captures_suffered=dict(self._ep_caps_suf[e_idx]),
                                win=winner,
                                score_team0=scores[0],
                                score_team1=scores[1],
                            )
                            self.agg.add(ep_m)
                            self._ep_returns[e_idx] = {i: 0.0 for i in range(_NUM_AGENTS)}
                            self._ep_lengths[e_idx] = 0
                            self._ep_food_col[e_idx] = {i: 0 for i in range(_NUM_AGENTS)}
                            self._ep_food_ret[e_idx] = {i: 0 for i in range(_NUM_AGENTS)}
                            self._ep_caps_made[e_idx] = {i: 0 for i in range(_NUM_AGENTS)}
                            self._ep_caps_suf[e_idx] = {i: 0 for i in range(_NUM_AGENTS)}
                            new_obs_reset = self.vec_env.envs[e_idx].reset()[0]
                            new_obs = new_obs_reset

                        obs_list[e_idx] = new_obs

                    self.buffer.insert(
                        step=step,
                        obs_grid=step_obs_grid,
                        obs_flat=step_obs_flat,
                        actions=step_actions,
                        log_probs=step_log_probs,
                        rewards=step_rewards,
                        values=step_values,
                        dones=step_dones,
                        action_masks=step_masks,
                        messages_sent=step_msgs_out,
                        global_states=step_gs,
                    )

                    total_steps += tcfg.num_envs

            # ============================================================
            # Compute advantages using bootstrap values
            # ============================================================
            last_values = torch.zeros(tcfg.num_envs, _NUM_AGENTS, device=self.device)
            last_dones = torch.zeros(tcfg.num_envs, device=self.device)

            for e_idx in range(tcfg.num_envs):
                env_obs = obs_list[e_idx]
                for aid, agent in self.agents.items():
                    if not hasattr(agent, "get_value"):
                        continue
                    gs = build_global_state(self.vec_env.envs[e_idx]._state, cfg.env)
                    gs_t = torch.tensor(gs, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        v = agent.get_value(gs_t)
                    last_values[e_idx, aid] = v.item()

            self.buffer.compute_returns_and_advantages(
                last_values, last_dones,
                gamma=tcfg.gamma,
                gae_lambda=tcfg.gae_lambda,
            )

            # ============================================================
            # PPO update
            # ============================================================
            losses = self.ppo.update(self.buffer, total_steps)
            self.buffer.reset()

            # ============================================================
            # Self-play snapshot & opponent swap
            # ============================================================
            if total_steps % tcfg.selfplay_update_interval < tcfg.num_envs:
                self.self_play.snapshot(self.agents, total_steps)
                opp_att, opp_def = self.self_play.sample_opponent(team_id=1)
                self.agents[2] = opp_att
                self.agents[3] = opp_def

            # ============================================================
            # Logging
            # ============================================================
            if total_steps % tcfg.log_interval < tcfg.num_envs:
                self.logger.log_scalars(losses, total_steps)
                summary = self.agg.summarize()
                if summary:
                    self.logger.log_episode(summary, total_steps)
                self.agg.reset()

            if total_steps % tcfg.print_interval < tcfg.num_envs:
                elapsed = time.time() - t_start
                sps = total_steps / max(1, elapsed)
                print(
                    f"[{total_steps:>10,}] "
                    f"loss={losses.get('total_loss', 0):.4f} | "
                    f"pol={losses.get('policy_loss', 0):.4f} | "
                    f"val={losses.get('value_loss', 0):.4f} | "
                    f"ent={losses.get('entropy_loss', 0):.4f} | "
                    f"kl={losses.get('approx_kl', 0):.4f} | "
                    f"sps={sps:.0f}"
                )

            # ============================================================
            # Checkpoint
            # ============================================================
            if total_steps % tcfg.checkpoint_interval < tcfg.num_envs:
                self.save_checkpoint(total_steps)

        print(f"[Trainer] Training complete. Total steps: {total_steps:,}")
        self.logger.close()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, timestep: int) -> None:
        """Save all agent state dicts and config to disk.

        Args:
            timestep: Current training timestep (used in filename).
        """
        from ctf_pacman.utils.config import save_config
        path = os.path.join(self.ckpt_dir, f"ckpt_{timestep}.pt")
        payload = {
            "timestep": timestep,
            "agent_state_dicts": {
                aid: agent.state_dict()
                for aid, agent in self.agents.items()
                if hasattr(agent, "state_dict")
            },
        }
        torch.save(payload, path)
        save_config(self.config, os.path.join(self.ckpt_dir, "config.yaml"))
        print(f"[Trainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent state dicts from a checkpoint file.

        Args:
            path: Path to a .pt checkpoint file.
        """
        payload = torch.load(path, map_location=self.device)
        sd = payload.get("agent_state_dicts", {})
        for aid, state_dict in sd.items():
            if aid in self.agents and hasattr(self.agents[aid], "load_state_dict"):
                self.agents[aid].load_state_dict(state_dict)
        print(f"[Trainer] Checkpoint loaded from: {path}")
