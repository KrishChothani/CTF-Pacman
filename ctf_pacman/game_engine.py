"""Top-level game engine used at inference/evaluation time."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from ctf_pacman.environment.env import CTFPacmanEnv
from ctf_pacman.environment.events import (
    AgentCapturedEvent,
    EpisodeEndEvent,
    FoodCollectedEvent,
    FoodReturnedEvent,
)
from ctf_pacman.utils.config import Config
from ctf_pacman.utils.metrics import EpisodeMetrics

_NUM_AGENTS = 4
_AGENT_TEAMS = {0: 0, 1: 0, 2: 1, 3: 1}
_AGENT_ROLES = {0: "attacker", 1: "defender", 2: "attacker", 3: "defender"}


class GameEngine:
    """Coordinates a full episode at inference or evaluation time.

    Handles the full agent-environment interaction loop including:
      - Dispatching observations to each agent.
      - Exchanging teammate messages.
      - Collecting actions (deterministic or stochastic).
      - Stepping the environment.
      - Accumulating episode metrics.
      - Optional ASCII rendering with a configurable frame delay.

    Args:
        env:    A pre-constructed CTFPacmanEnv instance.
        agents: Dict mapping agent_id -> agent (BaseAgent or RuleBasedAgent).
        config: Top-level Config dataclass.
    """

    def __init__(
        self,
        env: CTFPacmanEnv,
        agents: Dict[int, object],
        config: Config,
    ) -> None:
        self.env = env
        self.agents = agents
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(
        self,
        render: bool = False,
        deterministic: bool = True,
        frame_delay: float = 0.05,
    ) -> EpisodeMetrics:
        """Run one complete episode from reset to termination.

        Args:
            render:      If True, print ASCII render each step.
            deterministic: If True, all neural agents act greedily.
            frame_delay: Seconds to sleep between frames when rendering.

        Returns:
            EpisodeMetrics for the completed episode.
        """
        cfg = self.config
        obs_dict, _ = self.env.reset()

        # Initialise teammate message buffers (zeros)
        messages: Dict[int, torch.Tensor] = {
            aid: torch.zeros(cfg.agent.message_dim, device=self.device)
            for aid in range(_NUM_AGENTS)
        }

        # Per-episode accumulators
        ep_return: Dict[int, float] = {i: 0.0 for i in range(_NUM_AGENTS)}
        ep_food_col: Dict[int, int] = {i: 0 for i in range(_NUM_AGENTS)}
        ep_food_ret: Dict[int, int] = {i: 0 for i in range(_NUM_AGENTS)}
        ep_caps_made: Dict[int, int] = {i: 0 for i in range(_NUM_AGENTS)}
        ep_caps_suf: Dict[int, int] = {i: 0 for i in range(_NUM_AGENTS)}
        ep_length = 0
        final_scores = {0: 0, 1: 0}
        winner = 0

        while True:
            if render:
                print(self.env.render())
                time.sleep(frame_delay)

            # ----------------------------------------------------------------
            # Collect actions from all agents
            # ----------------------------------------------------------------
            actions: Dict[int, int] = {}
            new_messages: Dict[int, torch.Tensor] = {}

            for aid in range(_NUM_AGENTS):
                agent = self.agents[aid]
                obs = obs_dict[aid]
                legal_mask = self.env.get_legal_action_mask(aid)
                mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

                teammate_id = {0: 1, 1: 0, 2: 3, 3: 2}[aid]
                recv_msg = messages[teammate_id]

                if hasattr(agent, "forward"):
                    # Neural agent
                    grid_t = torch.tensor(
                        obs["grid"], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    flat_t = torch.tensor(
                        obs["flat"], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    agent.eval()
                    with torch.no_grad():
                        out = agent.forward(
                            grid_obs=grid_t,
                            flat_obs=flat_t,
                            received_message=recv_msg.unsqueeze(0),
                            global_state=None,
                            action_mask=mask_t.unsqueeze(0),
                        )
                    if deterministic:
                        action = int(out["action_logits"].argmax(dim=-1).item())
                    else:
                        action = int(out["action_dist"].sample().item())

                    new_messages[aid] = out["message"].squeeze(0).detach()
                else:
                    # Rule-based agent: augment state with required fields
                    enriched_state = dict(self.env._state)
                    enriched_state["grid"] = self.env._grid
                    enriched_state["width"] = cfg.env.map_width
                    enriched_state["height"] = cfg.env.map_height
                    action = agent.act(enriched_state, aid)
                    new_messages[aid] = torch.zeros(cfg.agent.message_dim, device=self.device)

                actions[aid] = action

            # Update message buffers for next step
            messages = new_messages

            # ----------------------------------------------------------------
            # Step environment
            # ----------------------------------------------------------------
            obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
            ep_length += 1

            for aid in range(_NUM_AGENTS):
                ep_return[aid] += rewards.get(aid, 0.0)

            # Parse events for metrics (use agent 0's info which has all events)
            for evt in info.get(0, {}).get("events", []):
                if isinstance(evt, FoodCollectedEvent):
                    ep_food_col[evt.agent_id] += 1
                elif isinstance(evt, FoodReturnedEvent):
                    ep_food_ret[evt.agent_id] += evt.food_count
                elif isinstance(evt, AgentCapturedEvent):
                    ep_caps_made[evt.capturing_id] += 1
                    ep_caps_suf[evt.captured_id] += 1
                elif isinstance(evt, EpisodeEndEvent):
                    final_scores = dict(evt.final_scores)
                    winner = evt.winner  # 0=team0, 1=team1, -1=draw

            done = any(terminated.values()) or any(truncated.values())
            if done:
                if render:
                    print(self.env.render())
                break

        # Normalise winner to EpisodeMetrics convention:
        #   1 = team 0 wins, -1 = team 0 loses, 0 = draw
        win_flag = 0
        if winner == 0:
            win_flag = 1
        elif winner == 1:
            win_flag = -1

        return EpisodeMetrics(
            episode_return=ep_return,
            episode_length=ep_length,
            food_collected=ep_food_col,
            food_returned=ep_food_ret,
            captures_made=ep_caps_made,
            captures_suffered=ep_caps_suf,
            win=win_flag,
            score_team0=final_scores.get(0, 0),
            score_team1=final_scores.get(1, 0),
        )
