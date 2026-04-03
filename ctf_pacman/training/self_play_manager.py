"""Self-play manager with league-based opponent pool sampling."""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Tuple

import torch

from ctf_pacman.agents.attacker_agent import AttackerAgent
from ctf_pacman.agents.defender_agent import DefenderAgent
from ctf_pacman.agents.rule_based_agent import RuleBasedAgent
from ctf_pacman.utils.config import TrainingConfig


class SelfPlayManager:
    """Maintains a league of agent checkpoints for diverse opponent sampling.

    Opponent sampling fractions (from config):
      - ``latest_opponent_fraction``:    most recent checkpoint
      - ``historical_opponent_fraction``: random historical checkpoint
      - ``rule_based_opponent_fraction``: deterministic heuristic bot

    Args:
        config:                 TrainingConfig.
        agent_constructor_fn:   Callable(team_id) -> (AttackerAgent, DefenderAgent).
    """

    def __init__(
        self,
        config: TrainingConfig,
        agent_constructor_fn: Callable[[int], Tuple[AttackerAgent, DefenderAgent]],
    ) -> None:
        self.config = config
        self._constructor = agent_constructor_fn
        self.league: List[dict] = []   # [{"timestep": int, "state_dicts": {...}}]
        self.rule_based_attacker = RuleBasedAgent(role="attacker")
        self.rule_based_defender = RuleBasedAgent(role="defender")

    # ------------------------------------------------------------------
    # Snapshot current agents into the league
    # ------------------------------------------------------------------

    def snapshot(self, agents: Dict[int, object], timestep: int) -> None:
        """Deep-copy current agent weights into the league pool.

        If the pool exceeds ``league_size``, the oldest entry is removed.

        Args:
            agents:    Dict mapping agent_id -> BaseAgent.
            timestep:  Current training timestep (used as label).
        """
        state_dicts = {
            aid: copy.deepcopy(agent.state_dict())
            for aid, agent in agents.items()
            if hasattr(agent, "state_dict")
        }
        self.league.append({"timestep": timestep, "state_dicts": state_dicts})
        if len(self.league) > self.config.league_size:
            self.league.pop(0)  # remove oldest

    # ------------------------------------------------------------------
    # Sample an opponent pair
    # ------------------------------------------------------------------

    def sample_opponent(
        self, team_id: int
    ) -> Tuple[object, object]:
        """Sample an opponent attacker and defender for the given team.

        Returns a pair of agents (attacker, defender). The agents may be:
          - Fresh agents loaded from the latest checkpoint.
          - Fresh agents loaded from a random historical checkpoint.
          - The shared RuleBasedAgent instances.

        Args:
            team_id: The OPPONENT team's ID (agents will use this team's slot).

        Returns:
            Tuple of (attacker_agent, defender_agent).
        """
        cfg = self.config
        r = random.random()

        # Determine which checkpoint category to use
        if r < cfg.rule_based_opponent_fraction or len(self.league) == 0:
            return self.rule_based_attacker, self.rule_based_defender

        adj_r = r - cfg.rule_based_opponent_fraction
        latest_frac = cfg.latest_opponent_fraction
        hist_frac = cfg.historical_opponent_fraction
        total = latest_frac + hist_frac

        if adj_r < latest_frac / total or len(self.league) == 1:
            checkpoint = self.league[-1]
        else:
            # Random historical (exclude latest)
            checkpoint = random.choice(self.league[:-1])

        # Build fresh agent pair and load weights
        attacker, defender = self._constructor(team_id)
        attacker_id = 2 if team_id == 1 else 0
        defender_id = 3 if team_id == 1 else 1

        sd = checkpoint["state_dicts"]
        if attacker_id in sd:
            attacker.load_state_dict(sd[attacker_id])
        if defender_id in sd:
            defender.load_state_dict(sd[defender_id])

        attacker.eval()
        defender.eval()
        return attacker, defender
