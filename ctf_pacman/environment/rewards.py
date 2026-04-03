"""Reward calculation with named constants for all reward signals."""

from __future__ import annotations

from ctf_pacman.environment.events import (
    AgentCapturedEvent,
    EpisodeEndEvent,
    EventLog,
    FoodCollectedEvent,
    FoodReturnedEvent,
    PowerPelletConsumedEvent,
)
from ctf_pacman.utils.config import EnvConfig

# ---------------------------------------------------------------------------
# Named reward constants (never embed magic numbers inline)
# ---------------------------------------------------------------------------

REWARD_FOOD_COLLECTED: float = 1.0
REWARD_FOOD_RETURNED: float = 10.0
PENALTY_CAPTURED: float = -5.0
REWARD_DEFENDER_CAPTURE: float = 5.0
REWARD_DEFENDER_INVADER_PRESENT: float = 0.5  # per timestep invader is in home
PENALTY_FOOD_SCORED_BY_OPPONENT: float = -2.0
PENALTY_STEP: float = -0.05
REWARD_WIN: float = 15.0
PENALTY_LOSS: float = -15.0
REWARD_EPISODE_MULTIPLIER: float = 0.3


class RewardCalculator:
    """Computes shaped reward signals for each agent given the event log.

    Rewards are role-sensitive: defenders receive different signals than
    attackers for the same events.

    Args:
        config: Environment configuration (used for team sizes, etc.).
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def compute(
        self,
        events: EventLog,
        agent_id: int,
        agent_role: str,
        agent_team: int,
        invader_present: bool = False,
    ) -> float:
        """Sum all reward signals relevant to *agent_id* from *events*.

        Args:
            events:          The event log for the current step.
            agent_id:        The agent whose reward we are computing.
            agent_role:      "attacker" or "defender".
            agent_team:      0 or 1.
            invader_present: True if an opponent attacker is currently inside
                              this agent's home territory (defender bonus).

        Returns:
            Scalar reward for agent_id at this step.
        """
        reward: float = PENALTY_STEP  # constant step penalty for all agents

        for event in events.events:
            # ----------------------------------------------------------------
            # Food collected
            # ----------------------------------------------------------------
            if isinstance(event, FoodCollectedEvent):
                if event.agent_id == agent_id:
                    reward += REWARD_FOOD_COLLECTED

            # ----------------------------------------------------------------
            # Food returned home
            # ----------------------------------------------------------------
            elif isinstance(event, FoodReturnedEvent):
                if event.agent_id == agent_id:
                    reward += REWARD_FOOD_RETURNED
                elif agent_role == "defender":
                    # Opponent scored — penalty for the defending team
                    reward += PENALTY_FOOD_SCORED_BY_OPPONENT

            # ----------------------------------------------------------------
            # Agent captured
            # ----------------------------------------------------------------
            elif isinstance(event, AgentCapturedEvent):
                if event.captured_id == agent_id:
                    reward += PENALTY_CAPTURED
                if event.capturing_id == agent_id and agent_role == "defender":
                    reward += REWARD_DEFENDER_CAPTURE

            # ----------------------------------------------------------------
            # Episode end
            # ----------------------------------------------------------------
            elif isinstance(event, EpisodeEndEvent):
                reward += self.compute_team_bonus(event, team_id=agent_team)

        # Defender bonus: each step where an invader is present but not yet
        # caught rewards the defender for staying relevant in home territory.
        if agent_role == "defender" and invader_present:
            reward += REWARD_DEFENDER_INVADER_PRESENT

        return reward

    def compute_team_bonus(self, event: EpisodeEndEvent, team_id: int) -> float:
        """Compute the terminal reward/penalty from an EpisodeEndEvent.

        Args:
            event:   The episode termination event.
            team_id: The team whose perspective we compute the bonus for.

        Returns:
            +REWARD_WIN * MULTIPLIER, -REWARD_WIN * MULTIPLIER, or 0.0.
        """
        if event.winner == team_id:
            return REWARD_WIN * REWARD_EPISODE_MULTIPLIER
        elif event.winner == -1:  # draw
            return 0.0
        else:
            return PENALTY_LOSS * REWARD_EPISODE_MULTIPLIER
