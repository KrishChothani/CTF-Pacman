"""Game events emitted by the environment during a step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, TypeVar

T = TypeVar("T", bound="GameEvent")


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    """Abstract base for all game events."""
    pass


# ---------------------------------------------------------------------------
# Concrete events
# ---------------------------------------------------------------------------

@dataclass
class FoodCollectedEvent(GameEvent):
    """An agent collected a food pellet from the opponent's side.

    Attributes:
        agent_id:         ID of the collecting agent.
        x:                Grid x-coordinate of the collected food.
        y:                Grid y-coordinate of the collected food.
        food_count_carried: Total food this agent now carries.
    """
    agent_id: int = 0
    x: int = 0
    y: int = 0
    food_count_carried: int = 0


@dataclass
class FoodReturnedEvent(GameEvent):
    """An attacker successfully returned carried food to home territory.

    Attributes:
        agent_id:    ID of the returning agent.
        score_delta: Points added to the team's score.
        food_count:  Number of food items returned in this event.
    """
    agent_id: int = 0
    score_delta: int = 0
    food_count: int = 0


@dataclass
class AgentCapturedEvent(GameEvent):
    """An agent was captured by an opponent.

    Attributes:
        captured_id:   ID of the agent that was captured.
        capturing_id:  ID of the agent that performed the capture.
        food_lost:     Food items dropped by the captured agent.
    """
    captured_id: int = 0
    capturing_id: int = 0
    food_lost: int = 0


@dataclass
class PowerPelletConsumedEvent(GameEvent):
    """An agent consumed a power pellet, scaring the opposing team.

    Attributes:
        agent_id: ID of the agent that consumed the pellet.
        duration: Steps remaining that opponents will be scared.
    """
    agent_id: int = 0
    duration: int = 0


@dataclass
class EpisodeEndEvent(GameEvent):
    """The episode terminated (all food taken or step limit reached).

    Attributes:
        winner:       0 = team 0 wins, 1 = team 1 wins, -1 = draw.
        reason:       Human-readable termination reason.
        final_scores: Dict mapping team_id -> score at termination.
    """
    winner: int = -1
    reason: str = "unknown"
    final_scores: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class EventLog:
    """Ordered, queryable log of game events for a single episode.

    Typical usage::

        log = EventLog()
        log.add(FoodCollectedEvent(agent_id=0, x=10, y=5, food_count_carried=1))
        food_events = log.get_by_type(FoodCollectedEvent)
    """

    def __init__(self) -> None:
        self.events: List[GameEvent] = []

    def add(self, event: GameEvent) -> None:
        """Append an event to the log.

        Args:
            event: Any GameEvent subclass instance.
        """
        self.events.append(event)

    def get_by_type(self, event_type: Type[T]) -> List[T]:
        """Return all events of a specific type.

        Args:
            event_type: The GameEvent subclass to filter by.

        Returns:
            List of matching events in insertion order.
        """
        return [e for e in self.events if isinstance(e, event_type)]

    def clear(self) -> None:
        """Remove all events from the log."""
        self.events.clear()

    def __len__(self) -> int:
        return len(self.events)
