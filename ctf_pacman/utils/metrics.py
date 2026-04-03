"""Episode metrics tracking and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EpisodeMetrics:
    """Metrics recorded for a single episode.

    Attributes:
        episode_return:     Per-agent total undiscounted reward.
        episode_length:     Number of environment steps taken.
        food_collected:     Food pellets picked up per agent.
        food_returned:      Food pellets successfully returned home per agent.
        captures_made:      Captures initiated per agent.
        captures_suffered:  Times an agent was captured.
        win:                -1 = loss, 0 = draw, 1 = win for Team 0.
        score_team0:        Final food score for team 0.
        score_team1:        Final food score for team 1.
    """
    episode_return: Dict[int, float] = field(default_factory=lambda: {i: 0.0 for i in range(4)})
    episode_length: int = 0
    food_collected: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    food_returned: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    captures_made: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    captures_suffered: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    win: int = 0          # -1 team1 wins, 0 draw, 1 team0 wins
    score_team0: int = 0
    score_team1: int = 0


class MetricsAggregator:
    """Accumulates EpisodeMetrics over multiple episodes and computes means.

    Usage::

        agg = MetricsAggregator()
        for ep_metrics in run_episodes():
            agg.add(ep_metrics)
        summary = agg.summarize()
        agg.reset()
    """

    def __init__(self) -> None:
        self._episodes: List[EpisodeMetrics] = []

    def add(self, metrics: EpisodeMetrics) -> None:
        """Add one episode's metrics to the accumulator.

        Args:
            metrics: Completed episode stats.
        """
        self._episodes.append(metrics)

    def summarize(self) -> dict:
        """Compute mean values over all accumulated episodes.

        Returns:
            Dict mapping metric name to mean float value.
        """
        if not self._episodes:
            return {}

        n = len(self._episodes)
        agent_ids = list(range(4))

        # Per-agent aggregation
        mean_return = {
            a: sum(e.episode_return.get(a, 0.0) for e in self._episodes) / n
            for a in agent_ids
        }
        mean_food_col = {
            a: sum(e.food_collected.get(a, 0) for e in self._episodes) / n
            for a in agent_ids
        }
        mean_food_ret = {
            a: sum(e.food_returned.get(a, 0) for e in self._episodes) / n
            for a in agent_ids
        }
        mean_caps_made = {
            a: sum(e.captures_made.get(a, 0) for e in self._episodes) / n
            for a in agent_ids
        }
        mean_caps_suf = {
            a: sum(e.captures_suffered.get(a, 0) for e in self._episodes) / n
            for a in agent_ids
        }

        # Team-level stats
        team0_wins = sum(1 for e in self._episodes if e.win == 1)
        team1_wins = sum(1 for e in self._episodes if e.win == -1)
        draws = sum(1 for e in self._episodes if e.win == 0)

        summary: dict = {
            "num_episodes": float(n),
            "mean_episode_length": sum(e.episode_length for e in self._episodes) / n,
            "mean_score_team0": sum(e.score_team0 for e in self._episodes) / n,
            "mean_score_team1": sum(e.score_team1 for e in self._episodes) / n,
            "win_rate_team0": team0_wins / n,
            "win_rate_team1": team1_wins / n,
            "draw_rate": draws / n,
        }
        for a in agent_ids:
            summary[f"agent{a}/mean_return"] = mean_return[a]
            summary[f"agent{a}/mean_food_collected"] = mean_food_col[a]
            summary[f"agent{a}/mean_food_returned"] = mean_food_ret[a]
            summary[f"agent{a}/mean_captures_made"] = mean_caps_made[a]
            summary[f"agent{a}/mean_captures_suffered"] = mean_caps_suf[a]

        return summary

    def reset(self) -> None:
        """Clear all accumulated episodes."""
        self._episodes.clear()
