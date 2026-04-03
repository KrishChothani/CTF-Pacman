"""Heuristic rule-based agent used as an opponent during self-play."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class RuleBasedAgent:
    """Deterministic heuristic bot with attacker and defender modes.

    Does NOT use any neural network. Implements BFS-based path planning for
    both roles. Used as the ``rule_based_opponent_fraction`` of the opponent
    pool during self-play training.

    Args:
        role: "attacker" or "defender".
    """

    def __init__(self, role: str = "attacker") -> None:
        self.role = role
        self._patrol_step = 0   # defender patrol counter

    # ------------------------------------------------------------------
    # Public API (matches the signature expected by SelfPlayManager)
    # ------------------------------------------------------------------

    def act(self, state: dict, agent_id: int) -> int:
        """Select an action using the heuristic policy.

        Args:
            state:    Full environment state dict (see CTFPacmanEnv).
            agent_id: The agent this bot controls.

        Returns:
            Integer action index (0=North, 1=South, 2=East, 3=West, 4=Stop).
        """
        role = state["agent_roles"].get(agent_id, self.role)
        if role == "attacker":
            return self._attacker_policy(state, agent_id)
        else:
            return self._defender_policy(state, agent_id)

    # ------------------------------------------------------------------
    # Attacker heuristic
    # ------------------------------------------------------------------

    def _attacker_policy(self, state: dict, agent_id: int) -> int:
        from ctf_pacman.environment.grid import Grid
        positions: Dict[int, Tuple[int, int]] = state["agent_positions"]
        teams: Dict[int, int] = state["agent_teams"]
        carrying: Dict[int, int] = state["agent_carrying"]
        scared: Dict[int, int] = state["agent_scared"]
        food: Set[Tuple[int, int]] = state.get("food_positions", set())
        pellets: Set[Tuple[int, int]] = state.get("power_pellet_positions", set())
        walls_array = state.get("walls")   # np.ndarray or None
        grid: Optional[Grid] = state.get("grid")

        ax, ay = positions[agent_id]
        my_team = teams[agent_id]
        opp_team = 1 - my_team
        mid = state.get("width", 32) // 2

        in_opp_territory = (ax >= mid if my_team == 0 else ax < mid)

        # Identify defenders (opponents with defender role)
        opp_defenders = [
            a for a, t in teams.items()
            if t == opp_team and state["agent_roles"].get(a) == "defender"
        ]

        # Identify opp food (in opponent's half)
        opp_food = {
            pos for pos in food
            if (pos[0] >= mid and my_team == 0) or (pos[0] < mid and my_team == 1)
        }

        # ----------------------------------------------------------------
        # Rule 1: if carrying food and a defender is nearby, return home
        # ----------------------------------------------------------------
        if carrying.get(agent_id, 0) > 0:
            for def_id in opp_defenders:
                dx, dy = positions[def_id]
                if abs(dx - ax) + abs(dy - ay) <= 4:
                    # Rush home
                    if grid is not None:
                        home_target = (mid - 1 if my_team == 0 else mid + 1), ay
                        return self._bfs_next_action((ax, ay), home_target, grid)
                    return 3 if my_team == 0 else 2  # West or East

        # ----------------------------------------------------------------
        # Rule 2: power pellet priority if visible and not carrying food
        # ----------------------------------------------------------------
        if carrying.get(agent_id, 0) == 0 and pellets:
            nearest_pellet = min(
                pellets, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay)
            )
            if grid is not None:
                return self._bfs_next_action((ax, ay), nearest_pellet, grid)

        # ----------------------------------------------------------------
        # Rule 3: if in opponent territory, BFS toward nearest food
        # ----------------------------------------------------------------
        if in_opp_territory and opp_food:
            nearest = min(opp_food, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
            if grid is not None:
                return self._bfs_next_action((ax, ay), nearest, grid)

        # ----------------------------------------------------------------
        # Rule 4: move toward the divider gap (midline)
        # ----------------------------------------------------------------
        h = state.get("height", 16)
        gap_rows = [h // 4, h // 2, 3 * h // 4]
        closest_gap = min(gap_rows, key=lambda r: abs(r - ay))
        target = (mid, closest_gap)
        if grid is not None:
            return self._bfs_next_action((ax, ay), target, grid)

        # Fallback: move East (team 0) or West (team 1)
        return 2 if my_team == 0 else 3

    # ------------------------------------------------------------------
    # Defender heuristic
    # ------------------------------------------------------------------

    def _defender_policy(self, state: dict, agent_id: int) -> int:
        from ctf_pacman.environment.grid import Grid
        positions: Dict[int, Tuple[int, int]] = state["agent_positions"]
        teams: Dict[int, int] = state["agent_teams"]
        scared: Dict[int, int] = state["agent_scared"]
        food: Set[Tuple[int, int]] = state.get("food_positions", set())
        grid: Optional[Grid] = state.get("grid")

        ax, ay = positions[agent_id]
        my_team = teams[agent_id]
        opp_team = 1 - my_team
        mid = state.get("width", 32) // 2

        # Friendly food (in our half)
        friendly_food = {
            pos for pos in food
            if (pos[0] < mid and my_team == 0) or (pos[0] >= mid and my_team == 1)
        }

        # Identify opponent attackers
        opp_attackers = [
            a for a, t in teams.items()
            if t == opp_team and state["agent_roles"].get(a) == "attacker"
        ]

        # Invaders: opponent attackers in our home territory
        invaders = [
            a for a in opp_attackers
            if (positions[a][0] < mid and my_team == 0)
            or (positions[a][0] >= mid and my_team == 1)
        ]

        # ----------------------------------------------------------------
        # Rule 1: if scared, flee from nearest opponent
        # ----------------------------------------------------------------
        if scared.get(agent_id, 0) > 0:
            if opp_attackers and grid is not None:
                nearest_opp = min(
                    opp_attackers,
                    key=lambda a: abs(positions[a][0] - ax) + abs(positions[a][1] - ay),
                )
                ox, oy = positions[nearest_opp]
                # Find action that maximises distance
                best_action = 4
                best_dist = -1
                for action in grid.get_legal_actions(ax, ay):
                    nx, ny = grid.apply_action(ax, ay, action)
                    d = abs(nx - ox) + abs(ny - oy)
                    if d > best_dist:
                        best_dist = d
                        best_action = action
                return best_action
            return 4  # Stop if no opponents

        # ----------------------------------------------------------------
        # Rule 2: if invader visible, chase with BFS
        # ----------------------------------------------------------------
        if invaders and grid is not None:
            target_invader = min(
                invaders,
                key=lambda a: abs(positions[a][0] - ax) + abs(positions[a][1] - ay),
            )
            tx, ty = positions[target_invader]
            return self._bfs_next_action((ax, ay), (tx, ty), grid)

        # ----------------------------------------------------------------
        # Rule 3: patrol between top food clusters
        # ----------------------------------------------------------------
        if friendly_food and grid is not None:
            sorted_food = sorted(
                friendly_food, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay)
            )
            patrol_targets = sorted_food[:min(2, len(sorted_food))]
            target = patrol_targets[(self._patrol_step // 20) % len(patrol_targets)]
            self._patrol_step += 1
            return self._bfs_next_action((ax, ay), target, grid)

        return 4  # Stop

    # ------------------------------------------------------------------
    # BFS shortest path
    # ------------------------------------------------------------------

    def _bfs_next_action(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid,
    ) -> int:
        """Return the first action along the BFS shortest path from start to goal.

        Args:
            start: (x, y) starting cell.
            goal:  (x, y) target cell.
            grid:  Grid instance with is_wall() and get_legal_actions().

        Returns:
            Action index. Returns 4 (Stop) if goal is unreachable.
        """
        if start == goal:
            return 4

        _ACTION_DELTAS = {
            0: (0, -1),
            1: (0, 1),
            2: (1, 0),
            3: (-1, 0),
            4: (0, 0),
        }

        queue: deque = deque()
        queue.append((start, None))   # (position, first_action_taken)
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            (cx, cy), first_action = queue.popleft()
            for action in grid.get_legal_actions(cx, cy):
                if action == 4:
                    continue  # skip Stop in path search
                dx, dy = _ACTION_DELTAS[action]
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                fa = first_action if first_action is not None else action
                if (nx, ny) == goal:
                    return fa
                visited.add((nx, ny))
                queue.append(((nx, ny), fa))

        return 4  # goal unreachable
