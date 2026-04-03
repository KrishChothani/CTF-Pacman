"""Observation builder: converts raw state dict into neural-network inputs."""

from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np

from ctf_pacman.environment.grid import Grid
from ctf_pacman.utils.config import EnvConfig

# Channel index constants (documentation only)
_CH_WALLS = 0
_CH_FRIENDLY_FOOD = 1
_CH_OPPONENT_FOOD = 2
_CH_SELF = 3
_CH_TEAMMATE = 4
_CH_OPP1 = 5
_CH_OPP2 = 6
_CH_POWER_PELLETS = 7
_CH_CARRYING = 8
_CH_TERRITORY = 9

# Number of flat features returned
FLAT_FEATURE_DIM = 8


class ObservationBuilder:
    """Builds (grid_obs, flat_obs) tuples from raw game state.

    Grid observation channels (10 total):

    0. Walls — binary local view padded with 1.0 at borders.
    1. Friendly food — food belonging to this agent's team.
    2. Opponent food — food belonging to the opposing team.
    3. Self position — 1.0 at the agent's exact center cell.
    4. Teammate position — 1.0 if teammate is in the local window.
    5. Opponent 1 — exact if visible, Gaussian smear from last known position.
    6. Opponent 2 — same as channel 5 for the second opponent.
    7. Power pellets — 1.0 at power pellet cells.
    8. Carrying indicator — uniform fill equal to carry_count / num_food_per_team.
    9. Territory — 1.0 for home territory cells.

    Flat features (8 values):

    0. Normalized step.
    1. Score differential (own - opp) / num_food_per_team, clipped to [-1, 1].
    2. Teammate scared steps remaining (normalized).
    3. Self scared steps remaining (normalized).
    4. Distance to nearest friendly food (normalized).
    5. Distance to home boundary (normalized).
    6. Food carrying count (normalized).
    7. Opponent 1 scared indicator (binary).

    Args:
        config: Environment configuration.
        grid:   The current map grid.
    """

    def __init__(self, config: EnvConfig, grid: Grid) -> None:
        self.config = config
        self.grid = grid
        self.r = config.observation_radius
        self.window = 2 * self.r + 1
        self.C = config.num_observation_channels  # should be 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        state: dict,
        agent_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build observation arrays for the given agent.

        Args:
            state:    Full game state dictionary.
            agent_id: The agent for whom we build the observation.

        Returns:
            Tuple of (grid_obs, flat_obs):
              - grid_obs: float32 array of shape (C, window, window).
              - flat_obs: float32 array of shape (FLAT_FEATURE_DIM,).
        """
        config = self.config
        r = self.r
        w = self.window

        agent_positions: Dict[int, Tuple[int, int]] = state["agent_positions"]
        agent_teams: Dict[int, int] = state["agent_teams"]
        agent_roles: Dict[int, str] = state["agent_roles"]
        agent_carrying: Dict[int, int] = state["agent_carrying"]
        agent_scared: Dict[int, int] = state["agent_scared"]
        food_positions: Set[Tuple[int, int]] = state["food_positions"]
        power_pellet_positions: Set[Tuple[int, int]] = state["power_pellet_positions"]
        scores: Dict[int, int] = state["scores"]
        step: int = state["step"]
        last_known: Dict[int, Tuple[int, int, int]] = state.get("last_known_positions", {})

        my_team = agent_teams[agent_id]
        opp_team = 1 - my_team
        ax, ay = agent_positions[agent_id]

        # Identify teammate and opponents
        my_team_agents = [a for a, t in agent_teams.items() if t == my_team and a != agent_id]
        opp_agents = [a for a, t in agent_teams.items() if t == opp_team]

        teammate_id = my_team_agents[0] if my_team_agents else None
        opp1_id = opp_agents[0] if len(opp_agents) > 0 else None
        opp2_id = opp_agents[1] if len(opp_agents) > 1 else None

        # Identify which food belongs to which team
        # Team 0 food: x < mid. Team 1 food: x >= mid.
        mid = self.grid.width // 2
        friendly_food = {
            pos for pos in food_positions
            if (pos[0] < mid and my_team == 0) or (pos[0] >= mid and my_team == 1)
        }
        opponent_food = food_positions - friendly_food

        # ----------------------------------------------------------------
        # Build grid channels
        # ----------------------------------------------------------------
        grid_obs = np.zeros((self.C, w, w), dtype=np.float32)

        for gy in range(w):
            for gx in range(w):
                world_x = ax - r + gx
                world_y = ay - r + gy

                # Ch 0: Walls (1.0 if wall or OOB)
                if self.grid.is_wall(world_x, world_y):
                    grid_obs[_CH_WALLS, gy, gx] = 1.0

                # Ch 1: Friendly food
                if (world_x, world_y) in friendly_food:
                    grid_obs[_CH_FRIENDLY_FOOD, gy, gx] = 1.0

                # Ch 2: Opponent food
                if (world_x, world_y) in opponent_food:
                    grid_obs[_CH_OPPONENT_FOOD, gy, gx] = 1.0

                # Ch 7: Power pellets
                if (world_x, world_y) in power_pellet_positions:
                    grid_obs[_CH_POWER_PELLETS, gy, gx] = 1.0

                # Ch 9: Territory (home = 1.0)
                if 0 <= world_x < self.grid.width:
                    if self.grid.is_home_territory(world_x, my_team):
                        grid_obs[_CH_TERRITORY, gy, gx] = 1.0

        # Ch 3: Self at center
        grid_obs[_CH_SELF, r, r] = 1.0

        # Ch 4: Teammate
        if teammate_id is not None:
            tx, ty = agent_positions[teammate_id]
            tgx = tx - (ax - r)
            tgy = ty - (ay - r)
            if 0 <= tgx < w and 0 <= tgy < w:
                grid_obs[_CH_TEAMMATE, tgy, tgx] = 1.0

        # Ch 5 & 6: Opponents with Gaussian smear for unseen agents
        for ch_idx, opp_id in [(_CH_OPP1, opp1_id), (_CH_OPP2, opp2_id)]:
            if opp_id is None:
                continue
            ox, oy = agent_positions[opp_id]
            ogx = ox - (ax - r)
            ogy = oy - (ay - r)

            if 0 <= ogx < w and 0 <= ogy < w:
                # Opponent is directly visible in window
                grid_obs[ch_idx, ogy, ogx] = 1.0
            else:
                # Gaussian smear from last known position
                lk = last_known.get(opp_id)
                if lk is not None:
                    lx, ly, step_seen = lk
                    steps_since = max(0, step - step_seen)
                    sigma = np.sqrt(steps_since + 1)
                    # Place a 2D Gaussian centered at last known world position
                    for gy in range(w):
                        for gx in range(w):
                            world_x = ax - r + gx
                            world_y = ay - r + gy
                            dist2 = (world_x - lx) ** 2 + (world_y - ly) ** 2
                            val = np.exp(-dist2 / (2.0 * sigma ** 2))
                            grid_obs[ch_idx, gy, gx] = min(1.0, grid_obs[ch_idx, gy, gx] + val)

        # Ch 8: Carrying indicator (uniform fill)
        carry_norm = agent_carrying[agent_id] / max(1, config.num_food_per_team)
        grid_obs[_CH_CARRYING, :, :] = carry_norm

        # ----------------------------------------------------------------
        # Build flat features
        # ----------------------------------------------------------------
        max_dim = max(self.grid.width, self.grid.height)
        own_score = scores[my_team]
        opp_score = scores[opp_team]

        # Feature 4: distance to nearest friendly food
        if friendly_food:
            dist_friendly = min(
                abs(ax - fx) + abs(ay - fy) for fx, fy in friendly_food
            )
            dist_friendly_norm = dist_friendly / max_dim
        else:
            dist_friendly_norm = 1.0

        # Feature 5: distance to midline
        dist_mid = abs(ax - mid)
        dist_mid_norm = dist_mid / (self.grid.width / 2.0)

        # Feature 7: opp1 scared indicator
        opp1_scared = 1.0 if (opp1_id is not None and agent_scared.get(opp1_id, 0) > 0) else 0.0

        my_scared = agent_scared.get(agent_id, 0)
        tm_scared = agent_scared.get(teammate_id, 0) if teammate_id is not None else 0

        flat_obs = np.array([
            step / max(1, config.max_steps),                                           # 0
            np.clip((own_score - opp_score) / max(1, config.num_food_per_team), -1, 1),  # 1
            tm_scared / max(1, config.power_pellet_duration),                          # 2
            my_scared / max(1, config.power_pellet_duration),                          # 3
            dist_friendly_norm,                                                         # 4
            dist_mid_norm,                                                              # 5
            agent_carrying[agent_id] / max(1, config.num_food_per_team),              # 6
            opp1_scared,                                                                # 7
        ], dtype=np.float32)

        # Clip all flat features to [-1, 1] for numerical stability (Part 11)
        flat_obs = np.clip(flat_obs, -1.0, 1.0)

        return grid_obs, flat_obs
