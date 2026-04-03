"""Core CTF-Pacman Gymnasium environment."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

_log = logging.getLogger(__name__)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ctf_pacman.environment.events import (
    AgentCapturedEvent,
    EpisodeEndEvent,
    EventLog,
    FoodCollectedEvent,
    FoodReturnedEvent,
    PowerPelletConsumedEvent,
)
from ctf_pacman.environment.grid import Grid
from ctf_pacman.environment.observations import ObservationBuilder
from ctf_pacman.environment.rewards import RewardCalculator
from ctf_pacman.utils.config import EnvConfig

# ---------------------------------------------------------------------------
# Agent ID → (team_id, role) mapping
# ---------------------------------------------------------------------------
# 0 → team 0, attacker
# 1 → team 0, defender
# 2 → team 1, attacker
# 3 → team 1, defender

_AGENT_TEAMS = {0: 0, 1: 0, 2: 1, 3: 1}
_AGENT_ROLES = {0: "attacker", 1: "defender", 2: "attacker", 3: "defender"}
_NUM_AGENTS = 4


class CTFPacmanEnv(gym.Env):
    """Capture-the-Flag Pacman multi-agent environment.

    Two teams (each with one attacker and one defender) compete to steal
    the opposing team's food while protecting their own. The environment
    follows the Gymnasium API and returns per-agent observation/reward dicts.

    Agent IDs:
        0 — Team 0 attacker
        1 — Team 0 defender
        2 — Team 1 attacker
        3 — Team 1 defender

    Args:
        config: Environment configuration dataclass.
        seed:   RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig, seed: int = 42) -> None:
        super().__init__()
        self.config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Grid (generated once; re-generated only when options["regenerate_grid"])
        self._grid = Grid(
            width=config.map_width,
            height=config.map_height,
            wall_density=config.wall_density,
            seed=seed,
        )
        self._obs_builder = ObservationBuilder(config, self._grid)
        self._reward_calc = RewardCalculator(config)

        # Define spaces per agent
        r = config.observation_radius
        window = 2 * r + 1
        C = config.num_observation_channels

        single_obs_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=1.0,
                shape=(C, window, window),
                dtype=np.float32,
            ),
            "flat": spaces.Box(
                low=-1.0, high=1.0,
                shape=(8,),
                dtype=np.float32,
            ),
        })
        self.observation_space = spaces.Dict(
            {agent_id: single_obs_space for agent_id in range(_NUM_AGENTS)}
        )
        self.action_space = spaces.Dict(
            {agent_id: spaces.Discrete(5) for agent_id in range(_NUM_AGENTS)}
        )

        # Internal state (initialised in reset())
        self._state: dict = {}
        self._step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Reset the environment to an initial state.

        Args:
            seed:    Optional seed override.
            options: Optional dict. If ``options["regenerate_grid"]`` is True,
                     a new procedural map is generated.

        Returns:
            (observations, info) dicts keyed by agent_id.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        if options.get("regenerate_grid", False):
            new_seed = int(self._rng.integers(0, 2**31))
            self._grid = Grid(
                width=self.config.map_width,
                height=self.config.map_height,
                wall_density=self.config.wall_density,
                seed=new_seed,
            )
            self._obs_builder = ObservationBuilder(self.config, self._grid)

        self._step_count = 0
        self._done = False
        self._event_log = EventLog()

        # Place food
        food_positions = self._place_food()
        power_pellet_positions = self._place_power_pellets()

        # Spawn agents
        agent_positions = self._spawn_agents()

        self._state = {
            "agent_positions": agent_positions,
            "agent_teams": dict(_AGENT_TEAMS),
            "agent_roles": dict(_AGENT_ROLES),
            "agent_carrying": {i: 0 for i in range(_NUM_AGENTS)},
            "agent_scared": {i: 0 for i in range(_NUM_AGENTS)},
            "food_positions": food_positions,
            "power_pellet_positions": power_pellet_positions,
            "scores": {0: 0, 1: 0},
            "step": 0,
            "last_known_positions": {},
            # Track initial food counts per team for termination detection
            "_team_food": {
                0: len([p for p in food_positions if p[0] < self.config.map_width // 2]),
                1: len([p for p in food_positions if p[0] >= self.config.map_width // 2]),
            },
        }

        observations = self._build_all_observations()
        info = self._build_info()
        return observations, info

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[dict, dict, dict, dict, dict]:
        """Advance the environment by one timestep.

        Args:
            actions: Dict mapping agent_id -> action index (0–4).

        Returns:
            (observations, rewards, terminated, truncated, info)
            Each is a dict keyed by agent_id, except terminated/truncated
            which are also keyed by agent_id but share the same boolean value.
        """
        self._event_log.clear()

        cfg = self.config
        state = self._state
        grid = self._grid

        # ----------------------------------------------------------------
        # 1. Validate actions (illegal → Stop)
        # ----------------------------------------------------------------
        legal_masks = {aid: self.get_legal_action_mask(aid) for aid in range(_NUM_AGENTS)}
        clean_actions: Dict[int, int] = {}
        for aid in range(_NUM_AGENTS):
            a = actions.get(aid, 4)
            if not legal_masks[aid][a]:
                a = 4  # default to Stop
            clean_actions[aid] = a

        # ----------------------------------------------------------------
        # 2. Compute new positions (simultaneous; collision → stay)
        # ----------------------------------------------------------------
        proposed: Dict[int, Tuple[int, int]] = {}
        for aid in range(_NUM_AGENTS):
            cx, cy = state["agent_positions"][aid]
            proposed[aid] = grid.apply_action(cx, cy, clean_actions[aid])

        # Detect same-cell collisions among agents of the same team
        new_positions: Dict[int, Tuple[int, int]] = {}
        proposal_counts: Dict[Tuple[int, int], List[int]] = {}
        for aid, pos in proposed.items():
            proposal_counts.setdefault(pos, []).append(aid)

        for aid in range(_NUM_AGENTS):
            pos = proposed[aid]
            # Collision: two same-team agents propose the same cell
            same_team_rivals = [
                a for a in proposal_counts[pos]
                if a != aid and state["agent_teams"][a] == state["agent_teams"][aid]
            ]
            if same_team_rivals:
                new_positions[aid] = state["agent_positions"][aid]  # stay
            else:
                new_positions[aid] = pos

        state["agent_positions"] = new_positions

        # Update last known positions (opponents see each other if in window)
        self._update_last_known()

        # ----------------------------------------------------------------
        # 3. Food collection
        # ----------------------------------------------------------------
        for aid in range(_NUM_AGENTS):
            ax, ay = new_positions[aid]
            team = state["agent_teams"][aid]
            role = state["agent_roles"][aid]
            if role != "attacker":
                continue
            # Attacker must be in opponent territory
            if grid.is_home_territory(ax, team):
                continue
            if (ax, ay) in state["food_positions"]:
                state["food_positions"].discard((ax, ay))
                state["agent_carrying"][aid] += 1
                evt = FoodCollectedEvent(
                    agent_id=aid,
                    x=ax, y=ay,
                    food_count_carried=state["agent_carrying"][aid],
                )
                self._event_log.add(evt)

        # ----------------------------------------------------------------
        # 4. Power pellet collection
        # ----------------------------------------------------------------
        for aid in range(_NUM_AGENTS):
            ax, ay = new_positions[aid]
            if (ax, ay) in state["power_pellet_positions"]:
                state["power_pellet_positions"].discard((ax, ay))
                opp_team = 1 - state["agent_teams"][aid]
                for opp_aid in range(_NUM_AGENTS):
                    if state["agent_teams"][opp_aid] == opp_team:
                        state["agent_scared"][opp_aid] = cfg.power_pellet_duration
                self._event_log.add(
                    PowerPelletConsumedEvent(agent_id=aid, duration=cfg.power_pellet_duration)
                )

        # ----------------------------------------------------------------
        # 5. Captures  (two-phase: collect then resolve conflicts)
        # If agent X would be both the capturer and the captured in the same
        # step, neither capture proceeds.
        # ----------------------------------------------------------------
        intended: List[dict] = []

        for aid in range(_NUM_AGENTS):
            ax, ay = new_positions[aid]
            team = state["agent_teams"][aid]
            role = state["agent_roles"][aid]
            if role != "defender":
                continue
            for opp_aid in range(_NUM_AGENTS):
                if state["agent_teams"][opp_aid] == team:
                    continue
                opp_x, opp_y = new_positions[opp_aid]
                if (ax, ay) != (opp_x, opp_y):
                    continue
                if state["agent_scared"][aid] > 0:
                    # Scared defender: attacker captures the defender
                    intended.append({
                        "captured_id": aid,
                        "capturing_id": opp_aid,
                        "food_lost": 0,
                        "reset_scared_id": aid,
                        "role": "scared_flip",
                    })
                elif grid.is_home_territory(opp_x, team):
                    # Normal: defender captures invading attacker
                    intended.append({
                        "captured_id": opp_aid,
                        "capturing_id": aid,
                        "food_lost": state["agent_carrying"][opp_aid],
                        "reset_scared_id": None,
                        "role": "normal",
                    })

        # Resolve conflicts: agent cannot be both capturer and captive
        conflicted = (
            {c["captured_id"] for c in intended}
            & {c["capturing_id"] for c in intended}
        )
        for cap in intended:
            if cap["captured_id"] in conflicted or cap["capturing_id"] in conflicted:
                _log.debug(
                    "Simultaneous capture conflict — skipping agents %s/%s",
                    cap["captured_id"], cap["capturing_id"],
                )
                continue
            cap_id = cap["captured_id"]
            cap_by = cap["capturing_id"]
            state["agent_carrying"][cap_id] = 0
            state["agent_positions"][cap_id] = self._spawn_single(
                team=state["agent_teams"][cap_id],
                role=state["agent_roles"][cap_id],
            )
            if cap["reset_scared_id"] is not None:
                state["agent_scared"][cap["reset_scared_id"]] = 0
            self._event_log.add(
                AgentCapturedEvent(
                    captured_id=cap_id,
                    capturing_id=cap_by,
                    food_lost=cap["food_lost"],
                )
            )

        # ----------------------------------------------------------------
        # 6. Food return
        # ----------------------------------------------------------------
        for aid in range(_NUM_AGENTS):
            ax, ay = new_positions[aid]
            team = state["agent_teams"][aid]
            role = state["agent_roles"][aid]
            if role != "attacker":
                continue
            if state["agent_carrying"][aid] > 0 and grid.is_home_territory(ax, team):
                count = state["agent_carrying"][aid]
                state["scores"][team] += count
                state["agent_carrying"][aid] = 0
                self._event_log.add(
                    FoodReturnedEvent(
                        agent_id=aid,
                        score_delta=count,
                        food_count=count,
                    )
                )

        # ----------------------------------------------------------------
        # 7. Decrement scared timers
        # ----------------------------------------------------------------
        for aid in range(_NUM_AGENTS):
            if state["agent_scared"][aid] > 0:
                state["agent_scared"][aid] -= 1

        # ----------------------------------------------------------------
        # 8. Increment step
        # ----------------------------------------------------------------
        self._step_count += 1
        state["step"] = self._step_count

        # ----------------------------------------------------------------
        # 9. Termination check
        # ----------------------------------------------------------------
        terminated_flag = False
        truncated_flag = False
        end_reason = ""

        # Count remaining food per team's side
        remaining_food = {
            0: len([p for p in state["food_positions"] if p[0] < self.config.map_width // 2]),
            1: len([p for p in state["food_positions"] if p[0] >= self.config.map_width // 2]),
        }

        winner = -1
        if remaining_food[0] == 0:
            winner = 1  # Team 1 stole all of Team 0's food
            end_reason = "food_depleted"
            terminated_flag = True
        elif remaining_food[1] == 0:
            winner = 0
            end_reason = "food_depleted"
            terminated_flag = True

        if self._step_count >= cfg.max_steps and not terminated_flag:
            truncated_flag = True
            s0, s1 = state["scores"][0], state["scores"][1]
            if s0 > s1:
                winner = 0
            elif s1 > s0:
                winner = 1
            else:
                winner = -1
            end_reason = "timeout"

        if terminated_flag or truncated_flag:
            self._done = True
            self._event_log.add(
                EpisodeEndEvent(
                    winner=winner,
                    reason=end_reason,
                    final_scores=dict(state["scores"]),
                )
            )

        # ----------------------------------------------------------------
        # 10. Compute rewards
        # ----------------------------------------------------------------
        rewards: Dict[int, float] = {}
        for aid in range(_NUM_AGENTS):
            team = state["agent_teams"][aid]
            role = state["agent_roles"][aid]
            opp_team = 1 - team

            # Determine if any opponent attacker is in home territory
            invader_present = any(
                grid.is_home_territory(new_positions[opp_aid][0], team)
                and state["agent_roles"][opp_aid] == "attacker"
                for opp_aid in range(_NUM_AGENTS)
                if state["agent_teams"][opp_aid] == opp_team
            )

            rewards[aid] = self._reward_calc.compute(
                events=self._event_log,
                agent_id=aid,
                agent_role=role,
                agent_team=team,
                invader_present=invader_present,
            )

        # ----------------------------------------------------------------
        # 11. Build observations and info
        # ----------------------------------------------------------------
        observations = self._build_all_observations()
        terminated = {aid: terminated_flag for aid in range(_NUM_AGENTS)}
        truncated = {aid: truncated_flag for aid in range(_NUM_AGENTS)}
        info = self._build_info()

        return observations, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Legal action mask
    # ------------------------------------------------------------------

    def get_legal_action_mask(self, agent_id: int) -> np.ndarray:
        """Return a boolean array of shape (5,) marking legal actions.

        Args:
            agent_id: The queried agent.

        Returns:
            Boolean array; True = legal.
        """
        x, y = self._state["agent_positions"][agent_id]
        legal_list = self._grid.get_legal_actions(x, y)
        mask = np.zeros(5, dtype=bool)
        for a in legal_list:
            mask[a] = True
        return mask

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> str:
        """ASCII render of the current grid state.

        Returns:
            Multi-line string representation.
        """
        cfg = self.config
        state = self._state
        grid = self._grid

        # Build character map
        char_map = []
        for y in range(cfg.map_height):
            row = []
            for x in range(cfg.map_width):
                if grid.is_wall(x, y):
                    row.append("#")
                elif (x, y) in state.get("power_pellet_positions", set()):
                    row.append("o")
                elif (x, y) in state.get("food_positions", set()):
                    row.append(".")
                else:
                    row.append(" ")
            char_map.append(row)

        # Overlay agents
        agent_chars = {
            (0, "attacker"): "A",
            (0, "defender"): "D",
            (1, "attacker"): "a",
            (1, "defender"): "d",
        }
        for aid in range(_NUM_AGENTS):
            ax, ay = state["agent_positions"][aid]
            team = state["agent_teams"][aid]
            role = state["agent_roles"][aid]
            ch = agent_chars.get((team, role), "?")
            char_map[ay][ax] = ch

        lines = ["".join(row) for row in char_map]
        header = (
            f"Step: {state['step']} | "
            f"Score Team0: {state['scores'][0]} | "
            f"Score Team1: {state['scores'][1]}"
        )
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_all_observations(self) -> Dict[int, dict]:
        obs = {}
        for aid in range(_NUM_AGENTS):
            grid_obs, flat_obs = self._obs_builder.build(self._state, aid)
            obs[aid] = {"grid": grid_obs, "flat": flat_obs}
        return obs

    def _build_info(self) -> Dict[int, dict]:
        legal = {aid: self.get_legal_action_mask(aid) for aid in range(_NUM_AGENTS)}
        base = {
            "events": list(self._event_log.events),
            "scores": dict(self._state["scores"]),
            "step": self._step_count,
        }
        return {aid: {**base, "legal_actions": legal[aid]} for aid in range(_NUM_AGENTS)}

    def _place_food(self) -> Set[Tuple[int, int]]:
        """Place food symmetrically on both halves of the map."""
        cfg = self.config
        grid = self._grid
        mid = cfg.map_width // 2
        food: Set[Tuple[int, int]] = set()

        # Team 0 side: x < mid
        candidates = [
            (x, y)
            for x in range(1, mid)
            for y in range(1, cfg.map_height - 1)
            if not grid.is_wall(x, y)
        ]
        self._rng.shuffle(candidates)
        for pos in candidates[:cfg.num_food_per_team]:
            food.add(pos)
            # Mirror to team 1 side
            mx = cfg.map_width - 1 - pos[0]
            food.add((mx, pos[1]))

        return food

    def _place_power_pellets(self) -> Set[Tuple[int, int]]:
        """Place power pellets symmetrically."""
        cfg = self.config
        grid = self._grid
        mid = cfg.map_width // 2
        pellets: Set[Tuple[int, int]] = set()

        candidates = [
            (x, y)
            for x in range(1, mid)
            for y in range(1, cfg.map_height - 1)
            if not grid.is_wall(x, y)
        ]
        self._rng.shuffle(candidates)
        placed = 0
        for pos in candidates:
            if placed >= cfg.num_power_pellets:
                break
            pellets.add(pos)
            mx = cfg.map_width - 1 - pos[0]
            pellets.add((mx, pos[1]))
            placed += 1

        return pellets

    def _spawn_agents(self) -> Dict[int, Tuple[int, int]]:
        """Spawn all four agents at valid starting positions."""
        positions: Dict[int, Tuple[int, int]] = {}
        occupied: Set[Tuple[int, int]] = set()

        for aid in range(_NUM_AGENTS):
            team = _AGENT_TEAMS[aid]
            role = _AGENT_ROLES[aid]
            pos = self._spawn_single(team=team, role=role, occupied=occupied)
            positions[aid] = pos
            occupied.add(pos)

        return positions

    def _spawn_single(
        self,
        team: int,
        role: str,
        occupied: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """Find a valid spawn position for an agent.

        Args:
            team:     0 or 1.
            role:     "attacker" or "defender".
            occupied: Cells already taken.

        Returns:
            (x, y) spawn coordinate.
        """
        cfg = self.config
        occupied = occupied or set()
        mid = cfg.map_width // 2
        w, h = cfg.map_width, cfg.map_height

        if team == 0:
            if role == "defender":
                x_range = range(2, min(6, mid))
            else:  # attacker
                x_range = range(max(1, mid - 6), mid)
        else:
            if role == "defender":
                x_range = range(max(mid, w - 6), w - 1)
            else:
                x_range = range(mid + 1, min(w - 1, mid + 6))

        candidates = [
            (x, y)
            for x in x_range
            for y in range(1, h - 1)
            if not self._grid.is_wall(x, y) and (x, y) not in occupied
        ]

        if not candidates:
            # Fallback: any non-wall cell in the team's half
            candidates = [
                (x, y)
                for x in (range(1, mid) if team == 0 else range(mid + 1, w - 1))
                for y in range(1, h - 1)
                if not self._grid.is_wall(x, y) and (x, y) not in occupied
            ]

        if not candidates:
            raise RuntimeError(f"No valid spawn position for team={team}, role={role}")

        idx = int(self._rng.integers(len(candidates)))
        return candidates[idx]

    def _update_last_known(self) -> None:
        """Update per-agent tracking of opponent last-seen positions."""
        r = self.config.observation_radius
        state = self._state
        step = self._step_count

        for aid in range(_NUM_AGENTS):
            ax, ay = state["agent_positions"][aid]
            my_team = state["agent_teams"][aid]

            for opp_aid in range(_NUM_AGENTS):
                if state["agent_teams"][opp_aid] == my_team:
                    continue
                ox, oy = state["agent_positions"][opp_aid]
                if abs(ox - ax) <= r and abs(oy - ay) <= r:
                    # Opponent is visible — update last known
                    state["last_known_positions"][opp_aid] = (ox, oy, step)
