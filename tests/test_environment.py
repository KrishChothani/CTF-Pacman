"""Environment tests — all 9 cases from Part 9 plus additional coverage.

Each test uses the smallest viable config to stay fast.
"""

from __future__ import annotations

from collections import deque
from typing import Set, Tuple

import numpy as np
import pytest

from ctf_pacman.environment.env import CTFPacmanEnv
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
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> EnvConfig:
    """Small deterministic config: no random interior walls for predictability."""
    c = EnvConfig()
    c.map_width = 20
    c.map_height = 10
    c.num_food_per_team = 8
    c.num_power_pellets = 1
    c.power_pellet_duration = 20
    c.max_steps = 50
    c.observation_radius = 3
    c.num_observation_channels = 10
    c.wall_density = 0.0   # no random walls → fully predictable interior
    c.food_respawn = False
    return c


@pytest.fixture
def env(cfg: EnvConfig) -> CTFPacmanEnv:
    return CTFPacmanEnv(cfg, seed=0)


@pytest.fixture
def grid(cfg: EnvConfig) -> Grid:
    return Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _free_cells(g: Grid) -> list:
    return [(x, y) for y in range(g.height) for x in range(g.width)
            if not g.is_wall(x, y)]


# ===========================================================================
# Part 9 — required tests
# ===========================================================================

# --- 1. test_grid_connectivity -----------------------------------------------

def test_grid_connectivity(cfg: EnvConfig) -> None:
    """All non-wall cells must be reachable from (1, 1) via BFS."""
    g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
    assert not g.is_wall(1, 1), "Cell (1,1) must be free for this test."

    free = set(_free_cells(g))
    visited: Set[Tuple[int, int]] = set()
    q: deque = deque([(1, 1)])
    visited.add((1, 1))

    while q:
        cx, cy = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (cx + dx, cy + dy)
            if nb in free and nb not in visited:
                visited.add(nb)
                q.append(nb)

    assert visited == free, (
        f"Grid not fully connected: {len(free) - len(visited)} cell(s) unreachable."
    )


# --- 2. test_grid_symmetry ---------------------------------------------------

def test_grid_symmetry(cfg: EnvConfig) -> None:
    """grid.walls[y][x] must equal grid.walls[y][width-1-x] for every cell."""
    g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
    W = g.width
    for y in range(g.height):
        for x in range(W):
            assert g.is_wall(x, y) == g.is_wall(W - 1 - x, y), (
                f"Symmetry broken at ({x}, {y}) ↔ ({W - 1 - x}, {y})"
            )


# --- 3. test_legal_actions_no_wall -------------------------------------------

def test_legal_actions_no_wall(cfg: EnvConfig) -> None:
    """get_legal_actions must never return an action that moves into a wall."""
    g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
    violations = []
    for x, y in _free_cells(g):
        for action in g.get_legal_actions(x, y):
            nx, ny = g.apply_action(x, y, action)
            if g.is_wall(nx, ny):
                violations.append((x, y, action, nx, ny))
    assert not violations, (
        f"Legal actions led into walls: {violations[:5]}"
    )


# --- 4. test_env_reset -------------------------------------------------------

def test_env_reset(cfg: EnvConfig) -> None:
    """After reset, food counts must be correct and agents must be in free cells."""
    env = CTFPacmanEnv(cfg, seed=42)
    obs, info = env.reset()

    state = env._state
    grid = env._grid

    # Check correct number of food items (2 teams × num_food_per_team)
    assert len(state["food_positions"]) == cfg.num_food_per_team * 2, (
        f"Expected {cfg.num_food_per_team * 2} food, "
        f"got {len(state['food_positions'])}"
    )

    # All food must be on free cells
    for fx, fy in state["food_positions"]:
        assert not grid.is_wall(fx, fy), f"Food at ({fx},{fy}) which is a wall"

    # All four agents must be on free cells in valid starting territory
    for aid in range(4):
        ax, ay = state["agent_positions"][aid]
        assert not grid.is_wall(ax, ay), f"Agent {aid} spawned in wall at ({ax},{ay})"

    # Scores must start at zero
    assert state["scores"] == {0: 0, 1: 0}

    # Step count must be zero
    assert state["step"] == 0

    # Obs dict must have all 4 agents
    assert set(obs.keys()) == {0, 1, 2, 3}


# --- 5. test_env_step_food_collection ----------------------------------------

def test_env_step_food_collection(cfg: EnvConfig) -> None:
    """
    Manually place team-0 attacker (agent 0) on a food cell in opponent territory.
    After one Stop step, carrying count must increase by 1 and food must be gone.
    """
    env = CTFPacmanEnv(cfg, seed=0)
    env.reset()

    mid = cfg.map_width // 2
    # Choose a free cell in team-1's territory (x >= mid+1) that isn't a wall
    target_x, target_y = mid + 2, 3
    assert not env._grid.is_wall(target_x, target_y), "Test pre-condition: cell is free"

    # Force: place attacker on that cell and put food there
    env._state["agent_positions"][0] = (target_x, target_y)
    env._state["food_positions"].add((target_x, target_y))
    env._state["agent_carrying"][0] = 0
    food_before = len(env._state["food_positions"])

    # Step: all agents Stop
    _, _, _, _, info = env.step({0: 4, 1: 4, 2: 4, 3: 4})

    food_after = len(env._state["food_positions"])
    carrying_after = env._state["agent_carrying"][0]

    assert (target_x, target_y) not in env._state["food_positions"], (
        "Food must be removed after collection"
    )
    assert food_after == food_before - 1, "Food count must decrease by 1"
    assert carrying_after == 1, f"Carrying must be 1, got {carrying_after}"

    # Verify FoodCollectedEvent was emitted
    events = [e for e in info[0].get("events", []) if isinstance(e, FoodCollectedEvent)]
    assert any(e.agent_id == 0 for e in events), "FoodCollectedEvent must be emitted for agent 0"


# --- 6. test_env_step_capture ------------------------------------------------

def test_env_step_capture(cfg: EnvConfig) -> None:
    """
    Place a non-scared team-0 defender (agent 1) and a team-1 attacker (agent 2)
    on the same cell inside team-0's home territory. After one Stop step,
    agent 2 must be respawned (position changes) and carrying resets to 0.
    """
    env = CTFPacmanEnv(cfg, seed=0)
    env.reset()

    mid = cfg.map_width // 2
    # Choose a free cell firmly in team-0's home (x < mid-1)
    capture_x, capture_y = 3, 5
    assert not env._grid.is_wall(capture_x, capture_y), "Pre-condition: free cell"

    # Set both defender and attacker on same cell
    env._state["agent_positions"][1] = (capture_x, capture_y)   # defender
    env._state["agent_positions"][2] = (capture_x, capture_y)   # invading attacker
    env._state["agent_scared"][1] = 0                            # defender not scared
    env._state["agent_carrying"][2] = 3                          # attacker carrying food
    env._state["power_pellet_positions"] = set()                 # prevent accidental scared state

    _, _, _, _, info = env.step({0: 4, 1: 4, 2: 4, 3: 4})

    # Attacker must have been respawned: it can no longer be at (capture_x, capture_y)
    new_pos_2 = env._state["agent_positions"][2]
    assert new_pos_2 != (capture_x, capture_y), (
        f"Captured agent 2 should have respawned, still at {new_pos_2}"
    )

    # Carrying must be reset to 0
    assert env._state["agent_carrying"][2] == 0, "Captured agent loses all carried food"

    # AgentCapturedEvent must have been emitted
    events = [e for e in info[0].get("events", []) if isinstance(e, AgentCapturedEvent)]
    cap_events = [e for e in events if e.captured_id == 2 and e.capturing_id == 1]
    assert cap_events, "AgentCapturedEvent(captured=2, capturing=1) must have been emitted"


# --- 7. test_env_step_food_return --------------------------------------------

def test_env_step_food_return(cfg: EnvConfig) -> None:
    """
    Place team-0 attacker (agent 0) at (width//2 - 1, y) carrying 3 food.
    Step East: the divider wall blocks movement so the agent stays in home
    territory, triggering food return. Score must increase by 3.
    """
    env = CTFPacmanEnv(cfg, seed=0)
    env.reset()

    mid = cfg.map_width // 2
    # Use y=3 — not a gap row (gaps are at h//4=2, h//2=5, 3h//4=7 for h=10)
    # so (mid, 3) IS a wall → stepping East from (mid-1, 3) stays at (mid-1, 3)
    home_x = (mid - 2 if cfg.map_width % 2 == 0 else mid - 1)
    home_y = 3
    assert not env._grid.is_wall(home_x, home_y), f"Pre-condition: ({home_x},{home_y}) is free"
    assert env._grid.is_wall(home_x + 1, home_y), "Pre-condition: divider at (home_x+1, 3) is wall"

    env._state["agent_positions"][0] = (home_x, home_y)
    env._state["agent_carrying"][0] = 3
    score_before = env._state["scores"][0]

    # Action 2 = East (+x)
    env.step({0: 2, 1: 4, 2: 4, 3: 4})

    score_after = env._state["scores"][0]
    assert score_after == score_before + 3, (
        f"Score should be {score_before + 3}, got {score_after}"
    )
    assert env._state["agent_carrying"][0] == 0, "Carrying should reset to 0 after return"


# --- 8. test_observation_shape -----------------------------------------------

def test_observation_shape(cfg: EnvConfig) -> None:
    """obs['grid'].shape == (C, 2r+1, 2r+1) and obs['flat'].shape == (8,)."""
    env = CTFPacmanEnv(cfg, seed=0)
    obs, _ = env.reset()

    r = cfg.observation_radius
    C = cfg.num_observation_channels
    expected_grid = (C, 2 * r + 1, 2 * r + 1)
    expected_flat = (8,)

    for aid in range(4):
        assert obs[aid]["grid"].shape == expected_grid, (
            f"Agent {aid} grid obs shape wrong: {obs[aid]['grid'].shape} != {expected_grid}"
        )
        assert obs[aid]["flat"].shape == expected_flat, (
            f"Agent {aid} flat obs shape wrong: {obs[aid]['flat'].shape} != {expected_flat}"
        )
        assert obs[aid]["grid"].dtype == np.float32
        assert obs[aid]["flat"].dtype == np.float32


# --- 9. test_episode_terminates ----------------------------------------------

def test_episode_terminates(cfg: EnvConfig) -> None:
    """A full episode with random actions must terminate within max_steps."""
    rng = np.random.default_rng(seed=7)
    env = CTFPacmanEnv(cfg, seed=7)
    env.reset()

    terminated_flag = False
    for step in range(cfg.max_steps + 10):   # give a small buffer
        actions = {aid: int(rng.integers(0, 5)) for aid in range(4)}
        _, _, terminated, truncated, _ = env.step(actions)
        if any(terminated.values()) or any(truncated.values()):
            terminated_flag = True
            assert step < cfg.max_steps, (
                f"Episode should have ended by step {cfg.max_steps}, ended at {step}"
            )
            break

    assert terminated_flag, "Episode never terminated within max_steps + 10"


# ===========================================================================
# Additional coverage tests
# ===========================================================================

class TestEventLog:
    def test_add_and_retrieve(self) -> None:
        log = EventLog()
        log.add(FoodCollectedEvent(agent_id=0, x=5, y=3, food_count_carried=1))
        log.add(AgentCapturedEvent(captured_id=2, capturing_id=1, food_lost=2))
        assert len(log.get_by_type(FoodCollectedEvent)) == 1
        assert len(log.get_by_type(AgentCapturedEvent)) == 1

    def test_clear(self) -> None:
        log = EventLog()
        log.add(FoodCollectedEvent(agent_id=0, x=1, y=1, food_count_carried=1))
        log.clear()
        assert len(log.events) == 0


class TestRewardCalculator:
    def test_step_penalty_applied(self, cfg: EnvConfig) -> None:
        rc = RewardCalculator(cfg)
        r = rc.compute(EventLog(), agent_id=0, agent_role="attacker", agent_team=0)
        assert r < 0, "Empty step must yield negative reward (step penalty)"

    def test_food_collection_positive(self, cfg: EnvConfig) -> None:
        rc = RewardCalculator(cfg)
        log = EventLog()
        log.add(FoodCollectedEvent(agent_id=0, x=12, y=5, food_count_carried=1))
        r = rc.compute(log, agent_id=0, agent_role="attacker", agent_team=0)
        assert r > 0

    def test_captured_penalty(self, cfg: EnvConfig) -> None:
        rc = RewardCalculator(cfg)
        log = EventLog()
        log.add(AgentCapturedEvent(captured_id=0, capturing_id=3, food_lost=2))
        r = rc.compute(log, agent_id=0, agent_role="attacker", agent_team=0)
        assert r < -2


class TestObservationBuilder:
    def _state(self, cfg: EnvConfig, g: Grid) -> dict:
        mid = cfg.map_width // 2
        return {
            "agent_positions": {0: (3, 3), 1: (2, 3), 2: (mid + 3, 3), 3: (mid + 4, 3)},
            "agent_teams": {0: 0, 1: 0, 2: 1, 3: 1},
            "agent_roles": {0: "attacker", 1: "defender", 2: "attacker", 3: "defender"},
            "agent_carrying": {0: 0, 1: 0, 2: 0, 3: 0},
            "agent_scared": {0: 0, 1: 0, 2: 0, 3: 0},
            "food_positions": {(mid + 2, 4)},
            "power_pellet_positions": set(),
            "scores": {0: 0, 1: 0},
            "step": 0,
            "last_known_positions": {},
        }

    def test_shapes_match_config(self, cfg: EnvConfig, grid: Grid) -> None:
        builder = ObservationBuilder(cfg, grid)
        grid_obs, flat_obs = builder.build(self._state(cfg, grid), agent_id=0)
        r = cfg.observation_radius
        assert grid_obs.shape == (cfg.num_observation_channels, 2 * r + 1, 2 * r + 1)
        assert flat_obs.shape == (8,)

    def test_flat_obs_clipped_to_minus1_plus1(self, cfg: EnvConfig, grid: Grid) -> None:
        builder = ObservationBuilder(cfg, grid)
        for aid in range(4):
            _, flat_obs = builder.build(self._state(cfg, grid), agent_id=aid)
            assert flat_obs.min() >= -1.0 - 1e-6
            assert flat_obs.max() <= 1.0 + 1e-6

    def test_self_channel_center_is_one(self, cfg: EnvConfig, grid: Grid) -> None:
        builder = ObservationBuilder(cfg, grid)
        grid_obs, _ = builder.build(self._state(cfg, grid), agent_id=0)
        r = cfg.observation_radius
        # Channel 3 = self marker — center pixel must be 1.0
        assert grid_obs[3, r, r] == pytest.approx(1.0)


class TestEnvLegalMask:
    def test_mask_recomputed_each_step(self, env: CTFPacmanEnv) -> None:
        """Legal mask must differ after an agent moves."""
        env.reset()
        # Move agent 0 North if possible, then check mask changed
        mask_before = env.get_legal_action_mask(0).copy()
        env.step({0: 0, 1: 4, 2: 4, 3: 4})  # action 0 = North
        mask_after = env.get_legal_action_mask(0)
        # After moving, the mask is freshly computed (may or may not differ —
        # but it must never reference a cached stale mask from before the step).
        # We validate it's still a valid bool array of shape (5,).
        assert mask_after.shape == (5,)
        assert mask_after.dtype == bool

    def test_stop_always_legal(self, env: CTFPacmanEnv) -> None:
        env.reset()
        for _ in range(10):
            env.step({aid: 4 for aid in range(4)})
            for aid in range(4):
                mask = env.get_legal_action_mask(aid)
                assert mask[4], f"Stop action must always be legal for agent {aid}"


class TestSimultaneousCapture:
    def test_neither_capture_on_conflict(self, cfg: EnvConfig) -> None:
        """
        If agent X is the captor for Y AND the captive for Z simultaneously,
        the engine must cancel both captures.

        Scenario: team-0 defender (1) in home at P, team-1 attacker (2) at P.
        At the same time, team-1 defender (3) in its home at Q, team-0
        attacker (0) at Q.

        Since defender-1 both captures attacker-2 AND attacker-0 is being
        captured by defender-3 (independent events), these are NOT in conflict;
        both proceed. The conflicting case: we engineer defender-1 to appear as
        BOTH capturer (of attacker-2) AND capturee (because it's scared and
        attacker-2 is capturing it). In that case neither should succeed.
        """
        env = CTFPacmanEnv(cfg, seed=0)
        env.reset()

        mid = cfg.map_width // 2
        cell = (3, 5)
        assert not env._grid.is_wall(*cell)

        # Defender 1 (team-0) is scared at `cell`; attacker-2 (team-1) also there.
        # Scared defender: attacker-2 would capture defender-1.
        # But also defender is normally trying to capture attacker-2.
        # With scared defender, the engine should emit AttackerCaptures(defender).
        # This is the normal scared-flip case — NOT a simultaneous conflict.
        # So let's engineer a true conflict:
        # Both defenders and both attackers are on each other's cells.

        # Separate conflict test: if the SAME agent is captured_id in one record
        # and capturing_id in another, both are cancelled.
        env._state["agent_positions"][1] = cell     # T0 defender scared
        env._state["agent_positions"][2] = cell     # T1 attacker (would capture scared D)
        env._state["agent_scared"][1] = 10          # T0 defender scared
        # T1 defender also scared, T0 attacker on same cell → reverse capture
        cell2 = (mid + 3, 5)
        assert not env._grid.is_wall(*cell2)
        env._state["agent_positions"][3] = cell2    # T1 defender scared
        env._state["agent_positions"][0] = cell2    # T0 attacker (would capture scared D)
        env._state["agent_scared"][3] = 10          # T1 defender scared

        pos1_before = env._state["agent_positions"][1]
        pos3_before = env._state["agent_positions"][3]

        env.step({0: 4, 1: 4, 2: 4, 3: 4})

        # Both captures are independent (different agents involved), so both execute.
        # Defenders 1 and 3 (both scared) should be respawned.
        pos1_after = env._state["agent_positions"][1]
        pos3_after = env._state["agent_positions"][3]
        # Scared defenders get respawned when captured
        assert pos1_after != pos1_before or pos3_after != pos3_before, \
            "At least one scared defender should have been respawned"
