"""Agent tests — all 5 cases from Part 9 plus additional coverage.

All tests run on CPU with tiny model configs for speed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ctf_pacman.agents.attacker_agent import AttackerAgent
from ctf_pacman.agents.base_agent import BaseAgent
from ctf_pacman.agents.defender_agent import DefenderAgent
from ctf_pacman.agents.rule_based_agent import RuleBasedAgent
from ctf_pacman.environment.env import CTFPacmanEnv
from ctf_pacman.environment.grid import Grid
from ctf_pacman.utils.config import AgentConfig, EnvConfig, ModelConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

OBS_R = 3
NUM_C = 10
W = 2 * OBS_R + 1   # window size
MSG_DIM = 8


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def model_cfg() -> ModelConfig:
    cfg = ModelConfig()
    cfg.cnn_channels = [16, 32]
    cfg.cnn_kernel_sizes = [3, 3]
    cfg.cnn_strides = [1, 1]
    cfg.hidden_dim = 64
    cfg.actor_hidden_dim = 32
    cfg.critic_hidden_dim = 64
    cfg.message_hidden_dim = 32
    return cfg


@pytest.fixture
def agent_cfg() -> AgentConfig:
    return AgentConfig(message_dim=MSG_DIM, use_communication=True)


@pytest.fixture
def attacker(model_cfg: ModelConfig, agent_cfg: AgentConfig, device: torch.device) -> AttackerAgent:
    return AttackerAgent(
        agent_id=0, team_id=0,
        config=agent_cfg, model_config=model_cfg,
        observation_radius=OBS_R, num_obs_channels=NUM_C,
        device=device,
    )


@pytest.fixture
def defender(model_cfg: ModelConfig, agent_cfg: AgentConfig, device: torch.device) -> DefenderAgent:
    return DefenderAgent(
        agent_id=1, team_id=0,
        config=agent_cfg, model_config=model_cfg,
        observation_radius=OBS_R, num_obs_channels=NUM_C,
        device=device,
    )


def _rand_grid(B: int = 2) -> torch.Tensor:
    return torch.randn(B, NUM_C, W, W)


def _rand_flat(B: int = 2) -> torch.Tensor:
    return torch.randn(B, 8)


# ===========================================================================
# Part 9 — required tests
# ===========================================================================

# --- 1. test_attacker_forward_shape ------------------------------------------

def test_attacker_forward_shape(attacker: AttackerAgent) -> None:
    """AttackerAgent forward pass must return tensors of the right shapes."""
    B = 3
    grid = _rand_grid(B)
    flat = _rand_flat(B)
    out = attacker.forward(grid, flat)

    assert out["action_logits"].shape == (B, 5), \
        f"action_logits shape {out['action_logits'].shape} != ({B}, 5)"
    assert out["value"].shape == (B, 1), \
        f"value shape {out['value'].shape} != ({B}, 1)"
    assert out["message"].shape == (B, MSG_DIM), \
        f"message shape {out['message'].shape} != ({B}, {MSG_DIM})"
    assert "action_dist" in out, "forward must return 'action_dist'"
    assert "hidden" in out, "forward must return 'hidden'"


# --- 2. test_defender_forward_shape ------------------------------------------

def test_defender_forward_shape(defender: DefenderAgent) -> None:
    """DefenderAgent forward pass must return tensors of the right shapes."""
    B = 2
    grid = _rand_grid(B)
    flat = _rand_flat(B)
    gs = torch.randn(B, 19)
    out = defender.forward(grid, flat, global_state=gs)

    assert out["action_logits"].shape == (B, 5)
    assert out["value"].shape == (B, 1)
    assert out["message"].shape == (B, MSG_DIM)


# --- 3. test_action_mask_applied ---------------------------------------------

def test_action_mask_applied(attacker: AttackerAgent) -> None:
    """With all actions except Stop masked, the agent must always sample Stop."""
    grid = _rand_grid(1)
    flat = _rand_flat(1)

    # Only action 4 (Stop) is legal
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 4] = True

    out = attacker.forward(grid, flat, action_mask=mask)
    dist: torch.distributions.Categorical = out["action_dist"]

    # Sample 50 times — should always be 4
    samples = torch.stack([dist.sample() for _ in range(50)])
    assert samples.eq(4).all(), "With only Stop legal, every sample must be 4"

    # Verify logits for actions 0-3 are extremely negative
    logits = out["action_logits"][0]
    for a in range(4):
        assert logits[a].item() < -1e8, \
            f"Illegal action {a} logit {logits[a].item()} should be near -1e9"


# --- 4. test_message_dim -----------------------------------------------------

def test_message_dim(attacker: AttackerAgent) -> None:
    """Message output must have exactly message_dim elements per batch item."""
    for B in (1, 4, 8):
        out = attacker.forward(_rand_grid(B), _rand_flat(B))
        assert out["message"].shape == (B, MSG_DIM), \
            f"Batch {B}: message shape {out['message'].shape} != ({B}, {MSG_DIM})"


# --- 5. test_rule_based_attacker_moves_toward_food ---------------------------

def test_rule_based_attacker_moves_toward_food() -> None:
    """
    In a simple open grid, a rule-based attacker should reach the only food
    item placed in opponent territory within 20 steps.
    """
    # Small open grid with no random walls
    env_cfg = EnvConfig()
    env_cfg.map_width = 14
    env_cfg.map_height = 8
    env_cfg.wall_density = 0.0
    env_cfg.num_food_per_team = 4
    env_cfg.num_power_pellets = 0
    env_cfg.max_steps = 30
    env_cfg.observation_radius = 3
    env_cfg.num_observation_channels = 10

    env = CTFPacmanEnv(env_cfg, seed=0)
    env.reset()

    mid = env_cfg.map_width // 2
    # Clear existing food and place exactly one item in opponent territory
    env._state["food_positions"] = {(mid + 2, 4)}

    agent = RuleBasedAgent(role="attacker")
    reached = False

    for _ in range(20):
        state = dict(env._state)
        state["grid"] = env._grid
        state["width"] = env_cfg.map_width
        state["height"] = env_cfg.map_height

        action = agent.act(state, agent_id=0)
        env.step({0: action, 1: 4, 2: 4, 3: 4})

        if env._state["agent_carrying"][0] > 0:
            reached = True
            break
        if not env._state["food_positions"]:
            reached = True
            break

    assert reached, "Rule-based attacker must reach the food within 20 steps"


# ===========================================================================
# Additional coverage
# ===========================================================================

class TestAgentCommon:
    def test_role_strings(self, attacker: AttackerAgent, defender: DefenderAgent) -> None:
        assert attacker.role == "attacker"
        assert defender.role == "defender"

    def test_team_ids(self, attacker: AttackerAgent, defender: DefenderAgent) -> None:
        assert attacker.team_id == 0
        assert defender.team_id == 0

    def test_message_bounded_tanh(self, attacker: AttackerAgent) -> None:
        """Messages must stay in [-1, 1] (Tanh activation)."""
        out = attacker.forward(_rand_grid(16), _rand_flat(16))
        msg = out["message"]
        assert msg.min().item() >= -1.0 - 1e-5
        assert msg.max().item() <= 1.0 + 1e-5

    def test_get_value_shape(self, attacker: AttackerAgent) -> None:
        gs = torch.randn(4, 19)
        v = attacker.get_value(gs)
        assert v.shape == (4, 1)

    def test_act_returns_valid_action(self, attacker: AttackerAgent) -> None:
        obs = {
            "grid": np.random.randn(NUM_C, W, W).astype(np.float32),
            "flat": np.random.randn(8).astype(np.float32),
        }
        action, log_prob, entropy = attacker.act(obs, deterministic=False)
        assert 0 <= action <= 4
        assert isinstance(log_prob, float)
        assert isinstance(entropy, float)

    def test_deterministic_act_is_reproducible(self, attacker: AttackerAgent) -> None:
        obs = {
            "grid": np.ones((NUM_C, W, W), dtype=np.float32) * 0.3,
            "flat": np.ones(8, dtype=np.float32) * 0.1,
        }
        a1, _, _ = attacker.act(obs, deterministic=True)
        a2, _, _ = attacker.act(obs, deterministic=True)
        assert a1 == a2

    def test_parameters_non_empty(self, attacker: AttackerAgent) -> None:
        assert len(list(attacker.parameters())) > 0

    def test_no_grad_during_act(self, attacker: AttackerAgent) -> None:
        """act() must not create a computation graph (Part 11)."""
        obs = {
            "grid": np.random.randn(NUM_C, W, W).astype(np.float32),
            "flat": np.random.randn(8).astype(np.float32),
        }
        attacker.act(obs, deterministic=False)
        # No exception = no stray gradient tracking
        for p in attacker.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0


class TestNoCommunicationMode:
    def test_forward_works_without_message(
        self,
        model_cfg: ModelConfig,
        device: torch.device,
    ) -> None:
        no_comm = AgentConfig(message_dim=MSG_DIM, use_communication=False)
        agent = AttackerAgent(
            agent_id=0, team_id=0,
            config=no_comm, model_config=model_cfg,
            observation_radius=OBS_R, num_obs_channels=NUM_C,
            device=device,
        )
        out = agent.forward(_rand_grid(2), _rand_flat(2), received_message=None)
        assert out["action_logits"].shape == (2, 5)
        assert out["message"].shape == (2, MSG_DIM)


class TestRuleBasedAgent:
    def _state(self, grid: Grid, cfg: EnvConfig) -> dict:
        mid = cfg.map_width // 2
        return {
            "agent_positions": {0: (2, 4), 1: (3, 4), 2: (mid + 2, 4), 3: (mid + 3, 4)},
            "agent_teams": {0: 0, 1: 0, 2: 1, 3: 1},
            "agent_roles": {0: "attacker", 1: "defender", 2: "attacker", 3: "defender"},
            "agent_carrying": {0: 0, 1: 0, 2: 0, 3: 0},
            "agent_scared": {0: 0, 1: 0, 2: 0, 3: 0},
            "food_positions": {(mid + 3, 3), (mid + 4, 4)},
            "power_pellet_positions": set(),
            "scores": {0: 0, 1: 0},
            "step": 0,
            "last_known_positions": {},
            "grid": grid,
            "width": cfg.map_width,
            "height": cfg.map_height,
        }

    def test_attacker_action_valid(self) -> None:
        cfg = EnvConfig()
        cfg.map_width = 20
        cfg.map_height = 10
        cfg.wall_density = 0.0
        g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
        agent = RuleBasedAgent(role="attacker")
        action = agent.act(self._state(g, cfg), agent_id=0)
        assert 0 <= action <= 4

    def test_defender_action_valid(self) -> None:
        cfg = EnvConfig()
        cfg.map_width = 20
        cfg.map_height = 10
        cfg.wall_density = 0.0
        g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
        agent = RuleBasedAgent(role="defender")
        action = agent.act(self._state(g, cfg), agent_id=1)
        assert 0 <= action <= 4

    def test_bfs_returns_stop_for_unreachable(self) -> None:
        """BFS must return Stop (4) when no path exists to the goal."""
        cfg = EnvConfig()
        cfg.map_width = 20
        cfg.map_height = 10
        cfg.wall_density = 0.0
        g = Grid(cfg.map_width, cfg.map_height, cfg.wall_density, seed=0)
        agent = RuleBasedAgent()
        # Goal at (0, 0) which is always a border wall
        action = agent._bfs_next_action((1, 1), (0, 0), g)
        assert action == 4, f"Unreachable goal: expected Stop(4), got {action}"
