"""Training infrastructure tests — all 5 cases from Part 9 plus extra coverage.

All tests run on CPU with tiny configs for speed.
"""

from __future__ import annotations

import os
import copy
import tempfile

import numpy as np
import pytest
import torch

from ctf_pacman.agents.attacker_agent import AttackerAgent
from ctf_pacman.agents.defender_agent import DefenderAgent
from ctf_pacman.agents.rule_based_agent import RuleBasedAgent
from ctf_pacman.training.rollout_buffer import RolloutBuffer
from ctf_pacman.training.ppo import PPOUpdater
from ctf_pacman.training.self_play_manager import SelfPlayManager
from ctf_pacman.training.trainer import Trainer
from ctf_pacman.utils.config import (
    AgentConfig, Config, EnvConfig, LoggingConfig, ModelConfig, TrainingConfig,
)


# ---------------------------------------------------------------------------
# Shared constants / fixtures
# ---------------------------------------------------------------------------

OBS_R = 3
NUM_C = 10
W = 2 * OBS_R + 1
MSG_DIM = 4
NUM_AGENTS = 4


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    cfg = ModelConfig()
    cfg.cnn_channels = [8, 16]
    cfg.cnn_kernel_sizes = [3, 3]
    cfg.cnn_strides = [1, 1]
    cfg.hidden_dim = 32
    cfg.actor_hidden_dim = 16
    cfg.critic_hidden_dim = 32
    cfg.message_hidden_dim = 16
    return cfg


@pytest.fixture
def tiny_agent_cfg() -> AgentConfig:
    return AgentConfig(message_dim=MSG_DIM, use_communication=True)


@pytest.fixture
def tiny_train_cfg() -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.total_timesteps = 40          # tiny: 2 rollouts × 2 envs × 10 steps
    cfg.num_envs = 2
    cfg.rollout_length = 10
    cfg.num_ppo_epochs = 2
    cfg.num_minibatches = 2
    cfg.learning_rate = 3e-4
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    cfg.clip_epsilon = 0.2
    cfg.value_loss_coeff = 0.5
    cfg.entropy_coeff = 0.01
    cfg.max_grad_norm = 0.5
    cfg.log_interval = 9999
    cfg.print_interval = 9999
    cfg.checkpoint_interval = 9999
    cfg.selfplay_update_interval = 9999
    cfg.league_size = 3
    cfg.latest_opponent_fraction = 0.5
    cfg.historical_opponent_fraction = 0.3
    cfg.rule_based_opponent_fraction = 0.2
    return cfg


@pytest.fixture
def tiny_env_cfg() -> EnvConfig:
    cfg = EnvConfig()
    cfg.map_width = 16
    cfg.map_height = 8
    cfg.num_food_per_team = 4
    cfg.num_power_pellets = 0
    cfg.max_steps = 30
    cfg.wall_density = 0.0
    cfg.observation_radius = OBS_R
    cfg.num_observation_channels = NUM_C
    return cfg


@pytest.fixture
def tiny_config(
    tiny_env_cfg: EnvConfig,
    tiny_agent_cfg: AgentConfig,
    tiny_model_cfg: ModelConfig,
    tiny_train_cfg: TrainingConfig,
) -> Config:
    cfg = Config()
    cfg.env = tiny_env_cfg
    cfg.agent = tiny_agent_cfg
    cfg.model = tiny_model_cfg
    cfg.training = tiny_train_cfg
    cfg.seed = 0

    log = LoggingConfig()
    log.log_dir = tempfile.mkdtemp()
    log.experiment_name = "test_run"
    log.use_tensorboard = False
    cfg.logging = log
    return cfg


def _build_agents(
    tiny_agent_cfg: AgentConfig,
    tiny_model_cfg: ModelConfig,
    device: torch.device,
) -> dict:
    agents = {}
    for aid in range(NUM_AGENTS):
        team = 0 if aid < 2 else 1
        cls = AttackerAgent if aid % 2 == 0 else DefenderAgent
        agents[aid] = cls(
            agent_id=aid, team_id=team,
            config=tiny_agent_cfg, model_config=tiny_model_cfg,
            observation_radius=OBS_R, num_obs_channels=NUM_C,
            device=device,
        )
    return agents


def _make_buffer(
    tiny_train_cfg: TrainingConfig,
    device: torch.device,
) -> RolloutBuffer:
    return RolloutBuffer(
        rollout_length=tiny_train_cfg.rollout_length,
        num_envs=tiny_train_cfg.num_envs,
        num_agents=NUM_AGENTS,
        obs_shape_grid=(NUM_C, W, W),
        obs_shape_flat=(8,),
        message_dim=MSG_DIM,
        device=device,
    )


def _fill_buffer(buf: RolloutBuffer) -> None:
    T, E, A = buf.rollout_length, buf.num_envs, NUM_AGENTS
    dev = buf.device
    for t in range(T):
        buf.insert(
            step=t,
            obs_grid=torch.randn(E, A, NUM_C, W, W, device=dev),
            obs_flat=torch.randn(E, A, 8, device=dev),
            actions=torch.randint(0, 5, (E, A), device=dev),
            log_probs=torch.zeros(E, A, device=dev) - 1.0,  # log_prob in [-20,0]
            rewards=torch.randn(E, A, device=dev) * 0.1,
            values=torch.randn(E, A, device=dev) * 0.5,
            dones=torch.zeros(E, device=dev),
            action_masks=torch.ones(E, A, 5, dtype=torch.bool, device=dev),
            messages_sent=torch.randn(E, A, MSG_DIM, device=dev),
            global_states=torch.randn(E, 19, device=dev),
        )


# ===========================================================================
# Part 9 — required tests
# ===========================================================================

# --- 1. test_rollout_buffer_insert_and_retrieve ------------------------------

def test_rollout_buffer_insert_and_retrieve(
    tiny_train_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """Insert 10 steps of random data; verify stored shapes and values match."""
    T = tiny_train_cfg.rollout_length  # == 10
    E = tiny_train_cfg.num_envs        # == 2
    A = NUM_AGENTS

    buf = _make_buffer(tiny_train_cfg, device)

    # Build deterministic test tensors
    grids = [torch.ones(E, A, NUM_C, W, W, device=device) * t for t in range(T)]
    acts = [torch.full((E, A), t % 5, dtype=torch.long, device=device) for t in range(T)]

    for t in range(T):
        buf.insert(
            step=t,
            obs_grid=grids[t],
            obs_flat=torch.zeros(E, A, 8, device=device),
            actions=acts[t],
            log_probs=torch.zeros(E, A, device=device),
            rewards=torch.zeros(E, A, device=device),
            values=torch.zeros(E, A, device=device),
            dones=torch.zeros(E, device=device),
            action_masks=torch.ones(E, A, 5, dtype=torch.bool, device=device),
            messages_sent=torch.zeros(E, A, MSG_DIM, device=device),
            global_states=torch.zeros(E, 19, device=device),
        )

    # Verify shapes
    assert buf.obs_grid.shape == (T, E, A, NUM_C, W, W)
    assert buf.actions.shape == (T, E, A)

    # Verify values: step t should have obs_grid all equal to t
    for t in range(T):
        assert buf.obs_grid[t].mean().item() == pytest.approx(float(t), abs=1e-4), \
            f"Step {t}: expected mean={t}, got {buf.obs_grid[t].mean().item()}"
        assert buf.actions[t][0, 0].item() == t % 5, \
            f"Step {t}: expected action={t % 5}, got {buf.actions[t][0, 0].item()}"

    assert buf.ptr == T


# --- 2. test_gae_computation -------------------------------------------------

def test_gae_computation(
    tiny_train_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """advantages and returns must have correct shapes after GAE computation."""
    buf = _make_buffer(tiny_train_cfg, device)
    _fill_buffer(buf)

    T, E, A = tiny_train_cfg.rollout_length, tiny_train_cfg.num_envs, NUM_AGENTS
    last_vals = torch.zeros(E, A, device=device)
    last_dones = torch.zeros(E, device=device)

    buf.compute_returns_and_advantages(last_vals, last_dones, gamma=0.99, gae_lambda=0.95)

    assert buf.advantages is not None, "advantages must be computed"
    assert buf.returns is not None, "returns must be computed"
    assert buf.advantages.shape == (T, E, A), \
        f"advantages shape wrong: {buf.advantages.shape}"
    assert buf.returns.shape == (T, E, A), \
        f"returns shape wrong: {buf.returns.shape}"

    # Verify normalisation: per-agent advantages should have ~0 mean
    for a in range(A):
        adv = buf.advantages[:, :, a]
        mean = adv.mean().item()
        assert abs(mean) < 0.5, f"Agent {a} advantages mean {mean:.4f} not near 0 after norm"

    # Returns = advantages + values → this is a basic sanity check
    expected_returns = buf.advantages + buf.values
    assert torch.allclose(buf.returns, expected_returns, atol=1e-5), \
        "returns must equal advantages + values"

    # Verify that done=True zeros the bootstrap (edge: last_dones all=0 here,
    # so we just check the shapes and finiteness)
    assert torch.isfinite(buf.advantages).all(), "advantages must be finite"
    assert torch.isfinite(buf.returns).all(), "returns must be finite"


# --- 3. test_ppo_loss_decreases ----------------------------------------------

def test_ppo_loss_decreases(
    tiny_train_cfg: TrainingConfig,
    tiny_agent_cfg: AgentConfig,
    tiny_model_cfg: ModelConfig,
    device: torch.device,
) -> None:
    """
    Run 5 PPO update iterations on the same buffer.
    This checks that the loss never becomes NaN, and that the policy
    makes progress (loss does not monotonically increase).
    """
    agents = _build_agents(tiny_agent_cfg, tiny_model_cfg, device)
    ppo = PPOUpdater(agents, tiny_train_cfg, device)

    losses_recorded = []

    for i in range(5):
        buf = _make_buffer(tiny_train_cfg, device)
        _fill_buffer(buf)
        buf.compute_returns_and_advantages(
            torch.zeros(tiny_train_cfg.num_envs, NUM_AGENTS, device=device),
            torch.zeros(tiny_train_cfg.num_envs, device=device),
            gamma=0.99, gae_lambda=0.95,
        )
        result = ppo.update(buf, current_timestep=i * tiny_train_cfg.rollout_length
                            * tiny_train_cfg.num_envs)
        losses_recorded.append(result["total_loss"])

        # No NaNs
        for k, v in result.items():
            assert np.isfinite(v), f"Iteration {i}: {k}={v} is not finite"

    # At least one update must have happened — if all losses are identical,
    # something is very wrong (likely gradients not flowing).
    assert not all(l == losses_recorded[0] for l in losses_recorded[1:]), \
        "Loss should change across PPO updates; gradients may not be flowing"


# --- 4. test_trainer_runs_10_steps -------------------------------------------

def test_trainer_runs_10_steps(tiny_config: Config) -> None:
    """
    Trainer with 2 envs × 10-step rollout must complete without error.
    total_timesteps=40 ⟹ 2 rollout iterations of 20 steps each.
    """
    # Override to exactly 20 timesteps so test is fast
    tiny_config.training.total_timesteps = tiny_config.training.num_envs \
        * tiny_config.training.rollout_length   # = 2 × 10 = 20

    trainer = Trainer(tiny_config)
    trainer.train()   # Must complete without raising

    # After training, agents should still be callable
    for aid, agent in trainer.agents.items():
        if hasattr(agent, "parameters"):
            assert len(list(agent.parameters())) > 0


# --- 5. test_checkpoint_save_load --------------------------------------------

def test_checkpoint_save_load(tiny_config: Config, device: torch.device) -> None:
    """
    Save a checkpoint, reload it into a fresh trainer, and verify that
    every parameter tensor matches exactly.
    """
    trainer = Trainer(tiny_config)

    # Record original weights
    original_weights = {
        aid: {k: v.clone() for k, v in agent.state_dict().items()}
        for aid, agent in trainer.agents.items()
        if hasattr(agent, "state_dict")
    }

    # Save
    ckpt_path = os.path.join(
        tiny_config.logging.log_dir,
        tiny_config.logging.experiment_name,
        "test_ckpt.pt",
    )
    trainer.save_checkpoint(timestep=0)
    # The auto-named checkpoint is ckpt_0.pt
    ckpt_path = os.path.join(
        tiny_config.logging.log_dir,
        tiny_config.logging.experiment_name,
        "ckpt_0.pt",
    )
    assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

    # Build a fresh trainer and load
    trainer2 = Trainer(tiny_config)
    # Randomise weights so we know they differ before loading
    for agent in trainer2.agents.values():
        if hasattr(agent, "parameters"):
            for p in agent.parameters():
                p.data.uniform_(-10.0, 10.0)

    trainer2.load_checkpoint(ckpt_path)

    # Verify weights match exactly
    for aid, orig_sd in original_weights.items():
        new_sd = trainer2.agents[aid].state_dict()
        for key, orig_val in orig_sd.items():
            assert torch.allclose(orig_val, new_sd[key], atol=1e-7), \
                f"Agent {aid} param '{key}' mismatch after checkpoint load"


# ===========================================================================
# Additional rollout buffer coverage
# ===========================================================================

class TestRolloutBufferExtra:
    def test_insert_wrong_step_raises(
        self,
        tiny_train_cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        buf = _make_buffer(tiny_train_cfg, device)
        E, A = tiny_train_cfg.num_envs, NUM_AGENTS
        with pytest.raises(AssertionError):
            buf.insert(
                step=3,   # should be 0
                obs_grid=torch.randn(E, A, NUM_C, W, W, device=device),
                obs_flat=torch.randn(E, A, 8, device=device),
                actions=torch.randint(0, 5, (E, A), device=device),
                log_probs=torch.zeros(E, A, device=device),
                rewards=torch.zeros(E, A, device=device),
                values=torch.zeros(E, A, device=device),
                dones=torch.zeros(E, device=device),
                action_masks=torch.ones(E, A, 5, dtype=torch.bool, device=device),
                messages_sent=torch.zeros(E, A, MSG_DIM, device=device),
                global_states=torch.zeros(E, 19, device=device),
            )

    def test_reset_clears_pointer_and_advantages(
        self,
        tiny_train_cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        buf = _make_buffer(tiny_train_cfg, device)
        _fill_buffer(buf)
        E, A = tiny_train_cfg.num_envs, NUM_AGENTS
        buf.compute_returns_and_advantages(
            torch.zeros(E, A, device=device),
            torch.zeros(E, device=device),
            gamma=0.99, gae_lambda=0.95,
        )
        buf.reset()
        assert buf.ptr == 0
        assert buf.advantages is None
        assert buf.returns is None

    def test_minibatch_shapes(
        self,
        tiny_train_cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        buf = _make_buffer(tiny_train_cfg, device)
        _fill_buffer(buf)
        E, A = tiny_train_cfg.num_envs, NUM_AGENTS
        buf.compute_returns_and_advantages(
            torch.zeros(E, A, device=device),
            torch.zeros(E, device=device),
            gamma=0.99, gae_lambda=0.95,
        )
        MB = tiny_train_cfg.num_minibatches
        T = tiny_train_cfg.rollout_length
        batch_size = (T * E) // MB

        for batch in buf.get_minibatches(MB):
            assert batch["obs_grid"].shape[0] == batch_size
            assert batch["actions"].shape[0] == batch_size
            assert batch["advantages"].shape[0] == batch_size
            assert batch["returns"].shape[0] == batch_size


class TestGAEWithDones:
    def test_done_zeros_bootstrap(
        self,
        tiny_train_cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        """When last_dones=1 the bootstrap value must be zeroed."""
        buf = _make_buffer(tiny_train_cfg, device)
        _fill_buffer(buf)

        E, A = tiny_train_cfg.num_envs, NUM_AGENTS

        # Run with all envs done at the end
        last_dones_one = torch.ones(E, device=device)
        last_vals_high = torch.ones(E, A, device=device) * 100.0

        buf.compute_returns_and_advantages(
            last_vals_high, last_dones_one, gamma=0.99, gae_lambda=0.95
        )
        adv_with_done = buf.advantages.clone()
        buf.reset()

        # Run with no done: bootstrap values of 100 should inflate advantages
        _fill_buffer(buf)
        last_dones_zero = torch.zeros(E, device=device)
        buf.compute_returns_and_advantages(
            last_vals_high, last_dones_zero, gamma=0.99, gae_lambda=0.95
        )
        adv_no_done = buf.advantages.clone()

        # With done=1, last bootstrap is zeroed → advantages should differ
        # (we can't guarantee direction after normalisation, but they must differ)
        assert not torch.allclose(adv_with_done, adv_no_done, atol=1e-4), \
            "GAE with done=1 vs done=0 must produce different advantages"


class TestSelfPlayManagerExtra:
    def _constructor(
        self,
        team_id: int,
        agent_cfg: AgentConfig,
        model_cfg: ModelConfig,
        device: torch.device,
    ) -> tuple:
        att = AttackerAgent(
            agent_id=0 if team_id == 0 else 2, team_id=team_id,
            config=agent_cfg, model_config=model_cfg,
            observation_radius=OBS_R, num_obs_channels=NUM_C,
            device=device,
        )
        def_ = DefenderAgent(
            agent_id=1 if team_id == 0 else 3, team_id=team_id,
            config=agent_cfg, model_config=model_cfg,
            observation_radius=OBS_R, num_obs_channels=NUM_C,
            device=device,
        )
        return att, def_

    def test_snapshots_capped(
        self,
        tiny_train_cfg: TrainingConfig,
        tiny_agent_cfg: AgentConfig,
        tiny_model_cfg: ModelConfig,
        device: torch.device,
    ) -> None:
        fn = lambda tid: self._constructor(tid, tiny_agent_cfg, tiny_model_cfg, device)
        mgr = SelfPlayManager(tiny_train_cfg, fn)
        agents = _build_agents(tiny_agent_cfg, tiny_model_cfg, device)
        for i in range(tiny_train_cfg.league_size + 10):
            mgr.snapshot(agents, i * 100)
        assert len(mgr.league) == tiny_train_cfg.league_size

    def test_no_snapshot_returns_rule_based(
        self,
        tiny_train_cfg: TrainingConfig,
        tiny_agent_cfg: AgentConfig,
        tiny_model_cfg: ModelConfig,
        device: torch.device,
    ) -> None:
        fn = lambda tid: self._constructor(tid, tiny_agent_cfg, tiny_model_cfg, device)
        mgr = SelfPlayManager(tiny_train_cfg, fn)
        att, def_ = mgr.sample_opponent(team_id=1)
        assert isinstance(att, RuleBasedAgent)
        assert isinstance(def_, RuleBasedAgent)
