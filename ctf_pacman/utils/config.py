"""Configuration system using dataclasses with YAML serialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List
import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Environment configuration."""
    map_width: int = 32
    map_height: int = 16
    num_food_per_team: int = 20
    num_power_pellets: int = 2
    power_pellet_duration: int = 40
    max_steps: int = 300
    observation_radius: int = 5
    num_observation_channels: int = 10
    wall_density: float = 0.15
    food_respawn: bool = False


@dataclass
class AgentConfig:
    """Agent behaviour configuration."""
    message_dim: int = 8
    use_communication: bool = True
    role_switch_enabled: bool = False


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    flat_feature_dim: int = 16
    hidden_dim: int = 256
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    message_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """PPO and self-play training configuration."""
    total_timesteps: int = 5_000_000
    num_envs: int = 16
    rollout_length: int = 128
    num_ppo_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    entropy_coeff_schedule: str = "linear"
    max_grad_norm: float = 0.5
    checkpoint_interval: int = 50_000
    selfplay_update_interval: int = 100_000
    league_size: int = 10
    latest_opponent_fraction: float = 0.5
    historical_opponent_fraction: float = 0.3
    rule_based_opponent_fraction: float = 0.2


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_dir: str = "runs/"
    experiment_name: str = "ctf_default"
    log_interval: int = 1000
    tensorboard: bool = True
    print_interval: int = 5000


@dataclass
class Config:
    """Top-level configuration object."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_dict_into_dataclass(dc_instance, d: dict):
    """Recursively merge a nested dict *d* into an existing dataclass instance."""
    for key, val in d.items():
        if not hasattr(dc_instance, key):
            raise ValueError(f"Unknown config key: '{key}'")
        current = getattr(dc_instance, key)
        if isinstance(current, (EnvConfig, AgentConfig, ModelConfig, TrainingConfig, LoggingConfig)):
            if isinstance(val, dict):
                _merge_dict_into_dataclass(current, val)
            else:
                raise TypeError(f"Expected dict for sub-config '{key}', got {type(val)}")
        else:
            setattr(dc_instance, key, val)


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert a dataclass (or list/primitive) to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(i) for i in obj]
    else:
        return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str) -> Config:
    """Load a YAML file and merge it into the default Config dataclass.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Fully populated Config instance with YAML values merged over defaults.
    """
    config = Config()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw:
        # Separate top-level non-dict fields (e.g. seed)
        sub_keys = {"env", "agent", "model", "training", "logging"}
        for key, val in raw.items():
            if key in sub_keys:
                if isinstance(val, dict):
                    _merge_dict_into_dataclass(getattr(config, key), val)
                else:
                    raise TypeError(f"Expected dict for section '{key}', got {type(val)}")
            else:
                if hasattr(config, key):
                    setattr(config, key, val)
                else:
                    raise ValueError(f"Unknown top-level config key: '{key}'")
    return config


def save_config(config: Config, path: str) -> None:
    """Serialize a Config dataclass to a YAML file.

    Args:
        config: Config instance to serialize.
        path:   Destination file path.
    """
    raw = _dataclass_to_dict(config)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)
