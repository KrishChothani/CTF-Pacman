# Capture the Flag Pacman (CTF-Pacman)

A complete, research-grade multi-agent reinforcement learning project implementing Capture the Flag in a Pacman-style grid environment.

## Overview

CTF-Pacman is a competitive multi-agent RL environment where two teams of agents (each with one attacker and one defender) compete to steal the opponent's food while protecting their own. It features:

- **Multi-agent PPO** with per-role specialization (attacker/defender)
- **Intra-team communication** via learned message passing
- **Centralized critic** with global state access during training
- **Self-play training** with a league of historical opponents
- **Procedurally generated** symmetric maps

## Project Structure

```
ctf_pacman/
├── configs/             # YAML configuration files
├── ctf_pacman/
│   ├── environment/     # Grid, env, observations, rewards, events
│   ├── agents/          # Neural network agents + rule-based heuristic
│   ├── models/          # CNN encoder, actor/critic/message heads
│   ├── training/        # PPO, rollout buffer, self-play, trainer
│   └── utils/           # Config, logging, seeding, metrics
├── scripts/             # train.py and evaluate.py entry points
└── tests/               # Unit tests
```

## Installation

```bash
cd ctf_pacman
pip install -r requirements.txt
pip install -e .
```

## Quick Start

**Train with default config:**
```bash
python scripts/train.py --config configs/default.yaml
```

**Train with small map:**
```bash
python scripts/train.py --config configs/small_map.yaml --experiment_name small_map_run
```

**Evaluate a checkpoint:**
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/ctf_default/ckpt_100000.pt --num_episodes 100 --render
```

## Configuration

All hyperparameters are defined in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.map_width` | 32 | Grid columns |
| `env.map_height` | 16 | Grid rows |
| `env.num_food_per_team` | 20 | Food items per team |
| `training.total_timesteps` | 5,000,000 | Training steps |
| `training.num_envs` | 16 | Parallel environments |
| `training.learning_rate` | 3e-4 | Adam LR |
| `training.selfplay_update_interval` | 100,000 | Self-play snapshot interval |

## Environment Details

### Game Rules
- 4 agents total: Team 0 (agents 0,1) vs Team 1 (agents 2,3)
- Each team has 1 attacker and 1 defender
- Attackers collect opponent food and bring it home
- Defenders protect home food and chase invaders
- Power pellets temporarily scare opponents
- Captured attackers lose all carried food and respawn

### Observation Space
Each agent receives:
- `grid`: `(10, 2r+1, 2r+1)` local view with channels for walls, food, agents, power pellets, territory
- `flat`: `(8,)` scalar features (step, score diff, scared timers, distances, etc.)

### Action Space
Discrete(5): North, South, East, West, Stop

### Reward Structure
| Event | Reward |
|-------|--------|
| Food collected | +1.0 |
| Food returned home | +10.0 |
| Captured by opponent | -5.0 |
| Defender captures invader | +5.0 |
| Per step | -0.05 |
| Episode win | +4.5 |
| Episode loss | -4.5 |

## Architecture

### Neural Network
Each agent has:
1. **CNN Encoder**: 3-layer conv net over local grid observation
2. **Trunk MLP**: Fuses CNN output + flat features + teammate message
3. **Actor Head**: Policy logits with action masking
4. **Critic Head**: Value estimate using global state (centralized training)
5. **Message Head**: Generates communication vector for teammate

### Training Algorithm
- **PPO** with GAE advantage estimation
- **Self-play** with league of historical checkpoints
- **Opponent sampling**: 50% latest, 30% historical, 20% rule-based

## Reproducibility

Fixed seed ensures deterministic results:
```bash
python scripts/train.py --seed 42
```

## TensorBoard

```bash
tensorboard --logdir runs/
```

Logs: policy_loss, value_loss, entropy_loss, win_rate, food_collected, captures_made, episode_length.
