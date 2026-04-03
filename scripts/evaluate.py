"""Evaluation entry point."""

import argparse

from ctf_pacman.environment.env import CTFPacmanEnv
from ctf_pacman.game_engine import GameEngine
from ctf_pacman.training.trainer import Trainer
from ctf_pacman.utils.config import load_config
from ctf_pacman.utils.metrics import MetricsAggregator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CTF-Pacman checkpoint."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pt checkpoint file produced by training."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of evaluation episodes (default: 100)."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render each episode step as ASCII art."
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use greedy (deterministic) action selection (default: True)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env = CTFPacmanEnv(config.env, seed=config.seed)

    # Build trainer object (without calling __init__ to skip env/buffer setup)
    trainer = Trainer.__new__(Trainer)
    trainer.config = config

    import torch
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.agents = trainer._build_agents()
    trainer.load_checkpoint(args.checkpoint)

    engine = GameEngine(env, trainer.agents, config)
    agg = MetricsAggregator()

    for i in range(args.num_episodes):
        metrics = engine.run_episode(render=args.render, deterministic=args.deterministic)
        agg.add(metrics)
        result = "WIN" if metrics.win == 1 else ("LOSS" if metrics.win == -1 else "DRAW")
        print(
            f"Episode {i + 1:>4}/{args.num_episodes} | "
            f"{result} | "
            f"Score: {metrics.score_team0}-{metrics.score_team1} | "
            f"Length: {metrics.episode_length}"
        )

    summary = agg.summarize()
    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<45} {v:.4f}")
        else:
            print(f"  {k:<45} {v}")


if __name__ == "__main__":
    main()
