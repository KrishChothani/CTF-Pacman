"""Training entry point."""

import argparse

from ctf_pacman.utils.config import load_config
from ctf_pacman.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CTF-Pacman multi-agent RL agents."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (overrides config.seed)."
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Experiment name override (overrides config.logging.experiment_name)."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
