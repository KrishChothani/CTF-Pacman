"""Logging utility with TensorBoard support."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from ctf_pacman.utils.config import LoggingConfig


class Logger:
    """Structured logger with optional TensorBoard integration.

    Writes scalar metrics to TensorBoard and maintains an in-memory history
    list for post-hoc analysis.

    Args:
        config: LoggingConfig instance controlling paths and switches.
    """

    def __init__(self, config: LoggingConfig) -> None:
        self.config = config
        self.history: List[dict] = []
        self._writer = None

        # Create log directory
        log_path = os.path.join(config.log_dir, config.experiment_name)
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._writer = SummaryWriter(log_dir=log_path)
            except ImportError:
                print(
                    "[Logger] WARNING: tensorboard not installed. "
                    "Falling back to in-memory logging only.",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Write a dict of scalar metrics to TensorBoard and history.

        Args:
            metrics: Dict mapping metric name to float value.
            step:    Global training timestep used as the x-axis.
        """
        record = {"step": step, **metrics}
        self.history.append(record)
        if self._writer is not None:
            for key, val in metrics.items():
                self._writer.add_scalar(key, val, global_step=step)

    def log_episode(self, episode_info: dict, step: int) -> None:
        """Log per-episode statistics.

        Args:
            episode_info: Dict of episode-level stats (returns, lengths, etc.).
            step:         Current training step.
        """
        prefixed = {f"episode/{k}": v for k, v in episode_info.items()}
        self.log_scalars(prefixed, step)

    def print_summary(self, step: int, extra: Optional[dict] = None) -> None:
        """Print a formatted summary table to stdout.

        Args:
            step:  Current training step.
            extra: Optional extra metrics to display.
        """
        recent = [r for r in self.history if r["step"] <= step]
        if not recent:
            return

        last = recent[-1]
        divider = "=" * 60
        print(divider)
        print(f"  Step: {step:,}")
        for key, val in last.items():
            if key == "step":
                continue
            if isinstance(val, float):
                print(f"  {key:<35} {val:.6f}")
            else:
                print(f"  {key:<35} {val}")
        if extra:
            for key, val in extra.items():
                if isinstance(val, float):
                    print(f"  {key:<35} {val:.6f}")
                else:
                    print(f"  {key:<35} {val}")
        print(divider)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
